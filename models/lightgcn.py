import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_brands, config, pretrained_item_emb=None):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_brands = num_brands
        self.embedding_dim = config.embedding_dim
        self.n_layers = config.n_layers
        self.debug = config.debug
        
        # --- CORE MODIFICATION: Conditional Initialization ---
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.brand_embedding = nn.Embedding(num_brands, self.embedding_dim) # Renamed to aux_embedding internally
        
        if pretrained_item_emb is not None:
            print("INFO: Initializing item embeddings from pretrained file.")
            # 检查维度是否匹配
            if pretrained_item_emb.shape[1] != self.embedding_dim:
                raise ValueError(f"Pretrained embedding dim ({pretrained_item_emb.shape[1]}) does not match model embedding dim ({self.embedding_dim}).")
            self.item_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_item_emb), freeze=False)
        else:
            print("INFO: Randomly initializing item embeddings.")
            self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
            nn.init.xavier_uniform_(self.item_embedding.weight)
        # --- END MODIFICATION ---

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight)
        
        self.final_brand_emb = None

    def forward(self, adj_mat, use_brand=True):
        # 【核心修改】始终拼接品牌嵌入（维度固定为 num_users+num_items+num_brands）
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        brand_emb_0 = self.brand_embedding.weight
        ego_embeddings = torch.cat([user_emb_0, item_emb_0, brand_emb_0], dim=0)
        all_embeddings = [ego_embeddings]
        
        # GNN核心传播（无维度不匹配问题）
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
            # 仅 debug+use_brand=True 时打印品牌范数（非debug/无品牌时跳过）
            if self.debug:
                brand_emb_i = ego_embeddings[self.num_users+self.num_items:]
                print(f"Layer {i+1} brand embedding L2 norm: {brand_emb_i.norm(2).item():.6f}")
        
        # 最终嵌入计算（平均各层）
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        # 始终拆分出品牌嵌入（即使use_brand=False，也不影响）
        final_user_emb, final_item_emb, final_brand_emb = torch.split(
            final_embeddings, [self.num_users, self.num_items, self.num_brands]
        )
        
        # 仅 debug+use_brand=True 时执行验证逻辑（非debug/无品牌时跳过）
        if self.debug:
            torch.manual_seed(42)
            random_item_idx = torch.randint(0, self.num_items, (100,)).to(user_emb_0.device)
            item_emb_with_brand = final_item_emb[random_item_idx]
            
            # 稀疏转稠密仅debug模式执行
            adj_dense = adj_mat.to_dense()
            adj_user_item = adj_dense[:self.num_users+self.num_items, :self.num_users+self.num_items]
            ego_no_brand = torch.cat([user_emb_0, item_emb_0], dim=0)
            ego_no_brand = torch.matmul(adj_user_item, ego_no_brand)
            item_emb_no_brand = ego_no_brand[self.num_users:self.num_users+self.num_items][random_item_idx]
            item_emb_no_brand = item_emb_0[random_item_idx] + item_emb_no_brand
            
            cos_sim = torch.nn.functional.cosine_similarity(
                item_emb_with_brand, item_emb_no_brand, dim=1
            ).mean()
            print(f"Average cos similarity (item emb with/without brand): {cos_sim.item():.6f}")
        
        # 返回值兼容原有逻辑（final_brand_emb在use_brand=False时无贡献，但不影响）
        return final_user_emb, final_item_emb, final_brand_emb, user_emb_0, item_emb_0