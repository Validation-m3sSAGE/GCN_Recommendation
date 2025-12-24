import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN_Fusion(nn.Module):
    def __init__(self, num_users, num_items, num_brands, config, pretrained_item_emb=None):
        super(LightGCN_Fusion, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_brands = num_brands
        self.embedding_dim = config.embedding_dim
        self.n_layers = config.n_layers
        
        if pretrained_item_emb is None:
            raise ValueError("LightGCN_Fusion model requires pretrained item embeddings.")
            
        content_emb_dim = pretrained_item_emb.shape[1]
        
        # 1. 可学习的协同ID嵌入 (随机初始化)
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, self.embedding_dim)#nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_item_emb), freeze=False)#
        self.brand_embedding = nn.Embedding(num_brands, self.embedding_dim)
        
        # 2. 固定的内容嵌入 (作为 buffer, 不参与训练)
        self.register_buffer('item_content_embedding', torch.FloatTensor(pretrained_item_emb))
        
        # 3. 融合层
        # 将拼接后的 (embedding_dim + content_emb_dim) 映射回 embedding_dim
        self.item_fusion_layer = nn.Linear(self.embedding_dim + content_emb_dim, self.embedding_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight)
        nn.init.xavier_uniform_(self.item_fusion_layer.weight)

    def forward(self, adj_mat, use_brand=True):
        # 初始嵌入
        user_emb_0 = self.user_embedding.weight
        item_id_emb_0 = self.item_id_embedding.weight
        brand_emb_0 = self.brand_embedding.weight
        
        # --- 核心修改：在GNN传播前进行特征融合 ---
        # 拼接可学习的ID嵌入和固定的内容嵌入
        combined_item_emb_0 = torch.cat([item_id_emb_0, self.item_content_embedding], dim=1)
        # 通过融合层
        fused_item_emb_0 = self.item_fusion_layer(combined_item_emb_0)
        # (可选) 可以加入激活函数和 Dropout
        fused_item_emb_0 = F.leaky_relu(fused_item_emb_0)
        # --- END ---
        
        ego_embeddings = torch.cat([user_emb_0, fused_item_emb_0, brand_emb_0], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        final_user_emb, final_item_emb, final_brand_emb = torch.split(
            final_embeddings, [self.num_users, self.num_items, self.num_brands])
        
        # 返回值需要适配 bpr_loss_reg，initial_item_emb 应该是可学习的 id_emb
        return final_user_emb, final_item_emb, final_brand_emb, user_emb_0, item_id_emb_0