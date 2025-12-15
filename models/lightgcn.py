# models/lightgcn.py

import torch
import torch.nn as nn

class LightGCN(nn.Module):
    """
    基于 LightGCN 思想的异构图推荐模型。
    处理用户、物品、品牌三种类型的节点。
    """
    def __init__(self, num_users, num_items, num_brands, config):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_brands = num_brands
        self.embedding_dim = config.embedding_dim
        self.n_layers = config.n_layers
        
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        self.brand_embedding = nn.Embedding(num_brands, self.embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight)
    #'''
    def forward(self, adj_mat):
        # 初始嵌入
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        brand_emb_0 = self.brand_embedding.weight
        
        ego_embeddings = torch.cat([user_emb_0, item_emb_0, brand_emb_0], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        final_user_emb, final_item_emb, _ = torch.split(
            final_embeddings, [self.num_users, self.num_items, self.num_brands])
        
        # 返回最终嵌入和初始嵌入，用于修复计算图断裂问题
        return final_user_emb, final_item_emb, user_emb_0, item_emb_0
    '''
    def forward(self, adj_mat, use_brand=True):  # 新增use_brand参数，控制是否验证品牌
        # 初始嵌入
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        brand_emb_0 = self.brand_embedding.weight
        
        # 拼接初始嵌入（仅当use_brand=True时包含品牌）
        if use_brand:
            ego_embeddings = torch.cat([user_emb_0, item_emb_0, brand_emb_0], dim=0)
        else:
            ego_embeddings = torch.cat([user_emb_0, item_emb_0], dim=0)
        all_embeddings = [ego_embeddings]
        
        # GNN传播
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
            # 仅在use_brand=True时打印品牌嵌入范数
            if use_brand:
                brand_emb_i = ego_embeddings[self.num_users+self.num_items:]
                print(f"Layer {i+1} brand embedding L2 norm: {brand_emb_i.norm(2).item():.6f}")
        
        # 计算最终嵌入（平均各层）
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        # 拆分最终嵌入
        if use_brand:
            final_user_emb, final_item_emb, final_brand_emb = torch.split(
                final_embeddings, [self.num_users, self.num_items, self.num_brands])
        else:
            final_user_emb, final_item_emb = torch.split(
                final_embeddings, [self.num_users, self.num_items])
            final_brand_emb = None
        
        # 仅在use_brand=True时验证商品嵌入的品牌贡献（核心修复：避免稀疏张量切片）
        if use_brand:
            # 随机选100个商品（固定种子，保证可复现）
            torch.manual_seed(42)
            random_item_idx = torch.randint(0, self.num_items, (100,)).to(user_emb_0.device)
            item_emb_with_brand = final_item_emb[random_item_idx]
            
            # 修复：稀疏张量无法切片，先转为稠密张量（Debug模式数据量小，可行）
            adj_dense = adj_mat.to_dense()  # 转为稠密矩阵
            # 仅取用户-商品子图的稠密矩阵
            adj_user_item = adj_dense[:self.num_users+self.num_items, :self.num_users+self.num_items]
            # 计算无品牌时的商品嵌入（仅用户-商品传播）
            ego_no_brand = torch.cat([user_emb_0, item_emb_0], dim=0)
            ego_no_brand = torch.matmul(adj_user_item, ego_no_brand)  # 稠密矩阵乘法
            item_emb_no_brand = ego_no_brand[self.num_users:self.num_users+self.num_items][random_item_idx]
            item_emb_no_brand = item_emb_0[random_item_idx] + item_emb_no_brand  # 匹配原逻辑
            
            # 计算余弦相似度（确保张量在同一设备）
            cos_sim = torch.nn.functional.cosine_similarity(
                item_emb_with_brand, item_emb_no_brand, dim=1
            ).mean()
            print(f"Average cos similarity (item emb with/without brand): {cos_sim.item():.6f}")
        
        # 返回最终嵌入（兼容原代码的返回格式）
        return final_user_emb, final_item_emb, user_emb_0, item_emb_0
    #'''