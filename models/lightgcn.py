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
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight)

    def forward(self, adj_mat):
        # 拼接所有类型的嵌入
        ego_embeddings = torch.cat([self.user_embedding.weight, 
                                    self.item_embedding.weight,
                                    self.brand_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # GNN 传播
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # 聚合各层嵌入
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        # 拆分嵌入以获取最终的用户和物品表征
        final_user_embeddings, final_item_embeddings, _ = torch.split(
            final_embeddings, [self.num_users, self.num_items, self.num_brands])
        
        return final_user_embeddings, final_item_embeddings