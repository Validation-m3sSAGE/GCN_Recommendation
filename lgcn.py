# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        self.review_file = 'dataset/Books.jsonl'
        self.metadata_file = 'dataset/meta_Books.jsonl' # NEW: 添加元数据文件路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 数据预处理参数
        self.min_user_interactions = 1 # 使用一个合理的值
        self.min_item_interactions = 1 # 使用一个合理的值
        self.max_lines_to_load = 5000 # 可以根据你的硬件调整

        # 模型参数
        self.embedding_dim = 64
        self.n_layers = 3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # 训练参数
        self.batch_size = 2048
        self.epochs = 20
        self.top_k = 20

config = Config()

# --- 2. 数据加载与预处理 ---
# MODIFIED: 优化了元数据的加载过程
def load_and_preprocess_data(config):
    """加载、过滤、并构建包含辅助信息的异构图"""
    print("Step 1: Loading and preprocessing data...")
    
    # 1. 加载和过滤评论数据 (与之前类似)
    reviews = []
    with open(config.review_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= config.max_lines_to_load:
                break
            reviews.append(json.loads(line.strip()))
    
    df = pd.DataFrame(reviews)[['user_id', 'asin', 'rating']]
    df.rename(columns={'asin': 'item_id'}, inplace=True)
    print(f"Loaded {len(df)} interactions initially.")

    # K-core filtering
    iteration = 0
    while True:
        iteration += 1
        initial_interactions = len(df)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        weak_users = user_counts[user_counts < config.min_user_interactions].index
        weak_items = item_counts[item_counts < config.min_item_interactions].index
        if len(weak_users) == 0 and len(weak_items) == 0: break
        df = df[~df['user_id'].isin(weak_users)]
        df = df[~df['item_id'].isin(weak_items)]
        if len(df) == initial_interactions: break

    print(f"\nFinal filtered data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items.")
    if df.empty:
        raise ValueError("DataFrame is empty after filtering.")

    # ------------------- MODIFICATION START -------------------
    # 2. NEW & OPTIMIZED: 按需加载和处理元数据
    
    # 首先，获取所有活跃物品的ID，放入一个Set中以便快速查找
    active_items_set = set(df['item_id'].unique())
    print(f"Loading metadata for {len(active_items_set)} active items...")
    
    meta_data = {}
    with open(config.metadata_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            # 只有当该物品在我们活跃物品集合中时，才处理它
            if record.get('parent_asin') in active_items_set:
                brand = record.get('details', {}).get('Brand', 'Unknown')
                meta_data[record['parent_asin']] = brand
    
    print(f"Successfully loaded metadata for {len(meta_data)} items.")

    # 只保留过滤后还存在的物品的元数据 (这部分逻辑可以简化)
    active_items_list = list(df['item_id'].unique())
    item_brand_df = pd.DataFrame({
        'item_id': active_items_list,
        'brand': [meta_data.get(item_id, 'Unknown') for item_id in active_items_list]
    })
    # -------------------- MODIFICATION END --------------------
    
    # 3. 创建所有节点的ID映射 (逻辑不变)
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    brand_map = {id: i for i, id in enumerate(item_brand_df['brand'].unique())}

    num_users = len(user_map)
    num_items = len(item_map)
    num_brands = len(brand_map)
    print(f"Found {num_brands} unique brands.")

    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    item_brand_df['item_idx'] = item_brand_df['item_id'].map(item_map)
    item_brand_df['brand_idx'] = item_brand_df['brand'].map(brand_map)
    
    # 4. 划分训练/测试集 (逻辑不变)
    df['rank_latest'] = df.groupby(['user_idx'])['rating'].rank(method='first', ascending=False)
    test_data = df[df['rank_latest'] == 1]
    train_data = df[df['rank_latest'] > 1]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # 5. 构建异构图邻接矩阵 (逻辑不变)
    num_nodes = num_users + num_items + num_brands
    adj_mat = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    # a. User-Item 关系
    train_user_indices = train_data['user_idx'].values
    train_item_indices = train_data['item_idx'].values
    R_user_item = sp.coo_matrix((np.ones(len(train_user_indices)), (train_user_indices, train_item_indices)),
                                shape=(num_users, num_items)).tolil()
    
    user_offset = 0
    item_offset = num_users
    brand_offset = num_users + num_items

    adj_mat[user_offset:item_offset, item_offset:brand_offset] = R_user_item
    adj_mat[item_offset:brand_offset, user_offset:item_offset] = R_user_item.T

    # b. Item-Brand 关系
    # 过滤掉没有有效item_idx或brand_idx的行，以防万一
    item_brand_df_clean = item_brand_df.dropna(subset=['item_idx', 'brand_idx'])
    item_brand_indices = item_brand_df_clean['item_idx'].astype(int).values
    brand_indices = item_brand_df_clean['brand_idx'].astype(int).values
    
    R_item_brand = sp.coo_matrix((np.ones(len(item_brand_indices)), (item_brand_indices, brand_indices)),
                                 shape=(num_items, num_brands)).tolil()

    adj_mat[item_offset:brand_offset, brand_offset:] = R_item_brand
    adj_mat[brand_offset:, item_offset:brand_offset] = R_item_brand.T
    
    adj_mat = adj_mat.todok()
    print("Heterogeneous adjacency matrix created.")
    
    # 6. 归一化邻接矩阵 (逻辑不变)
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj_mat.shape)).to(config.device)
    
    return train_data.drop(columns=['rank_latest']), test_data.drop(columns=['rank_latest']), \
           num_users, num_items, num_brands, norm_adj_tensor

# --- 3. PyTorch Dataset ---
# (BPRDataset 保持不变)
class BPRDataset(Dataset):
    def __init__(self, df, num_items):
        self.users = df['user_idx'].values
        self.pos_items = df['item_idx'].values
        self.num_items = num_items
        self.user_pos_items = df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if user not in self.user_pos_items or neg_item not in self.user_pos_items[user]:
                break
        return user, pos_item, neg_item

# --- 4. LightGCN 模型 ---
# MODIFIED: 模型已更新以处理异构图
class LightGCN_Hetero(nn.Module):
    def __init__(self, num_users, num_items, num_brands, config):
        super(LightGCN_Hetero, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_brands = num_brands # NEW
        self.embedding_dim = config.embedding_dim
        self.n_layers = config.n_layers
        
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        self.brand_embedding = nn.Embedding(num_brands, self.embedding_dim) # NEW
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight) # NEW

    def forward(self, adj_mat):
        # MODIFIED: 拼接所有类型的嵌入
        ego_embeddings = torch.cat([self.user_embedding.weight, 
                                    self.item_embedding.weight,
                                    self.brand_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # GNN 传播 (逻辑不变)
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        # MODIFIED: 拆分嵌入以获取最终的用户和物品表征
        final_user_embeddings, final_item_embeddings, _ = torch.split(
            final_embeddings, [self.num_users, self.num_items, self.num_brands])
        
        return final_user_embeddings, final_item_embeddings

# --- 5. BPR Loss ---
# (bpr_loss 保持不变)
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    return loss

# --- 6. 评估函数 ---
# (evaluate 保持不变)
def evaluate(model, test_data, train_data, all_user_emb, all_item_emb, k, device):
    model.eval()
    test_users = torch.LongTensor(test_data['user_idx'].unique()).to(device)
    recalls, ndcgs = [], []
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()

    with torch.no_grad():
        for user_idx in test_users:
            user_emb = all_user_emb[user_idx]
            scores = torch.matmul(user_emb, all_item_emb.T)
            if user_idx.item() in train_user_items:
                exclude_indices = train_user_items[user_idx.item()]
                scores[exclude_indices] = -1e10
            _, top_k_indices = torch.topk(scores, k=k)
            true_item_idx = test_data[test_data['user_idx'] == user_idx.item()]['item_idx'].values[0]
            hit = (true_item_idx in top_k_indices.cpu().numpy())
            recalls.append(1 if hit else 0)
            if hit:
                position = (top_k_indices.cpu().numpy() == true_item_idx).nonzero()[0][0]
                ndcgs.append(1 / np.log2(position + 2))
            else:
                ndcgs.append(0)
    return np.mean(recalls), np.mean(ndcgs)

# --- 7. 主训练流程 ---
# MODIFIED: 更新了函数调用
def main():
    # MODIFIED: 数据加载函数返回更多值
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = load_and_preprocess_data(config)
    
    # 创建Dataset和DataLoader
    train_dataset = BPRDataset(train_df, num_items)
    # Handle empty train_dataset case
    if len(train_dataset) == 0:
        print("Warning: No training data available after split. Exiting.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # MODIFIED: 初始化新的模型
    model = LightGCN_Hetero(num_users, num_items, num_brands, config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("\nStep 2: Starting model training with heterogeneous graph...")
    # (训练循环逻辑保持不变)
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for users, pos_items, neg_items in progress_bar:
            users, pos_items, neg_items = users.to(config.device), pos_items.to(config.device), neg_items.to(config.device)
            optimizer.zero_grad()
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            batch_user_emb = all_user_emb[users]
            batch_pos_item_emb = all_item_emb[pos_items]
            batch_neg_item_emb = all_item_emb[neg_items]
            loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            print("Evaluating...")
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
            print(f"Epoch {epoch+1} | Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    import scipy.sparse as sp
    main()