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
        self.review_file = 'dataset/Books.jsonl' # 修改为你的评论文件名
        #self.metadata_file = 'dataset/meta_Books.jsonl' # 修改为你的元数据文件名
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 数据预处理参数
        self.min_user_interactions = 1 # 至少有1次交互的用户
        self.min_item_interactions = 1 # 至少被1个用户交互过的物品
        self.max_lines_to_load = 5000 # 由于数据集很大，先加载前50万行进行测试

        # 模型参数
        self.embedding_dim = 64
        self.n_layers = 3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # 训练参数
        self.batch_size = 2048
        self.epochs = 20
        self.top_k = 20 # for evaluation

config = Config()

# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(config):
    """加载、过滤和预处理数据"""
    print("Step 1: Loading and preprocessing data...")
    
    reviews = []
    with open(config.review_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= config.max_lines_to_load:
                break
            reviews.append(json.loads(line.strip()))
    
    df = pd.DataFrame(reviews)[['user_id', 'asin', 'rating']]
    df.rename(columns={'asin': 'item_id'}, inplace=True)
    
    print(f"Loaded {len(df)} interactions initially.")

    # K-core filtering with logging
    iteration = 0
    while True:
        iteration += 1
        print(f"\nFiltering iteration {iteration}...")
        
        initial_interactions = len(df)
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        weak_users = user_counts[user_counts < config.min_user_interactions].index
        weak_items = item_counts[item_counts < config.min_item_interactions].index
        
        print(f"Found {len(weak_users)} weak users and {len(weak_items)} weak items.")
        
        if len(weak_users) == 0 and len(weak_items) == 0:
            print("No more weak users or items to remove. Filtering complete.")
            break
            
        df = df[~df['user_id'].isin(weak_users)]
        df = df[~df['item_id'].isin(weak_items)]

        print(f"Removed interactions. New count: {len(df)} (from {initial_interactions})")

        if len(df) == initial_interactions:
            # If no interactions were removed, break to avoid infinite loop on edge cases
            print("No change in interaction count. Stopping filtering.")
            break

    print(f"\nFinal filtered data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items.")

    if df.empty:
        raise ValueError("DataFrame is empty after filtering. Try lowering filtering thresholds or loading more data.")

    # 创建用户和物品的ID映射
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    num_users = len(user_map)
    num_items = len(item_map)

    # 划分训练集和测试集 (留一法)
    # 使用sort_values对每个用户进行分组并获取最后一个交互作为测试集
    # 这里我们假设没有时间戳，就按pandas内部顺序；如果有时间戳会更准确
    df['rank_latest'] = df.groupby(['user_idx'])['rating'].rank(method='first', ascending=False)
    test_data = df[df['rank_latest'] == 1]
    train_data = df[df['rank_latest'] > 1]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # 构建稀疏邻接矩阵
    train_user_indices = train_data['user_idx'].values
    train_item_indices = train_data['item_idx'].values

    R = sp.coo_matrix((np.ones(len(train_user_indices)), (train_user_indices, train_item_indices)),
                      shape=(num_users, num_items), dtype=np.float32)

    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    adj_mat[:num_users, num_users:] = R
    adj_mat[num_users:, :num_users] = R.T
    adj_mat = adj_mat.todok()
    print("Adjacency matrix created.")

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    # 修复 UserWarning: 使用推荐的 torch.sparse_coo_tensor
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj_mat.shape)).to(config.device)
    
    return train_data.drop(columns=['rank_latest']), test_data.drop(columns=['rank_latest']), num_users, num_items, norm_adj_tensor

# --- 3. PyTorch Dataset ---
class BPRDataset(Dataset):
    def __init__(self, df, num_items):
        self.users = df['user_idx'].values
        self.pos_items = df['item_idx'].values
        self.num_items = num_items
        
        # 预先计算每个用户的正样本
        self.user_pos_items = df.groupby('user_idx')['item_idx'].apply(list).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos_items[user]:
                break
        
        return user, pos_item, neg_item

# --- 4. LightGCN 模型 ---
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, config):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = config.embedding_dim
        self.n_layers = config.n_layers
        
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_mat):
        # 拼接用户和物品的初始嵌入
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # GNN 传播
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # 聚合各层嵌入 (mean pooling)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        
        # 分离用户和物品嵌入
        final_user_embeddings, final_item_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items])
        
        return final_user_embeddings, final_item_embeddings

# --- 5. BPR Loss ---
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    return loss

# --- 6. 评估函数 ---
def evaluate(model, test_data, train_data, all_user_emb, all_item_emb, k, device):
    model.eval()
    
    test_users = torch.LongTensor(test_data['user_idx'].unique()).to(device)
    
    recalls = []
    ndcgs = []

    # 获取训练数据中的交互，用于评估时过滤
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()

    with torch.no_grad():
        for user_idx in test_users:
            user_emb = all_user_emb[user_idx]
            scores = torch.matmul(user_emb, all_item_emb.T)
            
            # 过滤掉训练集中已经交互过的物品
            if user_idx.item() in train_user_items:
                exclude_indices = train_user_items[user_idx.item()]
                scores[exclude_indices] = -1e10 # 设置为极小值
                
            # 获取 Top-K 推荐
            _, top_k_indices = torch.topk(scores, k=k)
            
            # 获取测试集中的真实交互物品
            true_item_idx = test_data[test_data['user_idx'] == user_idx.item()]['item_idx'].values[0]
            
            # 计算 Recall@K
            hit = (true_item_idx in top_k_indices.cpu().numpy())
            recalls.append(1 if hit else 0)
            
            # 计算 NDCG@K
            if hit:
                position = (top_k_indices.cpu().numpy() == true_item_idx).nonzero()[0][0]
                ndcgs.append(1 / np.log2(position + 2))
            else:
                ndcgs.append(0)

    return np.mean(recalls), np.mean(ndcgs)

# --- 7. 主训练流程 ---
def main():
    # 数据加载和预处理
    train_df, test_df, num_users, num_items, norm_adj_tensor = load_and_preprocess_data(config)
    
    # 创建Dataset和DataLoader
    train_dataset = BPRDataset(train_df, num_items)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型和优化器
    model = LightGCN(num_users, num_items, config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("\nStep 2: Starting model training...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for users, pos_items, neg_items in progress_bar:
            users, pos_items, neg_items = users.to(config.device), pos_items.to(config.device), neg_items.to(config.device)
            
            optimizer.zero_grad()
            
            # 获取所有用户的最终嵌入
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            
            # 获取当前batch的嵌入
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
        
        # 评估
        if (epoch + 1) % 2 == 0:
            print("Evaluating...")
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
            print(f"Epoch {epoch+1} | Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    # 为了复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 导入scipy.sparse，因为在函数内部使用
    import scipy.sparse as sp

    main()