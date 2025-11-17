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
import os       # NEW: 导入 os 模块
import argparse # NEW: 导入 argparse 模块

# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        self.processed_data_dir = 'processed_data'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 模型参数
        self.embedding_dim = 64
        self.n_layers = 3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # 训练参数
        self.batch_size = 2048
        self.epochs = 20
        self.top_k = 20
        
        # NEW: Checkpoint 相关参数
        self.checkpoint_dir = 'checkpoints'
        self.best_model_name = 'best_model.pth'

# --- 2. 数据加载与预处理 ---
def load_preprocessed_data(data_dir, device):
    print("Step 1: Loading preprocessed data...")
    stats_path = os.path.join(data_dir, 'stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError("Preprocessed data not found. Please run 'prepare_data.py' first.")
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    num_users, num_items, num_brands = stats['num_users'], stats['num_items'], stats['num_brands']
    print(f"Stats: {num_users} users, {num_items} items, {num_brands} brands.")

    train_data = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_data = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    item_brand_data = pd.read_parquet(os.path.join(data_dir, 'item_brand.parquet'))
    print(f"Loaded {len(train_data)} training and {len(test_data)} testing interactions.")

    num_nodes = num_users + num_items + num_brands
    adj_mat = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32).tolil()

    # User-Item
    R_user_item = sp.coo_matrix((np.ones(len(train_data)), (train_data['user_idx'], train_data['item_idx'])),
                                shape=(num_users, num_items)).tolil()
    item_offset = num_users
    brand_offset = num_users + num_items
    adj_mat[:item_offset, item_offset:brand_offset] = R_user_item
    adj_mat[item_offset:brand_offset, :item_offset] = R_user_item.T

    # Item-Brand
    R_item_brand = sp.coo_matrix((np.ones(len(item_brand_data)), (item_brand_data['item_idx'], item_brand_data['brand_idx'])),
                                 shape=(num_items, num_brands)).tolil()
    adj_mat[item_offset:brand_offset, brand_offset:] = R_item_brand
    adj_mat[brand_offset:, item_offset:brand_offset] = R_item_brand.T
    
    adj_mat = adj_mat.todok()
    print("Heterogeneous adjacency matrix created.")
    
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj_mat.shape)).to(device)
    
    return train_data, test_data, num_users, num_items, num_brands, norm_adj_tensor

# --- 3. PyTorch Dataset ---
class BPRDataset(Dataset):
    # ... (代码保持不变) ...
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
class LightGCN_Hetero(nn.Module):
    # ... (代码保持不变) ...
    def __init__(self, num_users, num_items, num_brands, config):
        super(LightGCN_Hetero, self).__init__()
        self.num_users, self.num_items, self.num_brands = num_users, num_items, num_brands
        self.embedding_dim, self.n_layers = config.embedding_dim, config.n_layers
        
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        self.brand_embedding = nn.Embedding(num_brands, self.embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.brand_embedding.weight)

    def forward(self, adj_mat):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight, self.brand_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        final_user_embeddings, final_item_embeddings, _ = torch.split(final_embeddings, [self.num_users, self.num_items, self.num_brands])
        return final_user_embeddings, final_item_embeddings

# --- 5. BPR Loss & 评估函数 ---
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    # ... (代码保持不变) ...
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

def evaluate(model, test_data, train_data, all_user_emb, all_item_emb, k, device):
    # ... (代码保持不变) ...
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

# --- 6. NEW: 分离出训练和测试函数 ---

def train(config):
    """完整的模型训练流程"""
    # 加载数据
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)
    
    train_dataset = BPRDataset(train_df, num_items)
    if len(train_dataset) == 0:
        print("Warning: No training data available. Exiting.")
        return
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型
    model = LightGCN_Hetero(num_users, num_items, num_brands, config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 训练循环与模型保存
    best_recall = 0.0
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
        
    print("\nStep 2: Starting model training...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for users, pos_items, neg_items in progress_bar:
            users, pos_items, neg_items = users.to(config.device), pos_items.to(config.device), neg_items.to(config.device)
            optimizer.zero_grad()
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            batch_user_emb, batch_pos_item_emb, batch_neg_item_emb = all_user_emb[users], all_item_emb[pos_items], all_item_emb[neg_items]
            loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 定期评估并保存最佳模型
        if (epoch + 1) % 2 == 0:
            print("Evaluating...")
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
            print(f"Epoch {epoch+1} | Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

            if recall > best_recall:
                best_recall = recall
                save_path = os.path.join(config.checkpoint_dir, config.best_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with Recall@{config.top_k}: {best_recall:.4f} to {save_path}")

    print("Training finished.")

def test(config, model_path):
    """加载已训练模型并进行测试"""
    print("--- Starting Testing Mode ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    # 加载数据
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)

    # 初始化模型结构
    model = LightGCN_Hetero(num_users, num_items, num_brands, config).to(config.device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"Model loaded from {model_path}")

    # 运行评估
    print("Evaluating on the test set...")
    all_user_emb, all_item_emb = model(norm_adj_tensor)
    recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
    
    print("\n--- Test Results ---")
    print(f"Recall@{config.top_k}: {recall:.4f}")
    print(f"NDCG@{config.top_k}:   {ndcg:.4f}")
    print("--------------------")

# --- 7. 主入口 ---
if __name__ == '__main__':
    # NEW: 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Run LightGCN_Hetero model for recommendation.")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode to run the script in: 'train' or 'test'")
    parser.add_argument('--model_path', type=str, help="Path to the model checkpoint for testing.")
    
    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 初始化配置
    config = Config()

    if args.mode == 'train':
        train(config)
    elif args.mode == 'test':
        # 如果没有指定模型路径，则使用默认的最佳模型路径
        model_to_test = args.model_path if args.model_path else os.path.join(config.checkpoint_dir, config.best_model_name)
        test(config, model_to_test)