# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from scipy import sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
import argparse
import importlib

# --- 1. 动态模型加载 ---
def get_model(model_name):
    """
    根据模型名称动态地从 models 文件夹导入模型类。
    假设文件名和类名相同（忽略大小写）。
    例如: model_name 'LightGCN' -> 导入 models/lightgcn.py 中的 LightGCN 类
    """
    try:
        module_path = f"models.{model_name.lower()}"
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import model '{model_name}' from '{module_path}'.py.")
        print(f"Please ensure that the file 'models/{model_name.lower()}.py' exists and contains a class named '{model_name}'.")
        raise e

# --- 2. 配置参数 ---
class Config:
    def __init__(self):
        self.processed_data_dir = 'dataset/amazon_books/processed_data_5'
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
        
        # Checkpoint 相关参数
        self.checkpoint_dir = 'checkpoints'
        self.best_model_name = 'best_model.pth' # 会被动态修改

# --- 3. 数据加载与预处理 ---
def load_preprocessed_data(data_dir, device):
    """从 Parquet 文件加载预处理好的数据，并高效构建邻接矩阵"""
    print("Step 1: Loading preprocessed data...")
    stats_path = os.path.join(data_dir, 'stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Preprocessed data not found in '{data_dir}'. Please run 'prepare_data.py' first.")
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    num_users, num_items, num_brands = stats['num_users'], stats['num_items'], stats['num_brands']
    print(f"Stats: {num_users} users, {num_items} items, {num_brands} brands.")

    train_data = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_data = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    item_brand_data = pd.read_parquet(os.path.join(data_dir, 'item_brand.parquet'))
    print(f"Loaded {len(train_data)} training and {len(test_data)} testing interactions.")

    # --- MODIFICATION START: 高效构建 COO 矩阵 ---
    print("Efficiently building heterogeneous adjacency matrix in COO format...")
    
    # 节点索引偏移量
    item_offset = num_users
    brand_offset = num_users + num_items
    num_nodes = num_users + num_items + num_brands

    # 1. 收集所有边的坐标
    # User-Item 边
    user_indices = train_data['user_idx'].values
    item_indices_for_user = train_data['item_idx'].values + item_offset
    
    # Item-Brand 边
    item_indices_for_brand = item_brand_data['item_idx'].values + item_offset
    brand_indices = item_brand_data['brand_idx'].values + brand_offset

    # 将所有边拼接在一起。图是无向的，所以要添加两个方向的边。
    all_rows = np.concatenate([user_indices, item_indices_for_user, 
                               item_indices_for_brand, brand_indices])
    all_cols = np.concatenate([item_indices_for_user, user_indices,
                               brand_indices, item_indices_for_brand])
    
    # 所有边的权重都是1
    all_data = np.ones(all_rows.shape[0], dtype=np.float32)

    # 2. 一次性创建 COO 矩阵
    adj_mat = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    print("Heterogeneous adjacency matrix created.")
    
    # --- MODIFICATION END ---
    
    # 3. 归一化 (这部分逻辑不变，可以安全地处理 COO 矩阵)
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj_mat.shape)).to(device)
    
    return train_data, test_data, num_users, num_items, num_brands, norm_adj_tensor

# --- 4. PyTorch Dataset ---
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

# --- 5. BPR Loss & 评估函数 ---
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

def evaluate(model, test_data, train_data, all_user_emb, all_item_emb, k, device, batch_size=2048):
    """
    高效的批处理评估函数。
    """
    model.eval()
    
    # --- 1. 预处理 Ground Truth 和 过滤项 ---
    # 将 test_data 转换为字典，实现 O(1) 查找
    test_user_items = {}
    for _, row in test_data.iterrows():
        test_user_items[int(row['user_idx'])] = int(row['item_idx'])
        
    # 获取训练数据中的交互，用于评估时过滤
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    test_users = list(test_user_items.keys())
    
    recalls = []
    ndcgs = []

    with torch.no_grad():
        # --- 2. 分批处理用户 ---
        for i in tqdm(range(0, len(test_users), batch_size), desc="Evaluating"):
            batch_users = test_users[i: i + batch_size]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # --- 3. 矩阵化计算 ---
            # (batch_size, emb_dim) @ (emb_dim, num_items) -> (batch_size, num_items)
            batch_scores = torch.matmul(all_user_emb[batch_users_tensor], all_item_emb.T)
            
            # --- 4. 批处理过滤 ---
            # 过滤掉训练集中已经交互过的物品
            for j, user_idx in enumerate(batch_users):
                if user_idx in train_user_items:
                    exclude_indices = train_user_items[user_idx]
                    batch_scores[j, exclude_indices] = -1e10 # 设置为极小值
            
            # --- 5. 批处理 Top-K 和 指标计算 ---
            _, top_k_indices = torch.topk(batch_scores, k=k)
            top_k_indices_cpu = top_k_indices.cpu().numpy() # 一次性传输回CPU

            # 获取这批用户的真实交互物品
            batch_true_items = [test_user_items[user] for user in batch_users]

            for j, user_idx in enumerate(batch_users):
                true_item = batch_true_items[j]
                pred_items = top_k_indices_cpu[j]
                
                # 计算 Recall@K
                hit = true_item in pred_items
                recalls.append(1 if hit else 0)
                
                # 计算 NDCG@K
                if hit:
                    position = np.where(pred_items == true_item)[0][0]
                    ndcgs.append(1 / np.log2(position + 2))
                else:
                    ndcgs.append(0)

    return np.mean(recalls), np.mean(ndcgs)

# --- 6. 训练和测试函数 ---
def train(config, model_class):
    """完整的模型训练流程"""
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)
    
    train_loader = DataLoader(BPRDataset(train_df, num_items), batch_size=config.batch_size, shuffle=True)
    
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    print(f"Initialized model: {model.__class__.__name__}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    best_recall = 0.0
    os.makedirs(config.checkpoint_dir, exist_ok=True)
        
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
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        check_path = os.path.join(config.checkpoint_dir, "epoch_"+str(epoch+1)+".pth")
        torch.save(model.state_dict(), check_path)
        
        if True or (epoch + 1) % 2 == 0:
            print("Evaluating...")
            all_user_emb, all_item_emb = model(norm_adj_tensor)
            recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
            print(f"Epoch {epoch+1} | Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")

            if recall > best_recall:
                best_recall = recall
                save_path = os.path.join(config.checkpoint_dir, config.best_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with Recall@{config.top_k}: {best_recall:.4f} to '{save_path}'")

    print("Training finished.")

def test(config, model_class, model_path):
    """加载已训练模型并进行测试"""
    print("--- Starting Testing Mode ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'")
        
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)

    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"Model loaded from '{model_path}'")

    print("Evaluating on the test set...")
    all_user_emb, all_item_emb = model(norm_adj_tensor)
    recall, ndcg = evaluate(model, test_df, train_df, all_user_emb, all_item_emb, config.top_k, config.device)
    
    print("\n--- Test Results ---")
    print(f"Recall@{config.top_k}: {recall:.4f}")
    print(f"NDCG@{config.top_k}:   {ndcg:.4f}")
    print("--------------------")

# --- 7. 主入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GNN-based recommendation models.")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode: 'train' or 'test'")
    parser.add_argument('--model_name', type=str, default='LightGCN', choices=['LightGCN'],
                        help="The name of the model class (e.g., 'LightGCN'). Assumes file is 'models/lightgcn.py'.")
    parser.add_argument('--model_path', type=str, help="Path to checkpoint for testing.")
    
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    config = Config()
    
    ModelClass = get_model(args.model_name)

    # 动态设置checkpoint文件名
    config.best_model_name = f'best_{args.model_name.lower()}.pth'

    if args.mode == 'train':
        train(config, ModelClass)
    elif args.mode == 'test':
        model_to_test = args.model_path if args.model_path else os.path.join(config.checkpoint_dir, config.best_model_name)
        test(config, ModelClass, model_to_test)