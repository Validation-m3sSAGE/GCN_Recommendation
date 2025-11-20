# main.py

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
import matplotlib.pyplot as plt
import subprocess

def set_best_gpu():
    """自动选择最空闲的GPU并设置CUDA_VISIBLE_DEVICES"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        free_memory = [int(x) for x in result.strip().split('\n')]
        best_gpu_id = np.argmax(free_memory)
        print(f"GPUs free memory: {free_memory}. Automatically selecting GPU {best_gpu_id}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    except Exception as e:
        print(f"Could not automatically select GPU: {e}. Using default GPU settings.")

def get_model(model_name):
    """根据模型名称动态地从 models 文件夹导入模型类"""
    try:
        module_path = f"models.{model_name.lower()}"
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import model '{model_name}' from '{module_path}'.py.")
        print(f"Please ensure file 'models/{model_name.lower()}.py' exists and contains class '{model_name}'.")
        raise e

class Config:
    def __init__(self, core=5):
        self.processed_data_dir = f'dataset/amazon_books/processed_data_{core}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.embedding_dim = 64
        self.n_layers = 3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 2048
        self.epochs = 12 # 建议至少训练20个epoch
        self.top_k = 20
        self.checkpoint_dir = 'checkpoints'
        self.results_dir = 'results'
        self.best_model_name = 'best_model.pth'

class Logger:
    def __init__(self, results_dir, model_name):
        self.results_dir, self.model_name = results_dir, model_name
        os.makedirs(self.results_dir, exist_ok=True)
        self.history = {'step': [], 'batch_loss': [], 'epoch': [], 'epoch_avg_loss': [], 'recall': [], 'ndcg': []}
        self.current_step = 0

    def log_batch_loss(self, loss):
        self.history['step'].append(self.current_step)
        self.history['batch_loss'].append(loss)
        self.current_step += 1

    def log_epoch_metrics(self, epoch, avg_loss, recall, ndcg):
        self.history['epoch'].append(epoch)
        self.history['epoch_avg_loss'].append(avg_loss)
        self.history['recall'].append(recall)
        self.history['ndcg'].append(ndcg)
        print(f"Logger: Epoch {epoch} metrics logged.")

    def plot_and_save(self):
        if not self.history['epoch']:
            print("Logger: No data to plot.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle(f'Training History for {self.model_name}', fontsize=16)
        
        # Loss Curve
        ax1.plot(self.history['step'], self.history['batch_loss'], 'b-', alpha=0.3, label='Per-Batch Training Loss')
        if self.history['epoch_avg_loss']:
            num_batches_per_epoch = len(self.history['step']) / self.history['epoch'][-1]
            epoch_steps = [(e - 1) * num_batches_per_epoch for e in self.history['epoch']]
            ax1.plot(epoch_steps, self.history['epoch_avg_loss'], 'r-o', markersize=8, label='Per-Epoch Average Loss')
        ax1.set_title('Training Loss'); ax1.set_xlabel('Training Step'); ax1.set_ylabel('Loss')
        ax1.grid(True); ax1.legend(); ax1.set_yscale('log')
        
        # Metrics Curve
        ax2.plot(self.history['epoch'], self.history['recall'], 'r-s', label=f'Recall@{config.top_k}')
        ax2.plot(self.history['epoch'], self.history['ndcg'], 'g-^', label=f'NDCG@{config.top_k}')
        ax2.set_title('Evaluation Metrics per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Metric Value')
        ax2.grid(True); ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.results_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(save_path)
        print(f"Training curves plot saved to '{save_path}'")
        plt.close()

def load_preprocessed_data(data_dir, device):
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
    
    print("Efficiently building heterogeneous adjacency matrix in COO format...")
    item_offset = num_users
    brand_offset = num_users + num_items
    num_nodes = num_users + num_items + num_brands

    user_indices = train_data['user_idx'].values
    item_indices_for_user = train_data['item_idx'].values + item_offset
    item_indices_for_brand = item_brand_data['item_idx'].values + item_offset
    brand_indices = item_brand_data['brand_idx'].values + brand_offset
    
    all_rows = np.concatenate([user_indices, item_indices_for_user, item_indices_for_brand, brand_indices])
    all_cols = np.concatenate([item_indices_for_user, user_indices, brand_indices, item_indices_for_brand])
    all_data = np.ones(all_rows.shape[0], dtype=np.float32)

    adj_mat = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
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

class BPRDataset(Dataset):
    def __init__(self, df, num_items):
        self.users, self.pos_items, self.num_items = df['user_idx'].values, df['item_idx'].values, num_items
        self.user_pos_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user, pos_item = self.users[idx], self.pos_items[idx]
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos_items.get(user, set()):
                break
        return user, pos_item, neg_item

def bpr_loss_reg(final_user_emb, final_pos_item_emb, final_neg_item_emb,
                 initial_user_emb, initial_pos_item_emb, initial_neg_item_emb, lambda_reg):
    pos_scores = torch.sum(final_user_emb * final_pos_item_emb, dim=1)
    neg_scores = torch.sum(final_user_emb * final_neg_item_emb, dim=1)
    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    reg_loss = lambda_reg * (initial_user_emb.norm(2).pow(2) + initial_pos_item_emb.norm(2).pow(2) +
                             initial_neg_item_emb.norm(2).pow(2)) / float(len(final_user_emb))
    return bpr_loss + reg_loss

def evaluate(model, test_data, train_data, k, device, batch_size=1024):
    model.eval()
    test_user_items = dict(zip(test_data['user_idx'], test_data['item_idx']))
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    test_users = list(test_user_items.keys())
    recalls, ndcgs = [], []

    with torch.no_grad():
        all_user_emb, all_item_emb, _, _ = model(load_preprocessed_data.norm_adj_tensor)
        for i in tqdm(range(0, len(test_users), batch_size), desc="Evaluating"):
            batch_users = test_users[i: i + batch_size]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            batch_scores = torch.matmul(all_user_emb[batch_users_tensor], all_item_emb.T)

            for j, user_idx in enumerate(batch_users):
                if user_idx in train_user_items:
                    batch_scores[j, train_user_items[user_idx]] = -1e10
            
            _, top_k_indices = torch.topk(batch_scores, k=k)
            top_k_indices_cpu = top_k_indices.cpu().numpy()
            batch_true_items = [test_user_items[user] for user in batch_users]

            for j in range(len(batch_users)):
                pred_items, true_item = top_k_indices_cpu[j], batch_true_items[j]
                hit = true_item in pred_items
                recalls.append(1 if hit else 0)
                if hit:
                    position = np.where(pred_items == true_item)[0][0]
                    ndcgs.append(1 / np.log2(position + 2))
                else:
                    ndcgs.append(0)
    return np.mean(recalls), np.mean(ndcgs)

def train(config, model_class, model_name):
    logger = Logger(config.results_dir, model_name)
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)
    load_preprocessed_data.norm_adj_tensor = norm_adj_tensor # Cache tensor for eval
    
    train_loader = DataLoader(BPRDataset(train_df, num_items), batch_size=config.batch_size, shuffle=True, num_workers=4)
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    print(f"Initialized model: {model.__class__.__name__}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_recall = 0.0
    os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    print("\nStep 2: Starting model training...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for users, pos_items, neg_items in progress_bar:
            users, pos_items, neg_items = users.to(config.device), pos_items.to(config.device), neg_items.to(config.device)
            optimizer.zero_grad()
            
            final_user_emb_all, final_item_emb_all, initial_user_emb_all, initial_item_emb_all = model(norm_adj_tensor)
            final_user_emb, final_pos_item_emb, final_neg_item_emb = final_user_emb_all[users], final_item_emb_all[pos_items], final_item_emb_all[neg_items]
            initial_user_emb, initial_pos_item_emb, initial_neg_item_emb = initial_user_emb_all[users], initial_item_emb_all[pos_items], initial_item_emb_all[neg_items]

            loss = bpr_loss_reg(final_user_emb, final_pos_item_emb, final_neg_item_emb,
                                initial_user_emb, initial_pos_item_emb, initial_neg_item_emb, config.weight_decay)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            logger.log_batch_loss(batch_loss)
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        
        check_path = os.path.join(config.checkpoint_dir, model_name+"_epoch_"+str(epoch)+".pth")
        torch.save(model.state_dict(), check_path)
        
        if epoch % 2 == 0:
            print("Evaluating...")
            recall, ndcg = evaluate(model, test_df, train_df, config.top_k, config.device)
            print(f"Epoch {epoch} | Recall@{config.top_k}: {recall:.4f}, NDCG@{config.top_k}: {ndcg:.4f}")
            logger.log_epoch_metrics(epoch=epoch, avg_loss=avg_loss, recall=recall, ndcg=ndcg)

            if recall > best_recall:
                best_recall = recall
                save_path = os.path.join(config.checkpoint_dir, config.best_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with Recall@{config.top_k}: {best_recall:.4f} to '{save_path}'")
                
    print("Training finished.")
    logger.plot_and_save()

def test(config, model_class, model_path):
    print("--- Starting Testing Mode ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'")
    train_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device)
    load_preprocessed_data.norm_adj_tensor = norm_adj_tensor # Cache tensor for eval
    
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"Model loaded from '{model_path}'")

    print("Evaluating on the test set...")
    recall, ndcg = evaluate(model, test_df, train_df, config.top_k, config.device)
    print("\n--- Test Results ---")
    print(f"Recall@{config.top_k}: {recall:.4f}")
    print(f"NDCG@{config.top_k}:   {ndcg:.4f}")
    print("--------------------")

if __name__ == '__main__':
    set_best_gpu()
    parser = argparse.ArgumentParser(description="Run GNN-based recommendation models.")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode: 'train' or 'test'")
    parser.add_argument('--model_name', type=str, default='LightGCN', help="The name of the model class.")
    parser.add_argument('--core', type=int, default=5, help="K-core filtering threshold used for data preprocessing.")
    parser.add_argument('--model_path', type=str, help="Path to checkpoint for testing.")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    config = Config(core=args.core)
    ModelClass = get_model(args.model_name)
    config.best_model_name = f'best_{args.model_name.lower()}_core{args.core}.pth'

    if args.mode == 'train':
        train(config, ModelClass, args.model_name) 
    elif args.mode == 'test':
        model_to_test = args.model_path if args.model_path else os.path.join(config.checkpoint_dir, config.best_model_name)
        test(config, ModelClass, model_to_test)