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

# --- GPU & Model Loading ---
def set_best_gpu():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        gpus = [{'id': int(index), 'free_mem': int(free_memory), 'util': int(gpu_util)} 
                for index, free_memory, gpu_util in (line.split(',') for line in result.strip().split('\n'))]
        
        for gpu in gpus:
            gpu['score'] = gpu['free_mem'] - 2 * gpu['util']
        
        gpus.sort(key=lambda x: x['score'], reverse=True)
        best_gpu_id = gpus[0]['id']
        
        print("--- GPU Status ---")
        for gpu in gpus:
            print(f"GPU {gpu['id']}: Free Memory={gpu['free_mem']}MB, Utilization={gpu['util']}%, Score={gpu['score']:.2f}")
        print("--------------------")
        
        print(f"Automatically selecting the best single GPU: {best_gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    except Exception as e:
        print(f"Could not automatically select GPU: {e}. Using default GPU settings.")

def get_model(model_name):
    try:
        module_path = f"models.{model_name.lower()}"
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import model '{model_name}' from '{module_path}'.py.")
        raise e

# --- Config (MODIFIED for Debug Mode) ---
class Config:
    def __init__(self, args):
        self.debug = args.debug
        core = args.core
        
        self.processed_data_dir = f'dataset/amazon_books/processed_data_{core}'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.embedding_dim = 64
        self.n_layers = 3
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.top_k = 20
        self.num_workers = 4
        self.val_interval = 5
        
        # Checkpoint and results directory
        self.checkpoint_dir = 'exp/checkpoints/checkpoints'
        self.results_dir = 'exp/results/results'
        self.best_model_name = 'best_model.pth'

        if self.debug:
            print("--- RUNNING IN DEBUG MODE ---")
            self.epochs = 5
            self.batch_size = 128
            self.val_interval = 1
            # In debug mode, we can save checkpoints to a separate folder if needed
            self.checkpoint_dir = os.path.join('debug', self.checkpoint_dir)
            self.results_dir = os.path.join('debug', self.results_dir)
        else:
            self.epochs = args.epochs
            self.batch_size = 2048

# --- MODIFIED: Logger 类，增加保存CSV功能 ---
class Logger:
    def __init__(self, results_dir, model_name):
        self.results_dir = results_dir
        self.model_name = model_name
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.history = {
            'step': [], 'batch_loss': [], 'epoch': [], 'epoch_avg_loss': [],
            'recall': [], 'ndcg': []
        }
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

    def save(self, total_epochs): # 传入总epoch数以计算平均step
        if not self.history['epoch']:
            print("Logger: No epoch data to save.")
            return

        # --- 1. 保存 CSV ---
        epoch_history_df = pd.DataFrame({
            'epoch': self.history['epoch'],
            'avg_loss': self.history['epoch_avg_loss'],
            'recall': self.history['recall'],
            'ndcg': self.history['ndcg']
        })
        csv_save_path = os.path.join(self.results_dir, f'{self.model_name}_epoch_history.csv')
        epoch_history_df.to_csv(csv_save_path, index=False)
        print(f"Epoch-level history saved to '{csv_save_path}'")
        
        # --- 2. 绘制图像 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle(f'Training History for {self.model_name}', fontsize=16)

        # 绘制 Loss 曲线
        if self.history['step']:
            ax1.plot(self.history['step'], self.history['batch_loss'], 'b-', alpha=0.3, label='Per-Batch Training Loss')
        
        # --- MODIFICATION START to fix x-axis for epoch loss ---
        if self.history['epoch_avg_loss']:
            # 计算平均每个 epoch 有多少个 step (batch)
            # self.current_step 是总的 step 数
            # total_epochs 是总的训练轮数
            avg_steps_per_epoch = self.current_step / total_epochs
            # 计算每个 epoch 标记点在 x 轴上的精确位置
            epoch_steps = [e * avg_steps_per_epoch for e in self.history['epoch']]
            ax1.plot(epoch_steps, self.history['epoch_avg_loss'], 'r-o', markersize=8, label='Per-Epoch Average Loss')
        # --- MODIFICATION END ---
            
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        ax1.set_yscale('log')

        # 绘制 Recall 和 NDCG 曲线
        ax2.plot(self.history['epoch'], self.history['recall'], 'r-s', label=f'Recall@{config.top_k}')
        ax2.plot(self.history['epoch'], self.history['ndcg'], 'g-^', label=f'NDCG@{config.top_k}')
        ax2.set_title('Evaluation Metrics per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Value')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        img_save_path = os.path.join(self.results_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(img_save_path)
        print(f"Training curves plot saved to '{img_save_path}'")
        plt.close()

# --- Data Loading (MODIFIED for Debug Mode) ---
def load_preprocessed_data(data_dir, device, use_brand=True, debug=False):
    print("Step 1: Loading and preparing data...")
    stats_path = os.path.join(data_dir, 'stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Data not found in '{data_dir}'. Please run 'prepare_data.py'.")

    data_fraction = 0.01 if debug else 1.0 
    
    print("Loading and splitting train/validation sets at runtime...")
    all_train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    item_brand_df = pd.read_parquet(os.path.join(data_dir, 'item_brand.parquet'))

    if debug:
        print(f"Debug mode: Using a small fraction ({data_fraction * 100}%) of the data.")
        unique_users = all_train_df['user_idx'].unique()
        if len(unique_users) == 0:
             raise ValueError("No users found after loading data. Check data paths and content.")
        sample_size = int(len(unique_users) * data_fraction)
        if sample_size == 0:
             sample_size = 1 # Ensure at least one user is sampled
        sample_users = np.random.choice(unique_users, size=sample_size, replace=False)
        all_train_df = all_train_df[all_train_df['user_idx'].isin(sample_users)]
        test_df = test_df[test_df['user_idx'].isin(sample_users)]

    all_train_df['rank'] = all_train_df.groupby('user_idx')['user_idx'].rank(method='first', ascending=False)
    val_df = all_train_df[all_train_df['rank'] == 1]
    train_df = all_train_df[all_train_df['rank'] > 1]
    
    # 确保即使在debug模式下划分后仍有数据
    if train_df.empty or val_df.empty:
        print("Warning: train_df or val_df is empty after split. May cause issues in training/evaluation.")

    print(f"Data loaded: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test interactions.")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    num_users, num_items, num_brands = stats['num_users'], stats['num_items'], stats['num_brands']

    print("Building adjacency matrix...")
    item_offset = num_users
    brand_offset = num_users + num_items
    num_nodes = num_users + num_items + num_brands
    
    user_indices = train_df['user_idx'].values
    item_indices_for_user = train_df['item_idx'].values + item_offset
    
    # --- START DEBUGGING BLOCK ---
    print("\n" + "="*20 + " GRAPH CONSTRUCTION DEBUG " + "="*20)
    print(f"Mode: {'With Brand' if use_brand else 'No Brand'}")
    print(f"Total Users in train_df: {len(user_indices)}")
    
    if use_brand:
        print("Building graph with Brand information.")
        item_indices_for_brand = item_brand_df['item_idx'].values + item_offset
        brand_indices = item_brand_df['brand_idx'].values + brand_offset
        all_rows = np.concatenate([user_indices, item_indices_for_user, item_indices_for_brand, brand_indices])
        all_cols = np.concatenate([item_indices_for_user, user_indices, brand_indices, item_indices_for_brand])
    else:
        print("Building graph WITHOUT Brand information (ablation study).")
        all_rows = np.concatenate([user_indices, item_indices_for_user])
        all_cols = np.concatenate([item_indices_for_user, user_indices])
    
    # 定义 all_data
    all_data = np.ones(all_rows.shape[0], dtype=np.float32)

    # --- FINAL DIAGNOSIS BLOCK ---
    expected_edges = 0
    if use_brand:
        expected_edges = (len(user_indices) + len(item_brand_df['item_idx'].values)) * 2
    else:
        expected_edges = len(user_indices) * 2
        
    print(f"DEBUG: expected_edges = {expected_edges}")
    print(f"DEBUG: all_rows.shape[0] = {all_rows.shape[0]}")
    print(f"DEBUG: all_data.shape[0] = {all_data.shape[0]}")
    
    # 强制断言，如果长度不匹配，程序会在这里崩溃
    assert all_rows.shape[0] == expected_edges, "Error: Mismatch in row coordinates count!"
    assert all_data.shape[0] == expected_edges, "Error: Mismatch in data count!"
    # --- END DIAGNOSIS BLOCK ---

    adj_mat = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    
    # 检查稀疏矩阵的非零元素数量是否符合预期
    # coo_matrix 会合并重复项，所以 nnz 可能会略小于 expected_edges，但差距不应过大
    if abs(adj_mat.nnz - expected_edges) > 10: # 允许少量重复
         print(f"WARNING: Significant difference between expected edges ({expected_edges}) and matrix nnz ({adj_mat.nnz}).")

    print(f"Final Adjacency Matrix non-zero elements (adj_mat.nnz): {adj_mat.nnz}")
    print("="*60 + "\n")
    
    rowsum = np.array(adj_mat.sum(axis=1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj_mat.shape)).to(device)
    
    return train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor

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

def evaluate(model, val_or_test_data, train_data, norm_adj_tensor, k, device, batch_size=1024):
    model.eval()
    test_user_items = dict(zip(val_or_test_data['user_idx'], val_or_test_data['item_idx']))
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    test_users = list(test_user_items.keys())
    recalls, ndcgs = [], []

    with torch.no_grad():
        # GNN传播在评估开始时只进行一次
        # FIX: Use the explicitly passed norm_adj_tensor
        all_user_emb, all_item_emb, _, _ = model(norm_adj_tensor)
        
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

# --- train (MODIFIED for Debug Mode) ---
def train(config, model_class, model_name, use_brand):
    logger = Logger(config.results_dir, f"{model_name}_{'brand' if use_brand else 'no_brand'}")
    
    # 加载所有数据，包括用于最终测试过滤的全量训练数据
    train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device, use_brand=use_brand, debug=config.debug)
    
    train_loader = DataLoader(BPRDataset(train_df, num_items), 
                              batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True)
                              
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_recall = 0.0
    os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    print("\nStep 2: Starting model training...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        
        # 在调试模式下，每个epoch只运行几个batch
        max_batches_per_epoch = 10 if config.debug else len(train_loader)
        
        progress_bar = tqdm(train_loader, total=max_batches_per_epoch, desc=f"Epoch {epoch}/{config.epochs}")
        
        for i, (users, pos_items, neg_items) in enumerate(progress_bar):
            if i >= max_batches_per_epoch:
                break # 提前结束 epoch
                
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

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        
        if epoch % config.val_interval == 0:
            print("Evaluating on VALIDATION set...")
            # FIX: Explicitly pass norm_adj_tensor to the evaluate function
            recall, ndcg = evaluate(model, val_df, train_df, norm_adj_tensor, config.top_k, config.device)
            print(f"Epoch {epoch} | Val Recall@{config.top_k}: {recall:.4f}, Val NDCG@{config.top_k}: {ndcg:.4f}")
            logger.log_epoch_metrics(epoch=epoch, avg_loss=avg_loss, recall=recall, ndcg=ndcg)

            if recall > best_recall:
                best_recall = recall
                save_path = os.path.join(config.checkpoint_dir, config.best_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved...")

    print("Training finished.")
    logger.save(total_epochs=config.epochs)

def test(config, model_class, model_path, use_brand):
    print("--- Starting Testing Mode ---")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'")
    
    # 加载所有数据集
    train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor = \
        load_preprocessed_data(config.processed_data_dir, config.device, use_brand=use_brand, debug=config.debug)
        
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"Model loaded from '{model_path}'")

    print("Evaluating on the TEST set...")
    # 在测试时，过滤集是 train_df + val_df
    full_train_df_for_filter = pd.concat([train_df, val_df])
    
    # FIX: Explicitly pass norm_adj_tensor to the evaluate function
    recall, ndcg = evaluate(model, test_df, full_train_df_for_filter, norm_adj_tensor, config.top_k, config.device)
    
    print("\n--- Final Test Results ---")
    print(f"Recall@{config.top_k}: {recall:.4f}")
    print(f"NDCG@{config.top_k}:   {ndcg:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    # 调用新的单GPU选择函数
    set_best_gpu() 
    
    parser = argparse.ArgumentParser(description="Run GNN-based recommendation models.")
    parser.add_argument('mode', choices=['train', 'test'], help="Mode: 'train' or 'test'")
    parser.add_argument('--model_name', type=str, default='LightGCN', help="The name of the model class.")
    parser.add_argument('--core', type=int, default=10, help="K-core filtering threshold for data.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--model_path', type=str, help="Path to checkpoint for testing.")
    parser.add_argument('--no_brand', action='store_true', help="Run ablation study without brand info.")
    # --num_gpus 参数不再需要，可以移除
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for a quick run.")
    
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    
    # Config 初始化现在直接接收 args
    config = Config(args)
    ModelClass = get_model(args.model_name)
    
    ablation_suffix = '_no_brand' if args.no_brand else ''
    config.best_model_name = f'best_{args.model_name.lower()}_core{args.core}{ablation_suffix}.pth'

    if args.mode == 'train':
        train(config, ModelClass, args.model_name, use_brand=not args.no_brand) 
    elif args.mode == 'test':
        model_to_test = args.model_path if args.model_path else os.path.join(config.checkpoint_dir, config.best_model_name)
        test(config, ModelClass, model_to_test, use_brand=not args.no_brand)