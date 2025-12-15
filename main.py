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
    """
    加载预处理数据，并补充异构图核心统计信息
    :param data_dir: 数据目录
    :param device: 设备（cuda/cpu）
    :param use_brand: 是否使用品牌节点
    :param debug: 是否调试模式（小样本）
    :return: 训练/验证/测试集 + 节点数 + 归一化邻接矩阵 + 物品-品牌df + 统计信息dict
    """
    print("="*80)
    print("Step 1: Loading and preparing data (with graph structure stats)")
    print("="*80)
    
    # 1. 加载基础数据
    stats_path = os.path.join(data_dir, 'stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found in '{data_dir}'. Please run 'prepare_data.py' first.")

    # 加载交互数据和物品-品牌映射
    all_train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    item_brand_df = pd.read_parquet(os.path.join(data_dir, 'item_brand.parquet'))

    # 2. Debug模式下采样小样本
    data_fraction = 0.01 if debug else 1.0
    if debug:
        print(f"\n[Debug Mode] Using {data_fraction*100}% of the original data")
        unique_users = all_train_df['user_idx'].unique()
        sample_size = max(1, int(len(unique_users) * data_fraction))  # 至少1个用户
        sample_users = np.random.choice(unique_users, size=sample_size, replace=False)
        all_train_df = all_train_df[all_train_df['user_idx'].isin(sample_users)]
        test_df = test_df[test_df['user_idx'].isin(sample_users)]

    # 3. 划分训练/验证集（按用户最后一次交互为验证集）
    all_train_df['rank'] = all_train_df.groupby('user_idx')['user_idx'].rank(method='first', ascending=False)
    val_df = all_train_df[all_train_df['rank'] == 1].copy()
    train_df = all_train_df[all_train_df['rank'] > 1].copy()
    
    # 4. 基础统计（核心）
    with open(stats_path, 'r') as f:
        base_stats = json.load(f)
    num_users = base_stats['num_users']
    num_items = base_stats['num_items']
    num_brands = base_stats['num_brands']  # 属性（品牌）数

    # 5. 计算异构图关键统计指标
    graph_stats = {}
    
    # 5.1 基础节点计数
    graph_stats['num_users'] = num_users
    graph_stats['num_items'] = num_items
    graph_stats['num_brands'] = num_brands
    graph_stats['total_nodes'] = num_users + num_items + num_brands if use_brand else num_users + num_items

    # 5.2 用户-物品交互统计
    total_interactions = len(train_df)  # 训练集总交互数
    graph_stats['total_user_item_interactions'] = total_interactions
    
    # 每个用户平均交互物品数（均值/中位数/最大值）
    user_item_count = train_df.groupby('user_idx')['item_idx'].nunique()
    graph_stats['avg_items_per_user'] = round(user_item_count.mean(), 2)
    graph_stats['median_items_per_user'] = round(user_item_count.median(), 2)
    graph_stats['max_items_per_user'] = user_item_count.max()
    graph_stats['min_items_per_user'] = user_item_count.min()

    # 每个物品平均交互用户数（均值/中位数/最大值）
    item_user_count = train_df.groupby('item_idx')['user_idx'].nunique()
    graph_stats['avg_users_per_item'] = round(item_user_count.mean(), 2)
    graph_stats['median_users_per_item'] = round(item_user_count.median(), 2)
    graph_stats['max_users_per_item'] = item_user_count.max()
    graph_stats['min_users_per_item'] = item_user_count.min()

    # 5.3 物品-品牌（属性）关联统计
    # 每个物品对应品牌数（这里是1，若多属性可调整）
    item_brand_count = item_brand_df.groupby('item_idx')['brand_idx'].nunique()
    graph_stats['avg_brands_per_item'] = round(item_brand_count.mean(), 2)
    graph_stats['median_brands_per_item'] = round(item_brand_count.median(), 2)
    
    # 每个品牌对应物品数（均值/中位数/最大值）
    brand_item_count = item_brand_df.groupby('brand_idx')['item_idx'].nunique()
    graph_stats['avg_items_per_brand'] = round(brand_item_count.mean(), 2)
    graph_stats['median_items_per_brand'] = round(brand_item_count.median(), 2)
    graph_stats['max_items_per_brand'] = brand_item_count.max()
    graph_stats['min_items_per_brand'] = brand_item_count.min()

    # 5.4 图密度（交互数/总可能交互数，反映稀疏性）
    max_possible_user_item = num_users * num_items
    graph_stats['user_item_graph_density'] = round(total_interactions / max_possible_user_item * 100, 6)  # 百分比
    if use_brand:
        total_brand_item_edges = len(item_brand_df)
        max_possible_brand_item = num_brands * num_items
        graph_stats['brand_item_graph_density'] = round(total_brand_item_edges / max_possible_brand_item * 100, 6)

    # 6. 打印统计信息（清晰格式化）
    print("\n" + "="*40 + " Graph Structure Statistics " + "="*40)
    print(f"[Basic Node Count]")
    print(f"  - Users: {graph_stats['num_users']:,}")
    print(f"  - Items: {graph_stats['num_items']:,}")
    print(f"  - Brands (Attributes): {graph_stats['num_brands']:,}")
    print(f"  - Total Nodes (with brand): {graph_stats['total_nodes']:,}")
    
    print(f"\n[User-Item Interaction]")
    print(f"  - Total Interactions: {graph_stats['total_user_item_interactions']:,}")
    print(f"  - Avg Items per User: {graph_stats['avg_items_per_user']} (median: {graph_stats['median_items_per_user']})")
    print(f"  - Avg Users per Item: {graph_stats['avg_users_per_item']} (median: {graph_stats['median_users_per_item']})")
    print(f"  - User-Item Graph Density: {graph_stats['user_item_graph_density']}% (sparsity: {100-graph_stats['user_item_graph_density']:.6f}%)")
    
    print(f"\n[Item-Brand (Attribute) Association]")
    print(f"  - Avg Brands per Item: {graph_stats['avg_brands_per_item']} (median: {graph_stats['median_brands_per_item']})")
    print(f"  - Avg Items per Brand: {graph_stats['avg_items_per_brand']} (median: {graph_stats['median_items_per_brand']})")
    if use_brand:
        print(f"  - Brand-Item Graph Density: {graph_stats['brand_item_graph_density']}%")
    print("="*90 + "\n")

    # 7. 构建邻接矩阵（原有逻辑，补充统计）
    print("Step 2: Building adjacency matrix...")
    item_offset = num_users
    brand_offset = num_users + num_items
    
    # 确定节点总数
    num_nodes = num_users + num_items + num_brands  # 无论use_brand是否为True
    # 移除原有分支：if use_brand: num_nodes = ... else: num_nodes = ...

    # 提取用户-物品边（不变）
    user_indices = train_df['user_idx'].values
    item_indices_for_user = train_df['item_idx'].values + item_offset
    
    # 构建边（仅use_brand=True时添加商品-品牌边，否则只保留用户-物品边）
    print(f"\n[Adjacency Matrix Info]")
    print(f"  - Mode: {'With Brand' if use_brand else 'No Brand'}")
    print(f"  - Train Users Count: {len(np.unique(user_indices)):,}")
    print(f"  - Train Items Count: {len(np.unique(train_df['item_idx'])):,}")
    print(f"  - Total Nodes (fixed): {num_nodes:,} (users+items+brands)")
    
    if use_brand:
        # 有品牌：添加用户-物品边 + 商品-品牌边
        item_indices_for_brand = item_brand_df['item_idx'].values + item_offset
        brand_indices = item_brand_df['brand_idx'].values + brand_offset
        all_rows = np.concatenate([user_indices, item_indices_for_user, item_indices_for_brand, brand_indices])
        all_cols = np.concatenate([item_indices_for_user, user_indices, brand_indices, item_indices_for_brand])
        expected_edges = (len(user_indices) + len(item_brand_df)) * 2
    else:
        # 无品牌：仅保留用户-物品边（品牌节点孤立，无任何边）
        all_rows = np.concatenate([user_indices, item_indices_for_user])
        all_cols = np.concatenate([item_indices_for_user, user_indices])
        expected_edges = len(user_indices) * 2
    
    all_data = np.ones(all_rows.shape[0], dtype=np.float32)

    # 验证边数
    print(f"  - Expected Edges: {expected_edges:,}")
    print(f"  - Actual Edges: {all_rows.shape[0]:,}")
    assert all_rows.shape[0] == expected_edges, f"Edge count mismatch! Expected {expected_edges}, got {all_rows.shape[0]}"

    # 构建稀疏邻接矩阵
    adj_mat = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    print(f"  - Adjacency Matrix Shape: {adj_mat.shape} (nodes × nodes)")
    print(f"  - Non-zero Elements: {adj_mat.nnz:,}")
    
    # 8. 归一化邻接矩阵（原有逻辑）
    rowsum = np.array(adj_mat.sum(axis=1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    # 转为PyTorch稀疏张量
    indices = torch.LongTensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)))
    values = torch.FloatTensor(norm_adj_mat.data)
    norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size((num_nodes, num_nodes))).to(device)

    # 9. 打印最终数据概览
    print(f"\n[Final Data Overview]")
    print(f"  - Train Interactions: {len(train_df):,}")
    print(f"  - Val Interactions: {len(val_df):,}")
    print(f"  - Test Interactions: {len(test_df):,}")
    print(f"  - Normalized Adj Tensor Device: {norm_adj_tensor.device}")
    print("="*80 + "\n")

    # 返回值补充统计信息（可选，便于后续分析）
    return train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor, item_brand_df#, graph_stats

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

def evaluate(model, val_or_test_data, train_data, norm_adj_tensor, k, device, batch_size=1024, use_brand=True, item_brand_df=None):
    model.eval()
    test_user_items = dict(zip(val_or_test_data['user_idx'], val_or_test_data['item_idx']))
    train_user_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    test_users = list(test_user_items.keys())
    recalls, ndcgs = [], []

    with torch.no_grad():
        # 【修改】接收 final_brand_emb
        all_user_emb, all_item_emb, final_brand_emb, _, _ = model(norm_adj_tensor, use_brand=use_brand)
        
        # 仅在use_brand=True时构建商品-品牌嵌入映射
        if use_brand and final_brand_emb is not None and item_brand_df is not None:
            # 构建商品→品牌的映射张量（确保和模型同设备）
            item_brand_map = torch.LongTensor(item_brand_df['brand_idx'].values).to(device)
            # 每个商品对应的品牌嵌入
            item_brand_emb = final_brand_emb[item_brand_map]
        else:
            item_brand_emb = None

        for i in tqdm(range(0, len(test_users), batch_size), desc="Evaluating"):
            batch_users = test_users[i: i + batch_size]
            batch_users_tensor = torch.LongTensor(batch_users).to(device)
            
            # 基础评分：用户-商品匹配
            batch_scores = torch.matmul(all_user_emb[batch_users_tensor], all_item_emb.T)
            
            # 【核心】融合品牌偏好评分
            if use_brand and item_brand_emb is not None:
                # 计算用户对每个商品品牌的偏好分数
                brand_scores = torch.matmul(all_user_emb[batch_users_tensor], item_brand_emb.T)
                # 加权融合（权重可调，建议0.1-0.3）
                batch_scores = batch_scores + 0.2 * brand_scores
            
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
    train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor, item_brand_df = \
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
            
            final_user_emb_all, final_item_emb_all, initial_user_emb_all, initial_item_emb_all = model(norm_adj_tensor, use_brand=use_brand)
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
            recall, ndcg = evaluate(
                model, val_df, train_df, norm_adj_tensor, 
                config.top_k, config.device, 
                use_brand=use_brand, item_brand_df=item_brand_df  # 新增参数
            )
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
    
    # 【修改】加载数据时接收 item_brand_df
    train_df, val_df, test_df, num_users, num_items, num_brands, norm_adj_tensor, item_brand_df = \
        load_preprocessed_data(config.processed_data_dir, config.device, use_brand=use_brand, debug=config.debug)
        
    model = model_class(num_users, num_items, num_brands, config).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print(f"Model loaded from '{model_path}'")

    print("Evaluating on the TEST set...")
    # 在测试时，过滤集是 train_df + val_df
    full_train_df_for_filter = pd.concat([train_df, val_df])
    
    # 【修改】调用evaluate时传入 use_brand 和 item_brand_df
    recall, ndcg = evaluate(
        model, test_df, full_train_df_for_filter, norm_adj_tensor, 
        config.top_k, config.device, 
        use_brand=use_brand, item_brand_df=item_brand_df  # 新增参数
    )
    
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
    parser.add_argument('--core', type=int, default=20, help="K-core filtering threshold for data.")
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