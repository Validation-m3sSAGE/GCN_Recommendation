# prepare_data.py

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

def prepare_and_save_data(config):
    print("--- Starting Data Preparation for Steam Dataset ---")
    
    # --- 1. MODIFIED: 加载和过滤评论数据 (使用 'recommanded') ---
    print(f"Step 1: Loading POSITIVE ('recommanded=True') reviews from '{config['review_file']}'...")
    
    positive_reviews = []
    with open(config['review_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Reviews"):
            review = json.loads(line.strip())
            # --- CORE MODIFICATION: 使用 'recommanded' 字段 ---
            if review.get('recommanded') is True:
                positive_reviews.append(review)

    print(f"Loaded {len(positive_reviews)} positive interactions initially.")
    
    if not positive_reviews:
        print("Error: No positive reviews found. Please check the review file and format.")
        return

    df = pd.DataFrame(positive_reviews)[['user_id', 'item_id', 'timestamp']]
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp']) # 确保时间戳是数值类型
    
    # K-core filtering
    print("\nApplying K-core filtering...")
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        if config['min_interactions'] <= 1:
            break

        weak_users = user_counts[user_counts < config['min_interactions']].index
        weak_items = item_counts[item_counts < config['min_interactions']].index
        if len(weak_users) == 0 and len(weak_items) == 0:
            break
        df = df[~df['user_id'].isin(weak_users)]
        df = df[~df['item_id'].isin(weak_items)]
    
    print(f"Final filtered data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items.")
    
    # --- 2. 加载元数据 (MODIFIED for Steam 'genres' and 'tags') ---
    print(f"\nStep 2: Loading metadata and extracting 'genres' and 'tags' from '{config['meta_file']}'...")
    active_items_set = set(df['item_id'].unique())
    
    meta_categories = {}
    meta_embeddings = {}

    with open(config['meta_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Metadata"):
            record = json.loads(line.strip())
            item_id = record.get('item_id')
            
            if item_id in active_items_set:
                # --- CORE MODIFICATION: 提取 'genres' 和 'tags' ---
                genres = record.get('genres', [])
                # tags 是一个字典，我们需要它的 keys
                tags = list(record.get('tags', {}).keys())
                
                # 合并 genres 和 tags, 并去重
                all_cats = list(set(genres + tags))
                
                meta_categories[item_id] = all_cats if all_cats else ['Unknown']
                # --- END MODIFICATION ---

                embedding = record.get('embd')
                if embedding:
                    meta_embeddings[item_id] = embedding

    print(f"Extracted categories (genres/tags) for {len(meta_categories)} items.")

    # --- 3. 创建映射和处理索引 (逻辑不变，变量名统一为 'category') ---
    print("\nStep 3: Creating ID maps and indexing data...")
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    item_cat_list = [{'item_id': k, 'categories': v} for k, v in meta_categories.items()]
    item_cat_df_raw = pd.DataFrame(item_cat_list)
    
    item_cat_df = item_cat_df_raw.explode('categories').reset_index(drop=True)
    item_cat_df.rename(columns={'categories': 'category'}, inplace=True)

    category_map = {cat: i for i, cat in enumerate(item_cat_df['category'].unique())}
    
    item_cat_df['item_idx'] = item_cat_df['item_id'].map(item_map)
    item_cat_df['brand_idx'] = item_cat_df['category'].map(category_map)
    
    item_cat_df.dropna(subset=['item_idx'], inplace=True)
    item_cat_df['item_idx'] = item_cat_df['item_idx'].astype(int)

    # --- 4. 划分数据集 (MODIFIED: 使用 'timestamp') ---
    print("\nStep 4: Splitting data into training and testing sets using timestamp...")
    # 使用 timestamp 排序，最新的作为测试集
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df['rank'] = df.groupby('user_idx').cumcount(ascending=False) # 最后一次交互 rank=0
    
    test_df = df[df['rank'] == 0]
    train_df = df[df['rank'] > 0]
    print(f"Split to {len(train_df)} training interactions and {len(test_df)} testing interactions.")

    # --- 5. 保存处理好的数据 (文件名统一) ---
    output_dir = os.path.join(config['output_base_dir'], f"processed_data_{config['min_interactions']}_pos_only_cat")
    print(f"\nStep 5: Saving processed data to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    train_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    test_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    item_cat_df[['item_idx', 'brand_idx']].to_parquet(os.path.join(output_dir, 'item_brand.parquet'), index=False)
    
    # 保存 embedding 数据 (逻辑不变)
    embeddings_to_save = {item_map[k]: v for k, v in meta_embeddings.items() if k in item_map}
    if embeddings_to_save:
        embd_dim = len(next(iter(embeddings_to_save.values())))
        item_embeddings_matrix = np.zeros((len(item_map), embd_dim), dtype=np.float32)
        for item_idx, embd in embeddings_to_save.items():
            if item_idx < len(item_map): # 安全检查
                item_embeddings_matrix[item_idx] = embd
        np.save(os.path.join(output_dir, 'item_embeddings.npy'), item_embeddings_matrix)
        print("Item embeddings saved to 'item_embeddings.npy'.")

    # 保存统计信息 (变量名统一)
    stats = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_brands': len(category_map)
    }
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

    print("\n--- Data Preparation Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess recommendation data.")
    # 允许通过命令行参数轻松切换数据集
    parser.add_argument('--dataset', type=str, default='steam', choices=['steam', 'books'], help="Dataset to process.")
    parser.add_argument('--core', type=int, default=16, help="K-core filtering threshold.")
    args = parser.parse_args()
    
    if args.dataset == 'steam':
        prep_config = {
            'review_file': 'dataset/steam_emb/raw_data/steam_16_core_sentiment_20251226.jsonl',
            'meta_file': 'dataset/steam_emb/raw_data/steam_metadata_full_embd_20251227.jsonl',
            'min_interactions': args.core,
            'output_base_dir': 'dataset/steam_emb/'
        }
    
    prepare_and_save_data(prep_config)