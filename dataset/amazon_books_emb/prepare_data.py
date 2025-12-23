# prepare_data.py

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

def extract_meaningful_categories(categories):
    """
    辅助函数：从类别列表中提取有意义的子类别（第2、3项）。
    """
    meaningful_cats = []
    if isinstance(categories, list) and len(categories) > 1:
        # 添加第二项
        meaningful_cats.append(categories[1])
        if len(categories) > 2:
            # 添加第三项
            meaningful_cats.append(categories[2])
    return meaningful_cats if meaningful_cats else ['Unknown'] # 如果没有，返回Unknown

def prepare_and_save_data(config):
    print("--- Starting Data Preparation ---")
    
    # --- 1. MODIFIED: 加载和过滤评论数据 ---
    print(f"Step 1: Loading and filtering POSITIVE reviews from '{config['review_file']}'...")
    
    positive_reviews = []
    with open(config['review_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Reviews"):
            review = json.loads(line.strip())
            # --- CORE MODIFICATION: 只保留 sentiment 为 'positive' 的记录 ---
            if review.get('sentiment') == 'positive':
                positive_reviews.append(review)

    print(f"Loaded {len(positive_reviews)} positive interactions initially.")
    
    if not positive_reviews:
        print("Error: No positive reviews found. Please check the review file and format.")
        return

    # 现在 df 只包含正向交互
    # 注意：新的交互数据中 item_id 就是 item_id，不需要再用 parent_asin
    df = pd.DataFrame(positive_reviews)[['user_id', 'item_id', 'rating']]
    df.dropna(inplace=True)
    
    # K-core filtering
    # 即使原始数据是20-core，在过滤掉负样本后，一些用户或物品的交互数可能低于阈值，所以需要重新进行K-core过滤
    print("\nRe-applying K-core filtering on positive interactions...")
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        # 如果 min_interactions=0 或 1，则跳过过滤
        if config['min_interactions'] <= 1:
            print(f"Skipping K-core filtering as min_interactions is {config['min_interactions']}.")
            break

        weak_users = user_counts[user_counts < config['min_interactions']].index
        weak_items = item_counts[item_counts < config['min_interactions']].index
        if len(weak_users) == 0 and len(weak_items) == 0:
            break
        df = df[~df['user_id'].isin(weak_users)]
        df = df[~df['item_id'].isin(weak_items)]
    
    print(f"Final filtered data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items.")
    # --- 2. 加载元数据 (MODIFIED for multi-category) ---
    print(f"\nStep 2: Loading metadata and extracting categories from '{config['metadata_file']}'...")
    active_items_set = set(df['item_id'].unique())
    
    meta_categories = {}
    meta_embeddings = {}

    with open(config['metadata_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Metadata"):
            record = json.loads(line.strip())
            item_id = record.get('item_id')
            
            if item_id in active_items_set:
                # --- CORE MODIFICATION: 提取第2、3个类别 ---
                categories = record.get('categories', [])
                meaningful_cats = extract_meaningful_categories(categories)
                meta_categories[item_id] = meaningful_cats
                # --- END MODIFICATION ---

                embedding = record.get('embd')
                if embedding:
                    meta_embeddings[item_id] = embedding

    print(f"Extracted categories for {len(meta_categories)} items.")

    # --- 3. 创建映射和处理索引 (MODIFIED for multi-category) ---
    print("\nStep 3: Creating ID maps and indexing data...")
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    # --- CORE MODIFICATION: 处理多对多关系 ---
    # 1. 创建包含列表的 DataFrame
    item_cat_list = [{'item_id': k, 'categories': v} for k, v in meta_categories.items()]
    item_cat_df_raw = pd.DataFrame(item_cat_list)
    
    # 2. 使用 explode 将每个类别展开为单独的行
    item_cat_df = item_cat_df_raw.explode('categories').reset_index(drop=True)
    item_cat_df.rename(columns={'categories': 'category'}, inplace=True)

    # 3. 为所有唯一的类别创建映射
    category_map = {cat: i for i, cat in enumerate(item_cat_df['category'].unique())}
    
    # 4. 映射 item_id 和 category 到整数索引
    item_cat_df['item_idx'] = item_cat_df['item_id'].map(item_map)
    item_cat_df['brand_idx'] = item_cat_df['category'].map(category_map)
    
    item_cat_df.dropna(subset=['item_idx'], inplace=True)
    item_cat_df['item_idx'] = item_cat_df['item_idx'].astype(int)

    # --- 4. 划分数据集 (使用留一法) ---
    print("\nStep 4: Splitting data into training and testing sets...")
    # 使用 rating 排序来模拟时间，最高分的作为测试集
    df['rank'] = df.groupby('user_idx')['rating'].rank(method='first', ascending=False)
    test_df = df[df['rank'] == 1]
    # 其余的都作为 "大训练集"，之后在 main.py 中动态划分 val
    train_df = df[df['rank'] > 1]
    print(f"Split to {len(train_df)} training interactions and {len(test_df)} testing interactions.")

    # --- 5. 保存处理好的数据 (MODIFIED for multi-category) ---
    output_dir = os.path.join(config['output_base_dir'], f"processed_data_{config['min_interactions']}_pos_only_cat")
    print(f"\nStep 5: Saving processed data to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    train_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    test_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    
    # 保存 item-category 的映射关系
    item_cat_df[['item_idx', 'brand_idx']].to_parquet(os.path.join(output_dir, 'item_brand.parquet'), index=False)
    # 保存 embedding 数据
    # 将 item_id 映射为 item_idx
    embeddings_to_save = {item_map[k]: v for k, v in meta_embeddings.items() if k in item_map}
    # 创建一个 numpy 数组，大小为 (num_items, embd_dim)
    # 假设所有 embedding 维度相同
    if embeddings_to_save:
        embd_dim = len(next(iter(embeddings_to_save.values())))
        item_embeddings_matrix = np.zeros((len(item_map), embd_dim), dtype=np.float32)
        for item_idx, embd in embeddings_to_save.items():
            item_embeddings_matrix[item_idx] = embd
        np.save(os.path.join(output_dir, 'item_embeddings.npy'), item_embeddings_matrix)
        print("Item embeddings saved to 'item_embeddings.npy'.")

    # 保存统计信息
    stats = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_brands': len(category_map) # 对应新的类别节点
    }
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

    print("\n--- Data Preparation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Amazon review data.")
    parser.add_argument('--core', type=int, default=20, help="K-core filtering threshold.")
    parser.add_argument('--review_path', type=str, default='dataset/amazon_books_emb/raw_data/amazon_books_20_core_sentiment_20251214.jsonl', help="Path to the review data file with sentiment.")
    parser.add_argument('--meta_path', type=str, default='dataset/amazon_books_emb/raw_data/amazon_books_metadata_w_embd_20251221.jsonl', help="Path to the metadata file with embeddings.")
    parser.add_argument('--output_dir', type=str, default='dataset/amazon_books_emb/', help="Base directory for output.")
    args = parser.parse_args()
    
    prep_config = {
        'review_file': args.review_path,
        'metadata_file': args.meta_path,
        'min_interactions': args.core,
        'output_base_dir': args.output_dir
    }
    prepare_and_save_data(prep_config)