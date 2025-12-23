import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

def prepare_and_save_data(config):
    """
    一个完整的预处理流程：
    1. 加载和过滤评论
    2. 加载元数据
    3. 创建映射和索引
    4. 划分数据集
    5. 将所有处理好的数据保存为高效的 Parquet 格式
    """
    print("--- Starting Data Preparation ---")
    
    # --- 1. 加载和过滤评论数据 ---
    print(f"Step 1: Loading and filtering reviews from '{config['review_file']}'...")
    reviews = []
    
    # 如果 review_file 很大，这里可以考虑分块加载或限制加载行数进行快速测试
    # with open(config['review_file'], 'r') as f:
    #     for i, line in enumerate(tqdm(f, desc="Loading Reviews")):
    #         if i >= 5000000: # 限制加载行数进行调试
    #             break
    #         reviews.append(json.loads(line.strip()))

    with open(config['review_file'], 'r') as f:
        reviews = [json.loads(line.strip()) for line in tqdm(f, desc="Loading Reviews")]

    df = pd.DataFrame(reviews)[['user_id', 'parent_asin', 'rating']]
    df.rename(columns={'parent_asin': 'item_id'}, inplace=True)
    df.dropna(inplace=True)
    print(f"Loaded {len(df)} interactions initially.")

    # K-core filtering
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        weak_users = user_counts[user_counts < config['min_interactions']].index
        weak_items = item_counts[item_counts < config['min_interactions']].index
        if len(weak_users) == 0 and len(weak_items) == 0:
            break
        df = df[~df['user_id'].isin(weak_users)]
        df = df[~df['item_id'].isin(weak_items)]
    print(f"Filtered to {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items.")

    # --- 2. 加载元数据 ---
    print(f"\nStep 2: Loading metadata from '{config['metadata_file']}'...")
    active_items_set = set(df['item_id'].unique())
    meta_data = {}
    with open(config['metadata_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Metadata"):
            record = json.loads(line.strip())
            if record.get('parent_asin') in active_items_set:
                #brand = record.get('details', {}).get('Brand', 'Unknown')
                author = record.get('author')  # 先获取author值（可能是None/字典/其他）
                if isinstance(author, dict):  # 仅当author是字典时，取name
                    brand = author.get('name', 'Unknown')
                else:  # author是None/字符串/列表等，直接填Unknown
                    brand = 'Unknown'
                
                meta_data[record['parent_asin']] = brand
    
    # --- 3. 创建映射和处理索引 ---
    print("\nStep 3: Creating ID maps and indexing data...")
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    item_brand_list = [{'item_id': k, 'brand': v} for k, v in meta_data.items()]
    item_brand_df = pd.DataFrame(item_brand_list)
    
    # ===== 新增：验证作者（新品牌）的统计 =====
    print("\n=== Brand Statistics ===")
    print(f"Total unique: {item_brand_df['brand'].nunique()}")
    print(f"Value counts (Top 10):\n{item_brand_df['brand'].value_counts().head(10)}")
    print(f"Unknown ratio: {round(item_brand_df['brand'].value_counts()['Unknown'] / len(item_brand_df) * 100, 2)}%")
    # =========================================

    brand_map = {brand: i for i, brand in enumerate(item_brand_df['brand'].unique())}
    
    item_brand_df['item_idx'] = item_brand_df['item_id'].map(item_map)
    item_brand_df['brand_idx'] = item_brand_df['brand'].map(brand_map)
    
    item_brand_df.dropna(subset=['item_idx'], inplace=True)
    item_brand_df['item_idx'] = item_brand_df['item_idx'].astype(int)

    # --- 4. 划分数据集 ---
    print("\nStep 4: Splitting data into training and testing sets...")
    df['rank_latest'] = df.groupby(['user_idx'])['rating'].rank(method='first', ascending=False)
    test_df = df[df['rank_latest'] == 1]
    train_df = df[df['rank_latest'] > 1]

    # --- 5. 保存处理好的数据 ---
    output_dir = os.path.join(config['output_base_dir'], f"processed_data_{config['min_interactions']}")
    print(f"\nStep 5: Saving processed data to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    train_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    test_df[['user_idx', 'item_idx']].to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    item_brand_df[['item_idx', 'brand_idx']].to_parquet(os.path.join(output_dir, 'item_brand.parquet'), index=False)
    
    stats = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_brands': len(brand_map)
    }
    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

    print("\n--- Data Preparation Finished ---")
    print(f"Data for {config['min_interactions']}-core filtering saved in '{output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Amazon review data.")
    parser.add_argument('--core', type=int, default=20, help="K-core filtering threshold.")
    parser.add_argument('--review_path', type=str, default='dataset/amazon_books/raw_data/Books.jsonl', help="Path to the review data file.")
    parser.add_argument('--meta_path', type=str, default='dataset/amazon_books/raw_data/meta_Books.jsonl', help="Path to the metadata file.")
    parser.add_argument('--output_dir', type=str, default='dataset/amazon_books/', help="Base directory for output.")
    args = parser.parse_args()
    
    # 确保已安装: pip install pandas pyarrow tqdm
    prep_config = {
        'review_file': args.review_path,
        'metadata_file': args.meta_path,
        'min_interactions': args.core,
        'output_base_dir': args.output_dir
    }
    prepare_and_save_data(prep_config)