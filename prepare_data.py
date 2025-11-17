# prepare_data.py
import json
import pandas as pd
from tqdm import tqdm
import os

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
    # 注意：这里我们完整加载评论文件来构建最准确的图
    # 如果内存不足，可以考虑分块处理，但会更复杂
    with open(config['review_file'], 'r') as f:
        for line in tqdm(f, desc="Loading Reviews"):
            reviews.append(json.loads(line.strip()))
    
    df = pd.DataFrame(reviews)[['user_id', 'parent_asin', 'rating']]
    df.rename(columns={'parent_asin': 'item_id'}, inplace=True)
    df.dropna(inplace=True) # 删除缺失值
    print(f"Loaded {len(df)} interactions initially.")

    # K-core filtering
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        weak_users = user_counts[user_counts < config['min_interactions']].index
        weak_items = item_counts[item_counts < config['min_interactions']].index
        if len(weak_users) == 0 and len(weak_items) == 0: break
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
                brand = record.get('details', {}).get('Brand', 'Unknown')
                meta_data[record['parent_asin']] = brand
    
    # --- 3. 创建映射和处理索引 ---
    print("\nStep 3: Creating ID maps and indexing data...")
    user_map = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_map = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    
    item_brand_list = [{'item_id': k, 'brand': v} for k, v in meta_data.items()]
    item_brand_df = pd.DataFrame(item_brand_list)
    brand_map = {brand: i for i, brand in enumerate(item_brand_df['brand'].unique())}
    
    item_brand_df['item_idx'] = item_brand_df['item_id'].map(item_map)
    item_brand_df['brand_idx'] = item_brand_df['brand'].map(brand_map)
    
    # 过滤掉映射后可能产生的NaN (如果meta和review不完全匹配)
    item_brand_df.dropna(subset=['item_idx'], inplace=True)
    item_brand_df['item_idx'] = item_brand_df['item_idx'].astype(int)

    # --- 4. 划分数据集 ---
    print("\nStep 4: Splitting data into training and testing sets...")
    df['rank_latest'] = df.groupby(['user_idx'])['rating'].rank(method='first', ascending=False)
    test_df = df[df['rank_latest'] == 1]
    train_df = df[df['rank_latest'] > 1]

    # --- 5. 保存处理好的数据 ---
    print(f"\nStep 5: Saving processed data to '{config['output_dir']}'...")
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    train_df[['user_idx', 'item_idx']].to_parquet(os.path.join(config['output_dir'], 'train.parquet'), index=False)
    test_df[['user_idx', 'item_idx']].to_parquet(os.path.join(config['output_dir'], 'test.parquet'), index=False)
    item_brand_df[['item_idx', 'brand_idx']].to_parquet(os.path.join(config['output_dir'], 'item_brand.parquet'), index=False)
    
    # 保存映射和统计信息
    stats = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_brands': len(brand_map)
    }
    with open(os.path.join(config['output_dir'], 'stats.json'), 'w') as f:
        json.dump(stats, f)

    print("\n--- Data Preparation Finished ---")


if __name__ == '__main__':
    # 确保已安装: pip install pandas pyarrow
    prep_config = {
        'review_file': 'dataset/amazon_books/raw_data/Books.jsonl',
        'metadata_file': 'dataset/amazon_books/raw_data/meta_Books.jsonl',
        'min_interactions': 1,  # 1-core filtering
        'output_dir': 'dataset/amazon_books/processed_data' # 输出目录
    }
    prepare_and_save_data(prep_config)