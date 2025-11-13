# preprocess_metadata.py
import json
from tqdm import tqdm

def create_filtered_metadata(review_file, metadata_file, output_file):
    """
    根据评论文件中存在的物品，过滤元数据文件。
    """
    print(f"Step 1: Scanning review file '{review_file}' to find all active item ASINs...")
    
    # 使用集合来高效存储和查找唯一的 parent_asin
    active_parent_asins = set()
    
    with open(review_file, 'r') as f:
        # 使用tqdm来显示进度条
        for line in tqdm(f, desc="Scanning Reviews"):
            try:
                review = json.loads(line)
                # 确保 parent_asin 存在
                if 'parent_asin' in review:
                    active_parent_asins.add(review['parent_asin'])
            except json.JSONDecodeError:
                # 忽略格式错误的行
                continue

    print(f"Found {len(active_parent_asins)} unique parent ASINs in the review file.")
    
    print(f"\nStep 2: Filtering metadata file '{metadata_file}'...")
    
    written_count = 0
    with open(metadata_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, desc="Filtering Metadata"):
            try:
                meta = json.loads(line)
                # 检查当前元数据的 parent_asin 是否在我们收集的集合中
                if meta.get('parent_asin') in active_parent_asins:
                    # 如果是，则将该行写入新文件
                    outfile.write(line.strip() + '\n')
                    written_count += 1
            except json.JSONDecodeError:
                continue
                
    print(f"\nPreprocessing complete.")
    print(f"Wrote {written_count} relevant metadata records to '{output_file}'.")

if __name__ == '__main__':
    # --- 配置 ---
    # 确保这些路径是正确的
    REVIEW_FILE_PATH = 'dataset/Books.jsonl'
    METADATA_FILE_PATH = 'dataset/meta_Books.jsonl'
    OUTPUT_METADATA_PATH = 'dataset/meta_Books_filtered.jsonl' # 这是我们将生成的新文件
    
    create_filtered_metadata(REVIEW_FILE_PATH, METADATA_FILE_PATH, OUTPUT_METADATA_PATH)