# 源JSON文件路径
source_path = "dataset/meta_Books.jsonl"
# 目标JSON文件路径
target_path = "preview_filtered.jsonl"

# 读取前100行并写入新文件
with open(source_path, 'r', encoding='utf-8') as src, \
     open(target_path, 'w', encoding='utf-8') as dst:
    
    count = 0
    for line in src:
        # 只处理前100行
        if count < 100:
            # 写入当前行（保留原始格式）
            dst.write(line)
            count += 1
        else:
            break

print(f"已成功将前{count}行写入{target_path}")