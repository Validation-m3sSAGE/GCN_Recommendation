##快速开始

git clone https://github.com/Validation-m3sSAGE/GCN_Recommendation.git
cd GCN_Recommendation

##获取数据集

#######超级慢

mkdir dataset
cd dataset
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz
cd ..

##运行

#####测试阶段仅测试能否运行，不保存checkpoints

python lgcn.py