#!/bin/bash
# 在远程服务器上运行 benchmark 脚本

echo "正在连接到远程服务器 100.99.110.150..."
ssh chenlinfeng@100.99.110.150 << 'ENDSSH'
# 加载环境变量
source .env

# 激活虚拟环境
source /data/venv/bin/activate

# 进入项目目录
cd /data/python/vllm/github/nano-vllm

echo "当前目录: $(pwd)"
echo "使用 Python: $(which python)"
echo "Python 版本: $(python --version)"
echo ""
echo "开始运行 benchmark..."
echo "========================================"

# 运行 benchmark
python bench.py

echo "========================================"
echo "Benchmark 完成"
ENDSSH
