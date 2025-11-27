#!/bin/bash

# NanoVLLM API Server 启动脚本
# 用法: ./start_api_server.sh /path/to/model [port]

MODEL_PATH=${1}
PORT=${2:-8000}

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请提供模型路径"
    echo "用法: $0 <model_path> [port]"
    echo "示例: $0 /path/to/model 8000"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "启动 NanoVLLM OpenAI API Server"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "API 文档: http://localhost:$PORT/docs"
echo "=========================================="
echo ""

python -m nanovllm.entrypoints.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-num-seqs 512 \
    --max-model-len 4096 \
    --tensor-parallel-size 1
