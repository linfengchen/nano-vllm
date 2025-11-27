# NanoVLLM OpenAI API Server

这是 nanovllm 的 OpenAI 兼容 API server 实现，参考了 vLLM 的设计模式。

## 功能特性

- ✅ 兼容 OpenAI API 格式
- ✅ 支持 Chat Completions API (`/v1/chat/completions`)
- ✅ 支持 Text Completions API (`/v1/completions`)
- ✅ 支持流式和非流式响应
- ✅ 支持模型列表查询 (`/v1/models`)
- ✅ 健康检查端点 (`/health`)
- ✅ 自动 API 文档 (FastAPI Swagger UI)

## 安装依赖

首先确保安装了必要的依赖：

```bash
pip install fastapi uvicorn pydantic openai
```

或者重新安装 nanovllm：

```bash
pip install -e .
```

## 启动 API Server

### 方法 1: 使用命令行启动

```bash
python -m nanovllm.entrypoints.api_server \
    --model /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 512 \
    --max-model-len 4096 \
    --tensor-parallel-size 1
```

### 方法 2: 使用 Python 代码启动

```python
from nanovllm.entrypoints.api_server import run_server

run_server(
    model="/path/to/your/model",
    host="0.0.0.0",
    port=8000,
    max_num_seqs=512,
    max_model_len=4096,
    tensor_parallel_size=1,
)
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必需) | 模型路径 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--max-num-seqs` | `512` | 最大序列数 |
| `--max-model-len` | `4096` | 最大模型长度 |
| `--tensor-parallel-size` | `1` | 张量并行大小 |

## 使用 API

### 1. 使用 OpenAI Python 客户端

```python
import openai

# 配置客户端指向本地服务器
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # nanovllm 不需要 API key，但客户端要求提供
)

# Chat Completions
response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Text Completions
response = client.completions.create(
    model="qwen",
    prompt="Once upon a time,",
    max_tokens=50,
    temperature=0.7,
)
print(response.choices[0].text)

# Streaming
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Tell me a story."}],
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 2. 使用 curl

#### Chat Completions (非流式)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Chat Completions (流式)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "Tell me a story."}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

#### Text Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "prompt": "Once upon a time,",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### 列出模型

```bash
curl http://localhost:8000/v1/models
```

#### 健康检查

```bash
curl http://localhost:8000/health
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat completions API |
| `/v1/completions` | POST | Text completions API |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | API 文档 (Swagger UI) |

## 请求参数

### Chat Completions

```json
{
  "model": "string",              // 模型名称
  "messages": [                   // 消息列表
    {
      "role": "system|user|assistant",
      "content": "string"
    }
  ],
  "temperature": 1.0,             // 采样温度 (可选)
  "max_tokens": 64,               // 最大生成 token 数 (可选)
  "stream": false,                // 是否流式输出 (可选)
  "ignore_eos": false             // 是否忽略 EOS token (可选)
}
```

### Text Completions

```json
{
  "model": "string",              // 模型名称
  "prompt": "string",             // 输入提示词
  "temperature": 1.0,             // 采样温度 (可选)
  "max_tokens": 64,               // 最大生成 token 数 (可选)
  "stream": false,                // 是否流式输出 (可选)
  "echo": false,                  // 是否回显输入 (可选)
  "ignore_eos": false             // 是否忽略 EOS token (可选)
}
```

## 响应格式

### Chat Completions 响应

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "生成的文本内容"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Text Completions 响应

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "qwen",
  "choices": [
    {
      "index": 0,
      "text": "生成的文本内容",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

## 运行示例

我们提供了一个完整的示例脚本 `example_api_server.py`：

```bash
# 1. 启动 API server（在一个终端）
python -m nanovllm.entrypoints.api_server --model /path/to/model

# 2. 运行示例脚本（在另一个终端）
python example_api_server.py
```

## 架构设计

本实现参考了 vLLM 的设计模式：

```
nanovllm/entrypoints/
├── __init__.py
├── protocol.py          # OpenAI API 协议定义（类似 vLLM 的 protocol.py）
└── api_server.py        # API server 主逻辑（类似 vLLM 的 api_server.py）
```

### 主要组件

1. **protocol.py**: 定义了 OpenAI API 的请求/响应格式
   - 使用 Pydantic 模型进行数据验证
   - 包含 Chat/Text Completions、Models 等协议

2. **api_server.py**: 实现 API server 逻辑
   - `OpenAIServer` 类：封装 nanovllm 引擎和请求处理逻辑
   - FastAPI 应用：提供 REST API 端点
   - 支持流式和非流式响应

### 与 vLLM 的对比

| 特性 | vLLM | nanovllm |
|------|------|----------|
| 协议定义 | ✅ 完整 | ✅ 简化版 |
| Chat API | ✅ | ✅ |
| Completions API | ✅ | ✅ |
| 流式响应 | ✅ 原生支持 | ✅ 模拟实现 |
| 异步处理 | ✅ | ✅ |
| 工具调用 | ✅ | ❌ |
| Embeddings | ✅ | ❌ |

## 注意事项

1. **流式响应实现**: 当前实现是先生成完整响应再进行流式输出的模拟。真实的流式实现需要引擎层面支持逐 token 生成。

2. **聊天模板**: 当前使用简单的文本拼接格式化聊天消息。生产环境建议使用模型自带的 chat template。

3. **错误处理**: 实现了基本的错误处理，返回标准的 OpenAI 错误格式。

4. **性能优化**: 这是一个简化实现，主要用于演示。生产环境可能需要：
   - 真正的异步引擎接口
   - 请求队列管理
   - 连接池管理
   - 更完善的监控和日志

## 扩展开发

如需添加更多功能，可以参考：

1. **添加新端点**: 在 `create_app()` 函数中添加新的路由
2. **支持更多参数**: 扩展 `protocol.py` 中的请求模型
3. **改进流式**: 修改引擎层以支持真正的逐 token 生成
4. **添加认证**: 使用 FastAPI 的依赖注入系统

## 故障排查

### 服务器无法启动

- 检查端口是否被占用
- 确认模型路径是否正确
- 检查依赖是否完整安装

### 请求报错

- 查看服务器日志
- 确认请求格式是否正确
- 检查 API 文档：`http://localhost:8000/docs`

### 性能问题

- 调整 `max_num_seqs` 参数
- 使用适当的 `tensor_parallel_size`
- 监控 GPU 内存使用

## 参考资料

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [vLLM API Server 实现](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
