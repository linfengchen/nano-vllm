# 自定义CUDA Kernel快速开始指南

本指南帮助你快速上手在nano-vllm框架中使用和开发自定义CUDA kernel。

## 📚 文档结构

1. **CUSTOM_CUDA_KERNEL_GUIDE.md** - 完整的实现指南（CUDA C++和Triton）
2. **本文档** - 快速开始和常见用例

## 🚀 快速开始

### 1. 测试现有的自定义Kernel

```bash
# 运行完整的测试套件
python test_custom_kernels.py
```

这将测试：
- ✓ Kernel正确性验证
- ✓ 性能benchmark
- ✓ 内存使用分析
- ✓ Profiler分析

### 2. 测试自定义MLP层

```bash
# 测试使用自定义kernel的MLP层
python -m nanovllm.layers.custom_mlp
```

### 3. 在Benchmark中使用Profiling

```bash
# 运行带profiling的benchmark
python bench.py
```

输出文件：
- `profile_trace.json` - Chrome trace文件（在chrome://tracing中查看）
- `profile_report.txt` - 文本格式的详细报告

## 📖 使用示例

### 示例1: 直接使用Kernel

```python
import torch
from nanovllm.kernels import fused_add_gelu, element_wise_mul_add

# Fused Add + GELU
x = torch.randn(16, 128, 4096, device='cuda')
bias = torch.randn(4096, device='cuda')
output = fused_add_gelu(x, bias)

# Element-wise Mul + Add
a = torch.randn(16, 128, 4096, device='cuda')
b = torch.randn(16, 128, 4096, device='cuda')
c = torch.randn(16, 128, 4096, device='cuda')
result = element_wise_mul_add(a, b, c)  # a * b + c
```

### 示例2: 在自定义层中使用

```python
from torch import nn
from nanovllm.kernels import fused_add_gelu

class MyCustomLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        x = self.linear(x)
        # 使用自定义kernel而不是 F.gelu(x + bias)
        return fused_add_gelu(x, self.bias)
```

### 示例3: 集成到现有模型

在`nanovllm/models/qwen3.py`中使用自定义MLP：

```python
from nanovllm.layers.custom_mlp import CustomGatedMLP

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        
        # 替换标准MLP为自定义优化版本
        self.mlp = CustomGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_custom_kernels=True,
        )
        # ...
```

## 🔧 开发新的Kernel

### 方法1: 使用Triton（推荐）

```python
import triton
import triton.language as tl

@triton.jit
def my_custom_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 你的计算逻辑
    output = x * 2.0  # 示例
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

# Python接口
def my_custom_op(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    my_custom_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### 方法2: 使用CUDA C++（最高性能）

参考`CUSTOM_CUDA_KERNEL_GUIDE.md`中的详细说明。

## 📊 性能优化建议

### 1. Benchmark你的Kernel

```python
from nanovllm.kernels import benchmark_kernel

# 对比自定义kernel和PyTorch实现
time_custom = benchmark_kernel(my_custom_op, x, warmup=20, iters=100)
time_torch = benchmark_kernel(torch_reference, x, warmup=20, iters=100)

print(f"加速比: {time_torch/time_custom:.2f}x")
```

### 2. 使用Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = my_custom_op(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

### 3. 优化建议

- **Kernel融合**: 合并多个操作减少内存访问
- **内存布局**: 确保tensor是连续的（`contiguous()`）
- **Block大小**: 使用`triton.next_power_of_2()`自动调优
- **数值精度**: 根据需求使用FP16/BF16

## 🎯 常见使用场景

### 场景1: 融合算子加速

**问题**: 多个小算子频繁访问内存
**解决**: 使用fused kernel

```python
# 慢: 两次内存访问
x = x + bias
x = F.gelu(x)

# 快: 一次内存访问
x = fused_add_gelu(x, bias)
```

### 场景2: 自定义激活函数

```python
@triton.jit
def custom_activation_kernel(...):
    # 实现自定义激活函数
    # 例如: Swish, Mish, 或其他
    pass
```

### 场景3: 优化Attention计算

框架已经在`nanovllm/layers/attention.py`中使用Triton优化KV cache存储。

## 🐛 调试技巧

### 1. 验证正确性

```python
# 对比自定义kernel和参考实现
output_custom = my_custom_op(x)
output_reference = reference_implementation(x)

assert torch.allclose(output_custom, output_reference, rtol=1e-5)
```

### 2. 打印中间结果

```python
print(f"Input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
print(f"Output: shape={output.shape}")
print(f"Max diff: {(output_custom - output_reference).abs().max()}")
```

### 3. 检查内存

```python
torch.cuda.reset_peak_memory_stats()
output = my_custom_op(x)
mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory: {mem:.2f} MB")
```

## 📈 集成到生产环境

### 步骤1: 充分测试

```bash
# 运行所有测试
python test_custom_kernels.py

# Benchmark不同配置
python bench.py
```

### 步骤2: 添加Fallback

```python
def safe_custom_op(x):
    if x.is_cuda and KERNELS_AVAILABLE:
        return custom_kernel(x)
    else:
        return torch_fallback(x)
```

### 步骤3: 监控性能

```python
import time

start = time.time()
output = model(input)
latency = time.time() - start

print(f"Latency: {latency*1000:.2f}ms")
```

## 📚 参考资源

### 官方文档
- [Triton文档](https://triton-lang.org/)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 框架内部参考
- `nanovllm/layers/attention.py` - Triton kernel示例
- `nanovllm/kernels/triton_ops.py` - 自定义kernel实现
- `nanovllm/layers/custom_mlp.py` - 模型集成示例

## 💡 最佳实践

1. ✅ 总是先benchmark，确保有实际加速
2. ✅ 添加完善的错误处理和fallback
3. ✅ 编写单元测试验证正确性
4. ✅ 使用profiler找到真正的瓶颈
5. ✅ 优先优化热点路径（forward pass中频繁调用的算子）
6. ✅ 考虑不同的batch size和序列长度

## 🔍 性能检查清单

- [ ] Kernel输出正确性已验证
- [ ] 对比PyTorch实现有明显加速（>1.2x）
- [ ] 在不同输入尺寸下都表现良好
- [ ] 内存使用合理
- [ ] 添加了fallback实现
- [ ] 已使用profiler分析
- [ ] 代码有适当的注释和文档

## 🚦 下一步

1. **学习**: 阅读`CUSTOM_CUDA_KERNEL_GUIDE.md`了解详细实现
2. **实践**: 运行`test_custom_kernels.py`查看示例
3. **开发**: 基于模板创建你自己的kernel
4. **集成**: 将优化后的kernel集成到模型中
5. **测试**: 在实际workload上benchmark性能

---

**问题或建议？** 

查看项目Issues或提交新的Issue。Happy optimizing! 🚀
