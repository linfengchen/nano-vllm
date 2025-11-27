# 自定义CUDA算子集成指南

本指南展示如何实现自定义CUDA算子并集成到nano-vllm推理框架中。

## 方法一：使用PyTorch C++扩展 (推荐用于CUDA C++)

### 1. 创建CUDA Kernel

创建 `nanovllm/kernels/custom_ops/custom_kernel.cu`：

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel示例：自定义ReLU激活函数
__global__ void custom_relu_kernel(
    const float* input,
    float* output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0.0f;
    }
}

// CUDA kernel示例：Fused Add + ReLU
__global__ void fused_add_relu_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int bias_idx = idx % hidden_size;
        float val = input[idx] + bias[bias_idx];
        output[idx] = val > 0 ? val : 0.0f;
    }
}

// C++ wrapper函数
torch::Tensor custom_relu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    custom_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}

torch::Tensor fused_add_relu_forward(
    torch::Tensor input,
    torch::Tensor bias) {
    
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    fused_add_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1)
    );
    
    return output;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_relu_forward", &custom_relu_forward, "Custom ReLU forward (CUDA)");
    m.def("fused_add_relu_forward", &fused_add_relu_forward, "Fused Add+ReLU forward (CUDA)");
}
```

### 2. 创建C++接口文件

创建 `nanovllm/kernels/custom_ops/custom_ops.cpp`：

```cpp
#include <torch/extension.h>

// 声明CUDA函数
torch::Tensor custom_relu_forward(torch::Tensor input);
torch::Tensor fused_add_relu_forward(torch::Tensor input, torch::Tensor bias);

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_relu_forward", &custom_relu_forward, "Custom ReLU forward (CUDA)");
    m.def("fused_add_relu_forward", &fused_add_relu_forward, "Fused Add+ReLU forward (CUDA)");
}
```

### 3. 创建setup.py编译脚本

创建 `nanovllm/kernels/setup.py`：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_ops',
            sources=[
                'custom_ops/custom_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

### 4. 编译CUDA扩展

```bash
cd nanovllm/kernels
python setup.py install
```

或使用JIT编译（开发时更方便）：

创建 `nanovllm/kernels/__init__.py`：

```python
import os
from torch.utils.cpp_extension import load

# JIT编译
custom_cuda_ops = load(
    name="custom_cuda_ops",
    sources=[
        os.path.join(os.path.dirname(__file__), "custom_ops/custom_kernel.cu"),
    ],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
    ],
    verbose=True
)
```

### 5. 创建Python封装层

创建 `nanovllm/layers/custom_activation.py`：

```python
import torch
import torch.nn as nn
from nanovllm.kernels import custom_cuda_ops


class CustomReLU(nn.Module):
    """使用自定义CUDA kernel的ReLU"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return custom_cuda_ops.custom_relu_forward(x)
        else:
            # Fallback到PyTorch实现
            return torch.relu(x)


class FusedAddReLU(nn.Module):
    """Fused Add + ReLU操作"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return custom_cuda_ops.fused_add_relu_forward(x, self.bias)
        else:
            # Fallback
            return torch.relu(x + self.bias)
```

### 6. 集成到模型中

修改 `nanovllm/models/qwen3.py` 或其他模型文件：

```python
from nanovllm.layers.custom_activation import CustomReLU, FusedAddReLU

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(...)
        self.down_proj = RowParallelLinear(...)
        
        # 使用自定义activation
        self.act_fn = CustomReLU()  # 替代原来的SiLU
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        # 使用自定义kernel
        x = self.act_fn(gate_up)
        return self.down_proj(x)
```

---

## 方法二：使用Triton (推荐用于快速原型开发)

Triton是一个更高级的抽象，比CUDA C++更容易编写和调试。框架中已经使用了Triton（见`attention.py`）。

### 1. 创建Triton Kernel

创建 `nanovllm/kernels/triton_ops.py`：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def custom_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前block的起始位置
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # 创建offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # ReLU操作
    output = tl.where(x > 0, x, 0.0)
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fused_add_relu_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载输入和bias
    x = tl.load(input_ptr + offsets, mask=mask)
    bias_idx = offsets % hidden_size
    bias = tl.load(bias_ptr + bias_idx, mask=mask)
    
    # Fused Add + ReLU
    result = x + bias
    output = tl.where(result > 0, result, 0.0)
    
    # 存储
    tl.store(output_ptr + offsets, output, mask=mask)


# Python接口
def custom_relu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # 自动选择block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    custom_relu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def fused_add_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    hidden_size = x.size(-1)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_relu_kernel[grid](
        x, bias, output,
        n_elements, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

### 2. 集成Triton算子

```python
from nanovllm.kernels.triton_ops import custom_relu, fused_add_relu

class CustomActivation(nn.Module):
    def forward(self, x):
        return custom_relu(x)
```

---

## 方法三：直接使用现有框架中的Triton示例

框架已经在 `nanovllm/layers/attention.py` 中使用了Triton kernel：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # ... kernel实现
```

你可以参考这个模式来添加新的kernel。

---

## 性能优化建议

### 1. 使用Profiler验证性能

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = custom_relu(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2. Benchmark对比

```python
import time

def benchmark(func, input, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        func(input)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        func(input)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / iters

# 对比
input = torch.randn(1024, 4096, device='cuda')
time_custom = benchmark(custom_relu, input)
time_torch = benchmark(torch.relu, input)

print(f"Custom kernel: {time_custom*1000:.3f}ms")
print(f"PyTorch: {time_torch*1000:.3f}ms")
print(f"Speedup: {time_torch/time_custom:.2f}x")
```

### 3. 内存优化

- 使用 `torch.cuda.max_memory_allocated()` 监控内存使用
- 尽量使用in-place操作
- 注意tensor的连续性 (`contiguous()`)

### 4. 数值精度

- 支持FP16/BF16混合精度
- 添加数值稳定性检查

---

## 完整示例：集成自定义Linear层

创建 `nanovllm/layers/custom_linear.py`：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 实现优化的矩阵乘法
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # ... 详细实现
    pass


class CustomLinear(nn.Module):
    """使用自定义kernel的Linear层"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # 使用自定义kernel或fallback到torch.mm
        if x.is_cuda:
            # 调用自定义kernel
            return self.custom_matmul(x, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(x, self.weight, self.bias)
    
    def custom_matmul(self, x, weight, bias):
        # 实现调用Triton kernel的逻辑
        pass
```

---

## 调试技巧

1. **打印中间结果**：
```python
print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
print(f"Output shape: {output.shape}")
```

2. **数值验证**：
```python
output_custom = custom_relu(x)
output_torch = torch.relu(x)
assert torch.allclose(output_custom, output_torch, rtol=1e-5)
```

3. **使用CUDA错误检查**：
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
```

---

## 总结

- **CUDA C++**: 最高性能，但开发复杂度高
- **Triton**: 平衡性能和开发效率，推荐用于大多数场景
- **PyTorch扩展**: 便于集成现有CUDA代码

选择建议：
1. 原型开发阶段：使用Triton
2. 性能关键路径：使用CUDA C++优化
3. 已有CUDA代码：使用PyTorch C++扩展封装
