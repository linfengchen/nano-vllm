# CUDA C++ 算子开发完整指南

本指南提供了一个完整的 CUDA C++ 算子开发示例，从编写到测试的全流程。

## 📦 示例内容

本项目包含一个完整的 **Fused SiLU + Multiply** CUDA 算子实现：

```
nanovllm/kernels/cuda_ops/
├── fused_silu_mul.cu      # CUDA C++ kernel实现 (核心)
├── setup.py               # 编译配置
├── __init__.py            # Python接口封装
├── test_cuda_ops.py       # 完整测试套件
└── README.md              # 详细文档
```

## 🚀 快速编译和测试

### 方法1: JIT编译（推荐开发时使用）

```bash
# 在远程机器上
ssh chenlinfeng@100.99.110.183
source /data/nano/bin/activate
cd /data/python/vllm/github/nano-vllm

# 直接运行测试（会自动JIT编译）
python nanovllm/kernels/cuda_ops/test_cuda_ops.py
```

### 方法2: 预编译（推荐生产环境）

```bash
# 进入CUDA算子目录
cd nanovllm/kernels/cuda_ops

# 编译安装
python setup.py install

# 或开发模式（修改代码会立即生效）
python setup.py develop

# 测试
python test_cuda_ops.py
```

## 📖 核心代码解析

### 1. CUDA Kernel实现 (fused_silu_mul.cu)

```cuda
// 设备函数：SiLU激活
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// 基础kernel
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // 融合：silu(gate) * up
        float gate_val = gate[idx];
        float up_val = up[idx];
        float silu_val = silu(gate_val);
        output[idx] = silu_val * up_val;
    }
}
```

**关键优化点**：
- `__restrict__`: 告诉编译器指针不会重叠，允许更激进的优化
- `__forceinline__`: 强制内联设备函数
- 向量化版本 (float4): 一次处理4个元素

### 2. C++ Wrapper函数

```cuda
torch::Tensor fused_silu_mul_forward(
    torch::Tensor gate,
    torch::Tensor up) {
    
    // 参数检查
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be CUDA tensor");
    
    auto output = torch::empty_like(gate);
    int n_elements = gate.numel();
    
    // 启动kernel
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    fused_silu_mul_kernel<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
    );
    
    return output;
}
```

### 3. PyBind11绑定

```cuda
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul_forward", 
          &fused_silu_mul_forward, 
          "Fused SiLU + Multiply forward (CUDA)");
}
```

### 4. Python接口封装 (__init__.py)

```python
def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU + Multiply"""
    if CUDA_OPS_AVAILABLE and gate.is_cuda:
        return _cuda_ops.fused_silu_mul_forward(gate, up)
    else:
        # Fallback
        return torch.nn.functional.silu(gate) * up
```

## 🎯 实际应用示例

### 在MLP层中使用

```python
import torch
from torch import nn
from nanovllm.kernels.cuda_ops import fused_silu_mul

class OptimizedGatedMLP(nn.Module):
    """使用CUDA优化的Gated MLP"""
    
    def __init__(self, hidden_size=4096, intermediate_size=11008):
        super().__init__()
        # 使用merged projection减少内存访问
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size
    
    def forward(self, x):
        # 一次矩阵乘法获得gate和up
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        
        # 使用CUDA优化的fused kernel
        # 替代: intermediate = F.silu(gate) * up
        intermediate = fused_silu_mul(gate, up)
        
        return self.down_proj(intermediate)
```

### Benchmark对比

```python
import torch
import torch.nn.functional as F
import time

# 测试数据
gate = torch.randn(16, 128, 4096, device='cuda')
up = torch.randn(16, 128, 4096, device='cuda')

# CUDA版本
start = time.time()
for _ in range(100):
    output_cuda = fused_silu_mul(gate, up)
torch.cuda.synchronize()
time_cuda = time.time() - start

# PyTorch版本
start = time.time()
for _ in range(100):
    output_torch = F.silu(gate) * up
torch.cuda.synchronize()
time_torch = time.time() - start

print(f"CUDA: {time_cuda*10:.3f}ms")
print(f"PyTorch: {time_torch*10:.3f}ms")
print(f"Speedup: {time_torch/time_cuda:.2f}x")
```

## 🔧 编译参数详解

### NVCC编译选项 (setup.py)

```python
extra_compile_args={
    'nvcc': [
        '-O3',                          # 最高优化等级
        '--use_fast_math',              # 使用快速数学库
        '-std=c++14',                   # C++14标准
        
        # GPU架构支持
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
        '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
        
        # 高级特性
        '--expt-relaxed-constexpr',     # 放松constexpr限制
        '--expt-extended-lambda',       # 扩展lambda支持
    ]
}
```

## 📊 性能分析

### 期望性能提升

在典型场景下（A100 GPU, batch=16, seq_len=128, hidden=4096）：

| 指标 | PyTorch | CUDA Fused | 改进 |
|------|---------|------------|------|
| 延迟 | ~0.15ms | ~0.10ms | 1.5x faster |
| 内存 | 3 次访问 | 1 次访问 | 67% reduction |
| 带宽利用率 | 60% | 90% | 50% improvement |

### 为什么能加速？

1. **减少kernel启动次数**: 2次 → 1次
2. **减少内存访问**: 
   - PyTorch: load gate → store silu(gate) → load silu(gate) → load up → store result
   - CUDA: load gate → load up → store result
3. **提高缓存命中率**: 数据在寄存器中直接处理

## 🐛 常见问题和解决方案

### 问题1: 编译错误 "cannot find -lcuda"

**解决**:
```bash
# 检查CUDA路径
which nvcc
echo $CUDA_HOME

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 问题2: JIT编译很慢

**解决**: 使用预编译
```bash
python setup.py install
```

### 问题3: 运行时错误 "CUDA error: invalid device function"

**原因**: GPU架构不匹配

**解决**: 在setup.py中添加你的GPU架构
```python
'-gencode=arch=compute_XX,code=sm_XX'  # 替换XX为你的架构
```

查找GPU架构：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## 🎓 进阶优化技巧

### 1. Shared Memory优化

```cuda
__global__ void optimized_kernel(...) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    // 使用共享内存减少全局内存访问
    shared_data[threadIdx.x] = global_data[idx];
    __syncthreads();
    
    // 处理...
}
```

### 2. Warp-level优化

```cuda
#include <cuda_fp16.h>

__global__ void warp_optimized(...) {
    // 使用warp shuffle减少shared memory使用
    float val = data[idx];
    val = __shfl_xor_sync(0xffffffff, val, 1);
}
```

### 3. Tensor Core加速 (FP16)

```cuda
#include <mma.h>
using namespace nvcuda;

// 使用Tensor Core进行矩阵乘法
wmma::fragment<...> a, b, c;
wmma::load_matrix_sync(a, ...);
wmma::load_matrix_sync(b, ...);
wmma::mma_sync(c, a, b, c);
```

## 📚 学习资源

### 推荐阅读
1. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### 相关工具
- **Nsight Compute**: CUDA kernel性能分析
- **cuda-gdb**: CUDA调试器
- **nvprof**: CUDA profiler

## 🎯 下一步

1. **运行测试**: `python nanovllm/kernels/cuda_ops/test_cuda_ops.py`
2. **查看性能**: 对比CUDA和PyTorch实现
3. **集成到模型**: 在实际MLP层中使用
4. **开发新算子**: 参考这个示例实现其他fusion算子

## 📝 总结

这个示例展示了：
- ✅ 完整的CUDA C++算子开发流程
- ✅ 从kernel实现到Python接口的完整链路
- ✅ 性能优化技巧（向量化、FP16支持）
- ✅ 完善的测试和文档

你现在可以基于这个模板开发自己的CUDA算子了！

---

**作者**: nano-vllm团队  
**更新时间**: 2025-11-27
