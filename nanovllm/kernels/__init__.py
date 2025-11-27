"""
自定义CUDA Kernels模块

提供了使用Triton实现的高性能CUDA算子，用于加速LLM推理。

可用的kernel:
- fused_add_gelu: 融合的Add + GELU操作
- element_wise_mul_add: Element-wise的 a*b+c 操作
- fused_residual_rmsnorm: 融合的Residual + RMSNorm操作

使用示例:
    from nanovllm.kernels import fused_add_gelu
    
    output = fused_add_gelu(x, bias)
"""

try:
    from .triton_ops import (
        fused_add_gelu,
        element_wise_mul_add,
        fused_residual_rmsnorm,
        benchmark_kernel,
    )
    
    __all__ = [
        'fused_add_gelu',
        'element_wise_mul_add',
        'fused_residual_rmsnorm',
        'benchmark_kernel',
    ]
    
    KERNELS_AVAILABLE = True
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"无法导入自定义kernels: {e}\n"
        "将使用PyTorch标准实现。请确保已安装triton。",
        ImportWarning
    )
    
    KERNELS_AVAILABLE = False
    __all__ = []
