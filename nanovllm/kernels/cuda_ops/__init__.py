"""
CUDA算子Python接口

支持JIT编译和预编译两种方式
"""

import os
import torch
import torch.nn as nn

# 尝试导入预编译的版本
CUDA_OPS_AVAILABLE = False
try:
    import cuda_ops as _cuda_ops
    CUDA_OPS_AVAILABLE = True
    print("[CUDA Ops] Using pre-compiled CUDA operators")
except ImportError:
    # 尝试JIT编译
    try:
        from torch.utils.cpp_extension import load
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        _cuda_ops = load(
            name="cuda_ops",
            sources=[
                os.path.join(current_dir, "fused_silu_mul.cu"),
            ],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-std=c++14',
            ],
            verbose=False,
        )
        CUDA_OPS_AVAILABLE = True
        print("[CUDA Ops] Using JIT-compiled CUDA operators")
    except Exception as e:
        print(f"[CUDA Ops] Failed to load CUDA operators: {e}")
        print("[CUDA Ops] Falling back to PyTorch implementations")
        _cuda_ops = None


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU + Element-wise Multiply
    
    计算: output = silu(gate) * up
    
    相比分离操作的优势：
    1. 减少内存访问：一次kernel调用完成两个操作
    2. 减少中间结果存储
    3. 提高内存带宽利用率
    
    Args:
        gate: 输入tensor，将被应用SiLU激活
        up: 输入tensor，与激活后的gate相乘
        
    Returns:
        output: silu(gate) * up
        
    Example:
        >>> gate = torch.randn(16, 128, 4096, device='cuda')
        >>> up = torch.randn(16, 128, 4096, device='cuda')
        >>> output = fused_silu_mul(gate, up)
    """
    if CUDA_OPS_AVAILABLE and gate.is_cuda:
        # 使用CUDA kernel
        return _cuda_ops.fused_silu_mul_forward(gate, up)
    else:
        # Fallback到PyTorch实现
        return torch.nn.functional.silu(gate) * up


class FusedSiLUMul(nn.Module):
    """
    Fused SiLU + Multiply层
    
    用于Gated MLP等场景
    """
    
    def __init__(self):
        super().__init__()
        self.use_cuda_kernel = CUDA_OPS_AVAILABLE
    
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: [batch, seq_len, hidden_size]
            up: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        return fused_silu_mul(gate, up)
    
    def extra_repr(self) -> str:
        return f'using_cuda_kernel={self.use_cuda_kernel}'


__all__ = [
    'fused_silu_mul',
    'FusedSiLUMul',
    'CUDA_OPS_AVAILABLE',
]
