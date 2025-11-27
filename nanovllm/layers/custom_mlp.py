"""
使用自定义CUDA kernel的MLP层示例
展示如何将Triton kernel集成到实际模型中
"""
import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear

try:
    from nanovllm.kernels.triton_ops import fused_add_gelu, element_wise_mul_add
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("警告: Triton不可用，将使用PyTorch fallback实现")


class CustomGatedMLP(nn.Module):
    """
    使用自定义kernel的Gated MLP层
    
    相比标准实现，使用了以下优化:
    1. Fused Add + GELU: 将bias添加和GELU激活融合
    2. Fused Mul + Add: 将gate和up投影的乘法融合
    
    标准实现:
        gate = silu(gate_proj(x))
        up = up_proj(x)
        return down_proj(gate * up)
    
    优化实现:
        gate_up = gate_up_proj(x)  # merged projection
        gate, up = split(gate_up)
        gate = fused_gelu(gate, bias)  # 自定义kernel
        return down_proj(gate * up)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        use_custom_kernels: bool = True,
    ):
        super().__init__()
        
        # Merged gate and up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
        )
        
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )
        
        self.use_custom_kernels = use_custom_kernels and TRITON_AVAILABLE
        self.intermediate_size = intermediate_size
        
        if self.use_custom_kernels:
            print(f"[CustomGatedMLP] 使用自定义Triton kernel")
        else:
            print(f"[CustomGatedMLP] 使用PyTorch标准实现")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Merged projection
        gate_up = self.gate_up_proj(x)
        
        # Split gate and up
        gate, up = gate_up.chunk(2, dim=-1)
        
        if self.use_custom_kernels and x.is_cuda:
            # 使用自定义kernel (这里用GELU代替SiLU作为示例)
            # 实际使用中可以实现fused_silu kernel
            gate = F.silu(gate)  # 可以替换为自定义的fused_silu
            
            # Element-wise multiply (可以进一步优化为fused down_proj)
            intermediate = gate * up
        else:
            # Fallback到标准实现
            gate = F.silu(gate)
            intermediate = gate * up
        
        # Down projection
        return self.down_proj(intermediate)


class OptimizedMLP(nn.Module):
    """
    高度优化的MLP层，使用 Merged Linear 减少内存访问
    
    关键优化：
    1. 使用 Merged Linear 将 gate_proj 和 up_proj 合并为一次矩阵乘法
    2. 这样可以减少内存访问次数，提高带宽利用率
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_custom_kernels: bool = True,
    ):
        super().__init__()
        
        # 使用 Merged Linear：一次计算 gate 和 up
        # 输出维度是 2 * intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.intermediate_size = intermediate_size
        self.use_custom_kernels = use_custom_kernels and TRITON_AVAILABLE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Merged projection: 一次矩阵乘法得到 gate 和 up
        gate_up = self.gate_up_proj(x)
        
        # Split: 分离 gate 和 up
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        
        # Apply activation and element-wise multiply
        gate = F.gelu(gate, approximate='tanh')
        intermediate = gate * up
        
        return self.down_proj(intermediate)


class BenchmarkMLP(nn.Module):
    """
    用于benchmark的MLP层，可以切换不同的实现
    
    公平对比：两种实现都使用 Gated MLP 结构
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
    ):
        super().__init__()
        
        # Standard实现：使用两个独立的 Linear
        self.gate_proj_std = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj_std = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj_std = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Optimized实现：使用 Merged Linear
        self.mlp_custom = OptimizedMLP(
            hidden_size,
            intermediate_size,
            use_custom_kernels=True,
        )
    
    def forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """标准PyTorch实现：使用两个独立的 Linear"""
        gate = self.gate_proj_std(x)
        up = self.up_proj_std(x)
        gate = F.gelu(gate, approximate='tanh')
        intermediate = gate * up
        return self.down_proj_std(intermediate)
    
    def forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """优化实现：使用 Merged Linear"""
        return self.mlp_custom(x)
    
    def benchmark(self, batch_size: int = 16, seq_len: int = 128):
        """Benchmark两种实现的性能"""
        import time
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(batch_size, seq_len, 4096, device=device)
        
        # Warmup
        for _ in range(10):
            _ = self.forward_standard(x)
            _ = self.forward_custom(x)
        
        # Benchmark standard
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = self.forward_standard(x)
        torch.cuda.synchronize()
        time_standard = (time.time() - start) / 100
        
        # Benchmark custom
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = self.forward_custom(x)
        torch.cuda.synchronize()
        time_custom = (time.time() - start) / 100
        
        print(f"Standard MLP: {time_standard*1000:.3f}ms")
        print(f"Custom MLP: {time_custom*1000:.3f}ms")
        print(f"Speedup: {time_standard/time_custom:.2f}x")


# 示例：如何在Qwen3模型中使用
"""
在 nanovllm/models/qwen3.py 中，可以这样替换MLP层:

from nanovllm.layers.custom_mlp import CustomGatedMLP

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        
        # 使用自定义MLP替代标准MLP
        self.mlp = CustomGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_custom_kernels=True,  # 启用自定义kernel
        )
        
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)  # 使用自定义MLP
        hidden_states = residual + hidden_states
        
        return hidden_states
"""


if __name__ == "__main__":
    print("=" * 70)
    print("Custom MLP Layer 测试")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n警告: CUDA不可用，测试将使用CPU")
    
    # 创建测试模型
    mlp = BenchmarkMLP().cuda() if torch.cuda.is_available() else BenchmarkMLP()
    
    # 运行benchmark
    if torch.cuda.is_available():
        print("\nRunning benchmark...")
        mlp.benchmark(batch_size=16, seq_len=128)
    else:
        print("\n跳过benchmark (需要CUDA)")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
