"""
自定义Triton CUDA算子示例
展示如何在nano-vllm框架中实现和集成自定义kernel
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_gelu_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Add + GELU操作 (使用tanh近似，与PyTorch一致)
    output = GELU(input + bias)
    
    使用tanh近似（与PyTorch的approximate='tanh'一致）:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载输入和bias
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    bias_idx = offsets % hidden_size
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    
    # Add
    x = x + bias
    
    # GELU tanh近似（与PyTorch一致）
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    x_cubed = x * x * x
    tanh_input = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    
    # 手动实现tanh，使用数值稳定的方式
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # 为了数值稳定性，使用：
    # tanh(x) = -1 + 2 / (1 + exp(-2x)) for x >= 0
    # tanh(x) = 1 - 2 / (1 + exp(2x)) for x < 0
    tanh_val = tl.where(
        tanh_input >= 0.0,
        -1.0 + 2.0 / (1.0 + tl.exp(-2.0 * tanh_input)),
        1.0 - 2.0 / (1.0 + tl.exp(2.0 * tanh_input))
    )
    
    output = 0.5 * x * (1.0 + tanh_val)
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fused_residual_rmsnorm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Residual + RMSNorm操作
    
    1. residual = input + residual
    2. output = residual * weight / rms(residual)
    
    其中 rms(x) = sqrt(mean(x^2) + eps)
    """
    row_idx = tl.program_id(0)
    
    # 计算当前行的偏移
    row_start = row_idx * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    
    # 加载输入和residual
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    # 残差连接
    residual = x + residual
    
    # 计算RMS - 使用正确的归约方法
    # 首先计算平方和
    x_squared = residual * residual
    
    # 使用triton的reduce操作计算平方和
    sum_squared = tl.sum(x_squared, axis=0)
    
    # 计算均方根
    mean_squared = sum_squared / n_cols
    rms = tl.sqrt(mean_squared + eps)
    
    # 归一化
    normalized = residual / rms
    
    # 加载weight并应用 - 修复weight加载逻辑
    weight_offsets = tl.arange(0, BLOCK_SIZE) % n_cols
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=1.0)
    output = normalized * weight
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(residual_ptr + offsets, residual, mask=mask)  # 更新residual用于下一层


@triton.jit
def fused_silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU (Swish) 激活函数
    output = x * sigmoid(x)
    
    比GELU更快，常用于现代LLM的MLP层
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载输入
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x)
    # 使用数值稳定的sigmoid实现
    sigmoid_val = tl.where(
        x >= 0.0,
        1.0 / (1.0 + tl.exp(-x)),
        tl.exp(x) / (1.0 + tl.exp(x))
    )
    output = x * sigmoid_val
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def element_wise_mul_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise操作: output = a * b + c
    用于Gated MLP等场景
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # 计算
    output = a * b + c
    
    # 存储
    tl.store(output_ptr + offsets, output, mask=mask)


# ============ Python接口函数 ============

def fused_add_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Add + GELU操作
    
    Args:
        x: 输入tensor, shape: [batch, seq_len, hidden_size]
        bias: bias tensor, shape: [hidden_size]
    
    Returns:
        output: GELU(x + bias)
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert bias.is_cuda, "Bias must be on CUDA device"
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    hidden_size = x.size(-1)
    
    # 优化BLOCK_SIZE配置以提高内存带宽利用率
    # 对于大hidden_size，使用更大的block提高效率
    BLOCK_SIZE = 1024  # 固定使用1024以获得更好的性能
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_gelu_kernel[grid](
        x, bias, output,
        n_elements, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def fused_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Residual + RMSNorm操作
    
    Args:
        x: 输入tensor
        residual: 残差tensor (会被in-place更新)
        weight: RMSNorm的权重
        eps: 数值稳定性参数
    
    Returns:
        (output, updated_residual)
    """
    assert x.is_contiguous()
    assert residual.is_contiguous()
    assert x.shape == residual.shape
    
    output = torch.empty_like(x)
    n_rows = x.numel() // x.size(-1)
    n_cols = x.size(-1)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)
    
    grid = (n_rows,)
    
    fused_residual_rmsnorm_kernel[grid](
        x, residual, weight, output,
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, residual


def element_wise_mul_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor
) -> torch.Tensor:
    """
    Element-wise操作: output = a * b + c
    
    用于Gated MLP: output = gate * up + bias
    """
    assert a.shape == b.shape == c.shape
    
    output = torch.empty_like(a)
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    element_wise_mul_add_kernel[grid](
        a, b, c, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============ 性能benchmark函数 ============

def benchmark_kernel(func, *args, warmup=10, iters=100, name="Kernel"):
    """Benchmark一个kernel的性能"""
    import time
    
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    print(f"{name}: {avg_time*1000:.3f}ms per iteration")
    return avg_time


if __name__ == "__main__":
    # 测试和benchmark
    print("=" * 60)
    print("自定义Triton Kernel测试")
    print("=" * 60)
    
    # 设置
    batch_size = 16
    seq_len = 128
    hidden_size = 4096
    device = 'cuda'
    
    # 测试1: Fused Add + GELU
    print("\n[测试1] Fused Add + GELU")
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    bias = torch.randn(hidden_size, device=device)
    
    # 自定义kernel
    output_custom = fused_add_gelu(x, bias)
    
    # PyTorch参考实现
    def torch_add_gelu(x, bias):
        return torch.nn.functional.gelu(x + bias, approximate='tanh')
    
    output_torch = torch_add_gelu(x, bias)
    
    # 验证正确性
    max_diff = (output_custom - output_torch).abs().max().item()
    print(f"最大差异: {max_diff:.2e}")
    print(f"相对误差: {max_diff / output_torch.abs().max().item():.2e}")
    
    # Benchmark
    print("\nBenchmark:")
    time_custom = benchmark_kernel(fused_add_gelu, x, bias, name="Custom Fused")
    time_torch = benchmark_kernel(torch_add_gelu, x, bias, name="PyTorch")
    print(f"加速比: {time_torch/time_custom:.2f}x")
    
    # 测试2: Element-wise Mul + Add
    print("\n" + "=" * 60)
    print("[测试2] Element-wise Mul + Add")
    a = torch.randn(batch_size, seq_len, hidden_size, device=device)
    b = torch.randn(batch_size, seq_len, hidden_size, device=device)
    c = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    output_custom = element_wise_mul_add(a, b, c)
    output_torch = a * b + c
    
    max_diff = (output_custom - output_torch).abs().max().item()
    print(f"最大差异: {max_diff:.2e}")
    
    # Benchmark
    print("\nBenchmark:")
    time_custom = benchmark_kernel(element_wise_mul_add, a, b, c, name="Custom Fused")
    time_torch = benchmark_kernel(lambda a, b, c: a * b + c, a, b, c, name="PyTorch")
    print(f"加速比: {time_torch/time_custom:.2f}x")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
