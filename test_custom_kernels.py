"""
测试和benchmark自定义CUDA kernel
展示如何验证正确性和测量性能
"""
import torch
import sys
import os

# 确保能导入nanovllm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanovllm.kernels.triton_ops import (
    fused_add_gelu,
    element_wise_mul_add,
    benchmark_kernel
)


def test_correctness():
    """测试kernel的正确性"""
    print("=" * 70)
    print("测试自定义Kernel的正确性")
    print("=" * 70)
    
    batch_size = 8
    seq_len = 64
    hidden_size = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("警告: CUDA不可用，跳过测试")
        return
    
    # 测试1: Fused Add + GELU
    print("\n[测试1] Fused Add + GELU")
    print("-" * 70)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
    
    # 自定义kernel
    output_custom = fused_add_gelu(x, bias)
    
    # PyTorch参考实现
    output_torch = torch.nn.functional.gelu(x + bias, approximate='tanh')
    
    # 验证
    max_diff = (output_custom - output_torch).abs().max().item()
    mean_diff = (output_custom - output_torch).abs().mean().item()
    rel_error = max_diff / (output_torch.abs().max().item() + 1e-8)
    
    print(f"输入形状: {x.shape}")
    print(f"最大绝对误差: {max_diff:.2e}")
    print(f"平均绝对误差: {mean_diff:.2e}")
    print(f"最大相对误差: {rel_error:.2e}")
    
    if max_diff < 1e-4:
        print("✓ 正确性测试通过!")
    else:
        print("✗ 警告: 误差较大，可能存在问题")
    
    # 测试2: Element-wise Mul + Add
    print("\n[测试2] Element-wise Mul + Add")
    print("-" * 70)
    a = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    b = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    c = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    output_custom = element_wise_mul_add(a, b, c)
    output_torch = a * b + c
    
    max_diff = (output_custom - output_torch).abs().max().item()
    mean_diff = (output_custom - output_torch).abs().mean().item()
    
    print(f"输入形状: {a.shape}")
    print(f"最大绝对误差: {max_diff:.2e}")
    print(f"平均绝对误差: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ 正确性测试通过!")
    else:
        print("✗ 警告: 误差较大，可能存在问题")


def benchmark_kernels():
    """Benchmark kernel性能"""
    print("\n" + "=" * 70)
    print("Benchmark自定义Kernel性能")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告: CUDA不可用，跳过benchmark")
        return
    
    configs = [
        (16, 128, 4096, "中等规模"),
        (32, 256, 4096, "大规模"),
        (8, 64, 8192, "大hidden_size"),
    ]
    
    for batch_size, seq_len, hidden_size, desc in configs:
        print(f"\n配置: {desc}")
        print(f"  batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        print("-" * 70)
        
        # Fused Add + GELU benchmark
        print("\n1. Fused Add + GELU:")
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
        
        def torch_add_gelu(x, bias):
            return torch.nn.functional.gelu(x + bias, approximate='tanh')
        
        time_custom = benchmark_kernel(fused_add_gelu, x, bias, warmup=20, iters=100, name="  Custom Triton")
        time_torch = benchmark_kernel(torch_add_gelu, x, bias, warmup=20, iters=100, name="  PyTorch")
        speedup = time_torch / time_custom
        print(f"  加速比: {speedup:.2f}x")
        
        # Element-wise Mul + Add benchmark
        print("\n2. Element-wise Mul + Add:")
        a = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        b = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        c = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        
        time_custom = benchmark_kernel(element_wise_mul_add, a, b, c, warmup=20, iters=100, name="  Custom Triton")
        time_torch = benchmark_kernel(lambda a, b, c: a * b + c, a, b, c, warmup=20, iters=100, name="  PyTorch")
        speedup = time_torch / time_custom
        print(f"  加速比: {speedup:.2f}x")


def profile_kernel():
    """使用PyTorch profiler分析kernel"""
    print("\n" + "=" * 70)
    print("使用Profiler分析Kernel")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告: CUDA不可用，跳过profiling")
        return
    
    from torch.profiler import profile, ProfilerActivity, record_function
    
    batch_size = 16
    seq_len = 128
    hidden_size = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
    
    print("\n分析 Fused Add + GELU kernel...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("custom_fused_add_gelu"):
            for _ in range(10):
                output = fused_add_gelu(x, bias)
    
    # 打印结果
    print("\nTop 10 CUDA操作:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
        header="Top 10 operations by CUDA time"
    ))
    
    # 保存trace
    trace_file = "custom_kernel_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n✓ Chrome trace已保存到: {trace_file}")
    print(f"  在chrome://tracing中打开查看详细时间线")


def memory_usage_test():
    """测试内存使用"""
    print("\n" + "=" * 70)
    print("测试内存使用")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("警告: CUDA不可用，跳过测试")
        return
    
    batch_size = 16
    seq_len = 128
    hidden_size = 4096
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    bias = torch.randn(hidden_size, device=device, dtype=torch.float32)
    
    # 测试自定义kernel
    torch.cuda.reset_peak_memory_stats()
    output = fused_add_gelu(x, bias)
    mem_custom = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # 测试PyTorch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_torch = torch.nn.functional.gelu(x + bias, approximate='tanh')
    mem_torch = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"\n输入tensor大小: {batch_size * seq_len * hidden_size * 4 / 1024**2:.2f} MB")
    print(f"Custom kernel峰值内存: {mem_custom:.2f} MB")
    print(f"PyTorch峰值内存: {mem_torch:.2f} MB")
    print(f"内存节省: {mem_torch - mem_custom:.2f} MB ({(1 - mem_custom/mem_torch)*100:.1f}%)")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("自定义CUDA Kernel测试套件")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n错误: CUDA不可用!")
        print("请在支持CUDA的环境中运行此测试")
        return
    
    print(f"\nCUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 运行测试
    test_correctness()
    benchmark_kernels()
    memory_usage_test()
    profile_kernel()
    
    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)
    print("\n下一步:")
    print("1. 查看 custom_kernel_trace.json 了解详细性能")
    print("2. 根据benchmark结果决定是否使用自定义kernel")
    print("3. 将性能好的kernel集成到模型中")
    print("=" * 70)


if __name__ == "__main__":
    main()
