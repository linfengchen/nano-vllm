"""
CUDA算子测试脚本

测试内容：
1. 正确性验证
2. 性能Benchmark
3. 不同数据类型测试
4. 内存使用分析
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# 添加路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from nanovllm.kernels.cuda_ops import fused_silu_mul, CUDA_OPS_AVAILABLE


def test_correctness():
    """测试正确性"""
    print("=" * 70)
    print("测试1: 正确性验证")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    # 测试不同尺寸
    test_shapes = [
        (16, 128, 4096),      # 小batch
        (32, 512, 11008),     # 中等batch
        (4, 2048, 4096),      # 长序列
    ]
    
    for shape in test_shapes:
        print(f"\n测试shape: {shape}")
        
        # 生成测试数据
        gate = torch.randn(shape, device='cuda', dtype=torch.float32)
        up = torch.randn(shape, device='cuda', dtype=torch.float32)
        
        # CUDA kernel结果
        output_cuda = fused_silu_mul(gate, up)
        
        # PyTorch参考实现
        output_torch = F.silu(gate) * up
        
        # 比较结果
        max_diff = (output_cuda - output_torch).abs().max().item()
        mean_diff = (output_cuda - output_torch).abs().mean().item()
        
        print(f"  最大差异: {max_diff:.2e}")
        print(f"  平均差异: {mean_diff:.2e}")
        
        # 检查是否在可接受范围内
        if max_diff < 1e-5:
            print(f"  ✅ 通过")
        else:
            print(f"  ❌ 失败 (差异过大)")
    
    print("\n" + "=" * 70)


def test_fp16():
    """测试FP16支持"""
    print("\n测试2: FP16数据类型")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    shape = (16, 128, 4096)
    gate = torch.randn(shape, device='cuda', dtype=torch.float16)
    up = torch.randn(shape, device='cuda', dtype=torch.float16)
    
    try:
        output_cuda = fused_silu_mul(gate, up)
        output_torch = F.silu(gate) * up
        
        max_diff = (output_cuda.float() - output_torch.float()).abs().max().item()
        print(f"FP16最大差异: {max_diff:.2e}")
        
        if max_diff < 1e-3:  # FP16精度较低
            print("✅ FP16测试通过")
        else:
            print(f"❌ FP16测试失败")
    except Exception as e:
        print(f"❌ FP16测试出错: {e}")
    
    print("=" * 70)


def benchmark():
    """性能Benchmark"""
    print("\n测试3: 性能Benchmark")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    if not CUDA_OPS_AVAILABLE:
        print("⚠️  CUDA算子未编译，仅测试PyTorch实现")
    
    # 测试配置
    shape = (16, 128, 4096)
    warmup = 20
    iters = 100
    
    gate = torch.randn(shape, device='cuda', dtype=torch.float32)
    up = torch.randn(shape, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        if CUDA_OPS_AVAILABLE:
            _ = fused_silu_mul(gate, up)
        _ = F.silu(gate) * up
    
    # Benchmark CUDA kernel
    if CUDA_OPS_AVAILABLE:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            output = fused_silu_mul(gate, up)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / iters
        print(f"CUDA Kernel: {time_cuda * 1000:.3f} ms")
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        output = F.silu(gate) * up
    torch.cuda.synchronize()
    time_torch = (time.time() - start) / iters
    print(f"PyTorch:     {time_torch * 1000:.3f} ms")
    
    if CUDA_OPS_AVAILABLE:
        speedup = time_torch / time_cuda
        print(f"\n加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"✅ CUDA kernel更快 ({speedup:.2f}x)")
        else:
            print(f"⚠️  CUDA kernel较慢 ({1/speedup:.2f}x slower)")
    
    print("=" * 70)


def benchmark_different_sizes():
    """不同尺寸的性能测试"""
    print("\n测试4: 不同尺寸性能分析")
    print("=" * 70)
    
    if not torch.cuda.is_available() or not CUDA_OPS_AVAILABLE:
        print("❌ 跳过测试")
        return
    
    test_configs = [
        ("小batch, 短序列", (4, 64, 4096)),
        ("中batch, 中序列", (16, 128, 4096)),
        ("大batch, 长序列", (32, 512, 4096)),
        ("超大hidden", (16, 128, 11008)),
    ]
    
    print(f"\n{'配置':<20} {'CUDA(ms)':<12} {'PyTorch(ms)':<12} {'加速比':<10}")
    print("-" * 70)
    
    for name, shape in test_configs:
        gate = torch.randn(shape, device='cuda')
        up = torch.randn(shape, device='cuda')
        
        warmup = 10
        iters = 50
        
        # Warmup
        for _ in range(warmup):
            _ = fused_silu_mul(gate, up)
            _ = F.silu(gate) * up
        
        # CUDA
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = fused_silu_mul(gate, up)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / iters * 1000
        
        # PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = F.silu(gate) * up
        torch.cuda.synchronize()
        time_torch = (time.time() - start) / iters * 1000
        
        speedup = time_torch / time_cuda
        print(f"{name:<20} {time_cuda:>10.3f}  {time_torch:>10.3f}  {speedup:>8.2f}x")
    
    print("=" * 70)


def test_memory():
    """内存使用测试"""
    print("\n测试5: 内存使用分析")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    shape = (16, 128, 4096)
    
    # 测试CUDA kernel
    if CUDA_OPS_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        gate = torch.randn(shape, device='cuda')
        up = torch.randn(shape, device='cuda')
        output = fused_silu_mul(gate, up)
        mem_cuda = torch.cuda.max_memory_allocated() / 1024**2
        print(f"CUDA Kernel峰值内存: {mem_cuda:.2f} MB")
        del gate, up, output
    
    # 测试PyTorch
    torch.cuda.reset_peak_memory_stats()
    gate = torch.randn(shape, device='cuda')
    up = torch.randn(shape, device='cuda')
    output = F.silu(gate) * up
    mem_torch = torch.cuda.max_memory_allocated() / 1024**2
    print(f"PyTorch峰值内存:     {mem_torch:.2f} MB")
    
    if CUDA_OPS_AVAILABLE:
        print(f"\n内存节省: {mem_torch - mem_cuda:.2f} MB")
    
    print("=" * 70)


def main():
    print("\n" + "🔥" * 35)
    print("CUDA算子完整测试套件")
    print("🔥" * 35)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA不可用，无法进行测试")
        return
    
    print(f"\nCUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA算子状态: {'✅ 已加载' if CUDA_OPS_AVAILABLE else '❌ 未加载'}")
    
    # 运行所有测试
    test_correctness()
    test_fp16()
    benchmark()
    benchmark_different_sizes()
    test_memory()
    
    print("\n" + "🎉" * 35)
    print("所有测试完成!")
    print("🎉" * 35 + "\n")


if __name__ == "__main__":
    main()
