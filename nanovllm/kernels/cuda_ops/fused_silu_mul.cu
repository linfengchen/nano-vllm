/*
 * CUDA实现：Fused SiLU + Element-wise Multiply
 * 
 * 用于Gated MLP优化：output = silu(gate) * up
 * 相比分离操作，减少内存访问次数
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 设备函数：SiLU激活
__device__ __forceinline__ float silu(float x) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + expf(-x));
}

// CUDA Kernel：Fused SiLU + Multiply
// output = silu(gate) * up
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // 融合操作：计算 silu(gate) 并立即与 up 相乘
        float gate_val = gate[idx];
        float up_val = up[idx];
        
        // SiLU(gate) * up
        float silu_val = silu(gate_val);
        output[idx] = silu_val * up_val;
    }
}

// CUDA Kernel：Vectorized版本（使用float4优化）
__global__ void fused_silu_mul_kernel_vec4(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int n_elements) {
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n_elements) {
        // 使用向量化加载（一次加载4个float）
        float4 gate_vec = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 up_vec = *reinterpret_cast<const float4*>(&up[idx]);
        float4 output_vec;
        
        // 处理4个元素
        output_vec.x = silu(gate_vec.x) * up_vec.x;
        output_vec.y = silu(gate_vec.y) * up_vec.y;
        output_vec.z = silu(gate_vec.z) * up_vec.z;
        output_vec.w = silu(gate_vec.w) * up_vec.w;
        
        // 向量化存储
        *reinterpret_cast<float4*>(&output[idx]) = output_vec;
    }
    
    // 处理剩余元素
    for (int i = idx; i < min(idx + 4, n_elements); i++) {
        if (i >= n_elements - (n_elements % 4)) {
            output[i] = silu(gate[i]) * up[i];
        }
    }
}

// CUDA Kernel：FP16版本
__global__ void fused_silu_mul_kernel_fp16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    int n_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        float gate_val = __half2float(gate[idx]);
        float up_val = __half2float(up[idx]);
        
        float silu_val = silu(gate_val);
        output[idx] = __float2half(silu_val * up_val);
    }
}

// C++ wrapper函数
torch::Tensor fused_silu_mul_forward(
    torch::Tensor gate,
    torch::Tensor up) {
    
    // 检查输入
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(up.is_cuda(), "up must be a CUDA tensor");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    
    auto output = torch::empty_like(gate);
    int n_elements = gate.numel();
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    // 根据数据类型选择kernel
    if (gate.dtype() == torch::kFloat32) {
        // 检查是否可以使用向量化版本（需要16字节对齐）
        bool can_vectorize = (n_elements % 4 == 0) && 
                            (reinterpret_cast<uintptr_t>(gate.data_ptr<float>()) % 16 == 0) &&
                            (reinterpret_cast<uintptr_t>(up.data_ptr<float>()) % 16 == 0);
        
        if (can_vectorize && n_elements >= 1024) {
            // 使用向量化版本
            const int vec_threads = 256;
            const int vec_blocks = (n_elements / 4 + vec_threads - 1) / vec_threads;
            
            fused_silu_mul_kernel_vec4<<<vec_blocks, vec_threads>>>(
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                output.data_ptr<float>(),
                n_elements
            );
        } else {
            // 使用标准版本
            fused_silu_mul_kernel<<<blocks, threads>>>(
                gate.data_ptr<float>(),
                up.data_ptr<float>(),
                output.data_ptr<float>(),
                n_elements
            );
        }
    } else if (gate.dtype() == torch::kFloat16) {
        fused_silu_mul_kernel_fp16<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            n_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only float32 and float16 are supported.");
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// PyBind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul_forward", &fused_silu_mul_forward, 
          "Fused SiLU + Multiply forward (CUDA)",
          py::arg("gate"),
          py::arg("up"));
}
