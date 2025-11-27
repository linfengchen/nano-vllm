"""
CUDA算子编译脚本
用于编译自定义CUDA kernel

使用方法：
    python setup.py install
    
或使用develop模式（方便开发调试）：
    python setup.py develop
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='cuda_ops',
            sources=[
                os.path.join(current_dir, 'fused_silu_mul.cu'),
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++14',
                ],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++14',
                    # 为不同GPU架构编译
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 2080
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode=arch=compute_90,code=sm_90',  # H100
                    # 编译优化选项
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ]
            },
            include_dirs=[
                current_dir,
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    python_requires='>=3.7',
)
