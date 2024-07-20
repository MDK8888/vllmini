import os
from setuptools import setup, find_packages
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory of the current script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="paged_attention_cuda",
    version="0.1.0",
    author="Ken Ding + Claude 3.5 Sonnet",
    description="CUDA kernels for paged attention",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="paged_attention_cuda.paged_attention_cuda",
            sources=[
                os.path.join(ROOT_DIR, 'paged_attention_cuda', 'paged_attention_cuda.cpp'),
                os.path.join(ROOT_DIR, 'paged_attention_cuda', 'attention_kernels.cu'),
                os.path.join(ROOT_DIR, 'paged_attention_cuda', 'cache_kernels.cu'),  # Add this line
            ],
             include_dirs=torch.utils.cpp_extension.include_paths(),
             library_dirs=torch.utils.cpp_extension.library_paths(),
            extra_compile_args={
                'cxx': ['-O3', '-march=native', '-mtune=native'],
                'nvcc': [
                    '-O3',
                    '-gencode', 'arch=compute_70,code=sm_70',
                    '-gencode', 'arch=compute_75,code=sm_75',
                    '-gencode', 'arch=compute_80,code=sm_80',
                    '-gencode', 'arch=compute_86,code=sm_86',
                    '--use_fast_math',
                    '-lineinfo',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)