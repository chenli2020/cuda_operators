"""Setup script for CUDA operators."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, find_packages
import os
import sys

# Detect CUDA
CUDA_HOME = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
if not CUDA_HOME:
    raise RuntimeError("CUDA_HOME or CUDA_PATH environment variable must be set")

# CUDA architecture
CUDA_ARCH = os.environ.get('CUDA_ARCH', '80')

# Source files
sources = [
    'src/binding.cpp',
    'src/ops/reduce.cu',
    'src/ops/softmax.cu',
    'src/ops/layernorm.cu',
    'src/ops/rmsnorm.cu',
    'src/ops/matmul.cu',
]

# Include directories
include_dirs = [
    'src',
    'src/common',
    'src/ops',
    os.path.join(CUDA_HOME, 'include'),
    pybind11.get_include(),
]

# Library directories
library_dirs = [os.path.join(CUDA_HOME, 'lib64')]

# Libraries
libraries = ['cudart', 'cublas']

# Compiler flags
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        f'-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}',
        '--use_fast_math',
        '-std=c++17',
    ]
}

ext_modules = [
    Pybind11Extension(
        'cuda_ops',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language='c++',
        cuda=True,
    ),
]

setup(
    name='cuda_operators',
    version='0.1.0',
    description='CUDA operator development template',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'torch>=2.0.0',
    ],
)