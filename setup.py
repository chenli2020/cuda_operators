"""Setup script for CUDA operators with flexible build modes."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, find_packages
import os
import sys
import platform

def detect_cuda():
    """智能检测CUDA环境，支持多种构建模式"""
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))

    # 检查常见的CUDA安装路径
    if not cuda_home:
        common_paths = []
        if platform.system() == 'Linux':
            common_paths = [
                '/usr/local/cuda',
                '/opt/cuda',
                '/usr/cuda',
            ]
        elif platform.system() == 'Windows':
            common_paths = [
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0',
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
            ]

        for path in common_paths:
            if os.path.exists(path):
                cuda_home = path
                break

    # 尝试从PyTorch检测CUDA（支持Colab环境）
    if not cuda_home:
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                # 获取CUDA路径
                cuda_include = torch.cuda.include_dirs()[0] if hasattr(torch.cuda, 'include_dirs') else ''
                if cuda_include:
                    cuda_home = cuda_include.replace('include', '').rstrip(os.sep)
                    print(f"✓ Detected CUDA from PyTorch: {cuda_home}")
        except (ImportError, AttributeError, IndexError):
            pass

    return cuda_home

def get_build_config():
    """获取构建配置，支持多种模式"""
    cuda_home = detect_cuda()

    if not cuda_home:
        print("⚠️  CUDA not detected. Installing as CPU-only package.")
        print("   To build CUDA version, set CUDA_HOME or install PyTorch with CUDA.")
        return None

    print(f"✓ Using CUDA from: {cuda_home}")

    # CUDA架构配置
    cuda_arch = os.environ.get('CUDA_ARCH', '80')

    # 源文件
    sources = [
        'src/binding.cpp',
        'src/ops/reduce.cu',
        'src/ops/softmax.cu',
        'src/ops/layernorm.cu',
        'src/ops/rmsnorm.cu',
        'src/ops/matmul.cu',
    ]

    # 检查源文件是否存在
    missing_files = [f for f in sources if not os.path.exists(f)]
    if missing_files:
        print(f"⚠️  Missing source files: {missing_files}")
        print("   Installing as CPU-only package.")
        return None

    # 包含目录
    include_dirs = [
        'src',
        'src/common',
        'src/ops',
        os.path.join(cuda_home, 'include'),
        pybind11.get_include(),
    ]

    # 库目录
    library_dirs = [os.path.join(cuda_home, 'lib64')]
    if platform.system() == 'Windows':
        library_dirs.append(os.path.join(cuda_home, 'lib'))

    # 库
    libraries = ['cudart', 'cublas']

    # 编译器标志
    extra_compile_args = {
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            f'-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}',
            '--use_fast_math',
            '-std=c++17',
        ]
    }

    return {
        'sources': sources,
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'libraries': libraries,
        'extra_compile_args': extra_compile_args,
    }

def build_extension_config():
    """构建扩展配置"""
    config = get_build_config()

    if not config:
        # CPU-only版本（占位符）
        return [
            Pybind11Extension(
                'cuda_ops',
                sources=['src/binding.cpp'],  # 仅包含绑定文件
                include_dirs=['src', 'src/common', pybind11.get_include()],
                extra_compile_args={'cxx': ['-O3', '-std=c++17']},
                language='c++',
                cuda=False,
            )
        ]

    return [
        Pybind11Extension(
            'cuda_ops',
            sources=config['sources'],
            include_dirs=config['include_dirs'],
            library_dirs=config['library_dirs'],
            libraries=config['libraries'],
            extra_compile_args=config['extra_compile_args'],
            language='c++',
            cuda=True,
        ),
    ]

# 构建扩展模块
ext_modules = build_extension_config()

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