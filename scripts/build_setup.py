#!/usr/bin/env python3
"""
简化的构建脚本 - 支持多种环境

使用方法:
1. 本地CUDA环境: python scripts/build_setup.py
2. Colab环境: 直接使用 torch.utils.cpp_extension.load()
3. CPU环境: python scripts/build_setup.py --cpu-only
"""
import os
import sys
import argparse
import platform
import subprocess
from pathlib import Path


def check_environment():
    """检查当前环境"""
    print("🔍 检查构建环境...")

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8+")
        return False

    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # 检查CUDA环境
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
    if cuda_home and os.path.exists(cuda_home):
        print(f"✓ CUDA环境: {cuda_home}")
        return 'cuda'
    else:
        print("⚠️  未检测到CUDA环境")
        return 'cpu'


def build_with_colab_method():
    """使用PyTorch JIT编译方法（推荐用于Colab）"""
    print("\n🚀 使用PyTorch JIT编译方法")

    try:
        import torch
        from torch.utils.cpp_extension import load

        if not torch.cuda.is_available():
            print("❌ PyTorch CUDA版本未安装")
            return False

        print(f"✓ PyTorch {torch.__version__} (CUDA available)")

        # 设置CUDA架构
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU: {device_name}")

        # 自动检测GPU架构
        if 'A100' in device_name:
            arch = '80'
        elif 'V100' in device_name:
            arch = '70'
        elif 'T4' in device_name or 'Tesla T4' in device_name:
            arch = '75'
        else:
            arch = '70'  # 默认

        os.environ['TORCH_CUDA_ARCH_LIST'] = arch

        print(f"✓ 使用CUDA架构: sm_{arch}")

        # 编译扩展
        cuda_ops = load(
            name='cuda_ops',
            sources=[
                'src/binding.cpp',
                'src/ops/reduce.cu',
                'src/ops/softmax.cu',
                'src/ops/layernorm.cu',
                'src/ops/rmsnorm.cu',
                'src/ops/matmul.cu',
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=True
        )

        print("✓ 编译成功!")
        return cuda_ops

    except ImportError:
        print("❌ 需要安装PyTorch: pip install torch")
        return False
    except Exception as e:
        print(f"❌ 编译失败: {e}")
        return False


def build_with_setup_method(cuda_arch='80'):
    """使用setup.py构建方法"""
    print(f"\n🔨 使用setup.py构建 (CUDA架构: sm_{cuda_arch})")

    try:
        # 设置环境变量
        os.environ['CUDA_ARCH'] = cuda_arch

        # 调用setup.py
        cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
        print(f"执行: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode == 0:
            print("✓ 构建成功!")
            return True
        else:
            print(f"❌ 构建失败: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 构建错误: {e}")
        return False


def build_with_cmake_method():
    """使用CMake构建方法"""
    print("\n🏗️  使用CMake构建")

    try:
        # 检查CMake是否可用
        result = subprocess.run(['cmake', '--version'], capture_output=True)
        if result.returncode != 0:
            print("❌ CMake未安装")
            return False

        # 创建构建目录
        build_dir = Path('build')
        build_dir.mkdir(exist_ok=True)

        # 运行CMake配置
        cmake_cmd = [
            'cmake', '..',
            f'-DCMAKE_CUDA_ARCHITECTURES={os.environ.get("CUDA_ARCH", "80")}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]

        print(f"配置: {' '.join(cmake_cmd)}")
        result = subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        print("✓ CMake配置成功")

        # 编译
        print("编译...")
        build_cmd = ['cmake', '--build', '.', '--config', 'Release']
        result = subprocess.run(build_cmd, cwd=build_dir, check=True)
        print("✓ 编译成功")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ CMake构建失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 构建错误: {e}")
        return False


def install_test_dependencies():
    """安装测试依赖"""
    print("\n📦 安装测试依赖...")

    dependencies = [
        'torch>=2.0.0',
        'numpy',
        'pytest',
        'pybind11',
    ]

    for dep in dependencies:
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', dep],
                check=True
            )
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  {dep} 安装失败")


def run_tests():
    """运行测试"""
    print("\n🧪 运行测试...")

    test_files = [
        'tests/test_reduce.py',
        'tests/test_softmax.py',
        'tests/test_layernorm.py',
        'tests/test_rmsnorm.py',
        'tests/test_matmul.py',
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n运行 {test_file}...")
            try:
                subprocess.run([sys.executable, test_file], check=True)
                print(f"✓ {test_file} 通过")
            except subprocess.CalledProcessError:
                print(f"❌ {test_file} 失败")


def main():
    parser = argparse.ArgumentParser(description='CUDA算子构建脚本')
    parser.add_argument('--method', choices=['auto', 'colab', 'setup', 'cmake'],
                       default='auto', help='构建方法')
    parser.add_argument('--arch', default='80', help='CUDA架构 (例如: 75, 80, 86)')
    parser.add_argument('--cpu-only', action='store_true', help='仅CPU版本')
    parser.add_argument('--skip-tests', action='store_true', help='跳过测试')

    args = parser.parse_args()

    print("=" * 60)
    print("CUDA算子构建工具")
    print("=" * 60)

    # 检查环境
    env_type = check_environment()

    # CPU-only模式
    if args.cpu_only or env_type == 'cpu':
        print("\n⚠️  CPU-only模式 - CUDA功能将被禁用")
        # 这里可以实现CPU-only版本的构建
        return

    # 安装依赖
    install_test_dependencies()

    # 选择构建方法
    method = args.method
    if method == 'auto':
        # 自动检测最佳构建方法
        try:
            import torch
            if torch.cuda.is_available():
                method = 'colab'
            else:
                method = 'setup'
        except ImportError:
            method = 'setup'

    # 执行构建
    success = False
    if method == 'colab':
        success = build_with_colab_method()
    elif method == 'setup':
        success = build_with_setup_method(args.arch)
    elif method == 'cmake':
        success = build_with_cmake_method()

    if success:
        print("\n" + "=" * 60)
        print("✓ 构建完成!")
        print("=" * 60)

        # 运行测试
        if not args.skip_tests:
            run_tests()
    else:
        print("\n" + "=" * 60)
        print("❌ 构建失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()