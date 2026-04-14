#!/bin/bash
# 云平台（智星云/AutoDL）快速启动脚本

set -e

echo "=========================================="
echo "  CUDA Operators - 云平台快速启动"
echo "=========================================="
echo ""

# 1. 检查环境
echo "🔍 检查环境..."
nvidia-smi

echo ""
echo "📦 Python 版本:"
python --version

echo ""
echo "🔧 PyTorch 版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# 2. 安装依赖
echo ""
echo "📥 安装依赖包..."
pip install -q pybind11 pytest pytest-benchmark matplotlib

# 3. 编译 CUDA 扩展
echo ""
echo "🔨 编译 CUDA 扩展..."
python setup.py build_ext --inplace

# 4. 运行测试
echo ""
echo "✅ 运行测试..."
python -m pytest tests/ -v

echo ""
echo "=========================================="
echo "  🎉 环境准备完成！"
echo "=========================================="
echo ""
echo "📚 下一步："
echo "   1. 运行单个测试: python tests/test_layernorm.py"
echo "   2. 运行所有测试: python -m pytest tests/ -v"
echo "   3. 运行性能测试: python benchmark/benchmark.py"
echo ""
