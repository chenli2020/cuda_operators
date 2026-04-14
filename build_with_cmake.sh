#!/bin/bash
# CMake 构建脚本 - 适用于智星云/AutoDL 等平台

set -e

echo "=========================================="
echo "  CUDA Operators - CMake 构建脚本"
echo "=========================================="
echo ""

# 1. 检查环境
echo "🔍 检查环境..."

# 检查 CUDA
if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda
fi
if [ ! -d "$CUDA_HOME" ]; then
    echo "❌ CUDA 未找到: $CUDA_HOME"
    echo "请先安装 CUDA Toolkit"
    exit 1
fi
echo "✓ CUDA: $CUDA_HOME"

# 检查 Python
PYTHON_BIN=$(which python3 || which python)
echo "✓ Python: $($PYTHON_BIN --version)"

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake 未安装"
    echo "安装: sudo apt-get install -y cmake"
    exit 1
fi
echo "✓ CMake: $(cmake --version | head -1)"

# 检查 NVCC
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC 未找到"
    exit 1
fi
echo "✓ NVCC: $(nvcc --version | grep release)"

# 2. 安装依赖
echo ""
echo "📥 安装构建依赖..."
$PYTHON_BIN -m pip install -q pybind11 2>/dev/null || true

# 验证 pybind11 安装
echo "验证 pybind11..."
if $PYTHON_BIN -c "import pybind11; print('✓ pybind11 version:', pybind11.__version__)" 2>/dev/null; then
    echo "✓ pybind11 已安装"
else
    echo "❌ pybind11 安装失败"
    echo "手动安装: pip install pybind11"
    exit 1
fi

# 3. 创建构建目录
echo ""
echo "🔨 配置构建..."
BUILD_DIR="build_cmake"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 4. CMake 配置
echo "运行 CMake 配置..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
    -DCMAKE_PREFIX_PATH="$(python3 -m site --user-base)" \
    -DPython_EXECUTABLE=$PYTHON_BIN

# 5. 编译
echo ""
echo "🔧 开始编译..."
make -j$(nproc)

# 6. 检查编译结果
echo ""
echo "📦 检查编译结果..."
if [ -f "cuda_ops*.so" ] || ls *.so 1> /dev/null 2>&1; then
    echo "✅ 编译成功！"
    echo ""
    echo "生成的文件:"
    ls -lh *.so

    # 复制到项目根目录
    echo ""
    echo "📋 复制模块到项目根目录..."
    cp *.so ../
    echo "✓ 模块已复制到项目根目录"

    cd ..

    # 7. 测试导入
    echo ""
    echo "🧪 测试模块导入..."
    if $PYTHON_BIN -c "import cuda_ops; print('✓ cuda_ops 模块导入成功！')" 2>/dev/null; then
        echo "✅ 构建完全成功！"
    else
        echo "⚠️  模块已编译，但导入测试失败"
        echo "可能需要设置 LD_LIBRARY_PATH"
    fi
else
    echo "❌ 编译失败，未找到 .so 文件"
    exit 1
fi

echo ""
echo "=========================================="
echo "  🎉 CMake 构建完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 运行测试: python tests/test_layernorm.py"
echo "  2. 运行基准测试: python benchmark/benchmark.py"
echo ""
