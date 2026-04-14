# Quick Start Guide

## 1. 环境准备

### Windows
```bash
# 确保安装了 Visual Studio 2022 和 CUDA Toolkit
cuda --version  # 应该 >= 11.0

# 安装 Python 依赖
pip install -r requirements.txt
```

## 2. 构建项目

### 方法 1: 使用批处理脚本（推荐）
```bash
cd C:\study\cuda_operators

# 默认构建（Ampere架构 sm_80）
build.bat

# 指定架构（例如 RTX 4090 sm_89）
build.bat --arch 89

# 清理后构建
build.bat --clean

# Debug 模式
build.bat --debug
```

### 方法 2: 使用 CMake 手动构建
```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --config Release --parallel
```

### 方法 3: 使用 pip 安装
```bash
# 设置 CUDA 架构
set CUDA_ARCH=80
pip install -e .
```

## 3. 运行测试

```bash
# 运行所有测试
run_tests.bat

# 或单独运行
python tests/test_reduce.py
python tests/test_softmax.py
python tests/test_layernorm.py
python tests/test_rmsnorm.py
python tests/test_matmul.py
```

## 4. 使用示例

```python
import numpy as np
import cuda_ops

# Softmax
x = np.random.randn(32, 1024).astype(np.float32)
out = cuda_ops.softmax(x, rows=32, cols=1024)

# LayerNorm
weight = np.ones(1024, dtype=np.float32)
bias = np.zeros(1024, dtype=np.float32)
out = cuda_ops.layernorm(x, weight, bias, rows=32, cols=1024)

# RMSNorm (LLaMA风格)
out = cuda_ops.rmsnorm(x, weight, rows=32, cols=1024)

# MatMul
A = np.random.randn(1024, 512).astype(np.float32)
B = np.random.randn(512, 1024).astype(np.float32)
C = cuda_ops.matmul(A, B, M=1024, N=1024, K=512)

# Reduce
arr = np.random.randn(1000000).astype(np.float32)
sum_val = cuda_ops.reduce_sum(arr)[0]
```

## 5. 性能分析

### 使用 Nsight Compute
```bash
# 查看所有指标
ncu --metrics all python tests/test_matmul.py

# 关键指标
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__bytes_read.sum.per_second,\
    l1tex__t_bytes.sum.per_second \
    python tests/test_matmul.py
```

### 使用 Benchmark 脚本
```bash
python benchmark/benchmark.py
```

## 6. 调试技巧

```bash
# 内存检查
compute-sanitizer python tests/test_reduce.py

# 同步模式（更容易定位错误）
set CUDA_LAUNCH_BLOCKING=1
python tests/test_softmax.py

# CUDA 详细日志
set CUDA_VERBOSE_PTXAS=1
```

## 7. 文件结构速览

```
cuda_operators/
├── src/ops/           # 算子实现（.cu 文件）
│   ├── reduce.cu      # 归约求和（naive → warp → two-pass）
│   ├── softmax.cu     # Softmax（naive → online → warp）
│   ├── layernorm.cu   # LayerNorm（naive → warp → vectorized）
│   ├── rmsnorm.cu     # RMSNorm（LLaMA风格）
│   └── matmul.cu      # 矩阵乘法（naive → shared → 2D tiling）
├── tests/             # 测试文件
├── benchmark/         # 性能测试
└── build/             # 构建输出
```

## 8. 添加新算子

1. 在 `src/ops/` 创建 `myop.h` 和 `myop.cu`
2. 在 `src/binding.cpp` 添加 pybind11 绑定
3. 在 `tests/` 创建 `test_myop.py`
4. 重新运行 `build.bat`

## 常见问题

**Q: 找不到 cuda_ops 模块？**
A: 确保构建成功，并将 `build\Release` 添加到 PYTHONPATH，或运行 `build.bat` 自动复制。

**Q: CUDA 架构不匹配？**
A: 修改 `build.bat` 中的 `--arch` 参数（80=A100, 86=RTX 3090, 89=RTX 4090, 90=H100）。

**Q: 精度校验失败？**
A: 这是正常的，浮点运算顺序不同会导致微小差异。检查相对误差是否 < 1e-3。
