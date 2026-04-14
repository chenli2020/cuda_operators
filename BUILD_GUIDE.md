# 🔧 CUDA算子构建指南

## 快速解决setup.py问题

### 🎯 问题诊断

如果你遇到这个错误：
```python
RuntimeError: CUDA_HOME or CUDA_PATH environment variable must be set
```

这里有3种解决方案：

---

## ✅ 方案1：Colab环境（推荐用于学习）

**无需修改setup.py，直接使用PyTorch JIT编译**

```python
# 在Colab中运行这个cell即可
from torch.utils.cpp_extension import load

cuda_ops = load(
    name='cuda_ops',
    sources=[
        'src/binding.cpp',
        'src/ops/layernorm.cu',  # 只包含你需要的算子
        'src/ops/softmax.cu',
        'src/ops/reduce.cu',
    ],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

# 立即可用
result = cuda_ops.layernorm(input, weight, bias)
```

**优点：**
- ✅ 无需CUDA_HOME
- ✅ 自动检测GPU架构
- ✅ 修改代码后重新编译只需1分钟

---

## ✅ 方案2：本地环境（设置CUDA_HOME）

### Windows:
```powershell
# 在PowerShell中设置
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"

# 或者永久设置（系统环境变量）
# 设置 → 系统 → 高级系统设置 → 环境变量 → 新建
# 变量名: CUDA_HOME
# 变量值: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
```

### Linux:
```bash
# 添加到 ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新加载
source ~/.bashrc
```

### 验证设置:
```bash
echo $CUDA_HOME
# 应该显示: /usr/local/cuda 或你设置的路径
```

---

## ✅ 方案3：使用简化构建脚本

我已经创建了智能构建脚本，自动检测环境：

```bash
# 自动选择最佳构建方法
python scripts/build_setup.py

# 或者指定方法
python scripts/build_setup.py --method colab  # Colab/JIT方法
python scripts/build_setup.py --method setup  # setup.py方法
python scripts/build_setup.py --method cmake  # CMake方法

# 指定GPU架构
python scripts/build_setup.py --arch 75  # T4 GPU
python scripts/build_setup.py --arch 80  # A100 GPU
python scripts/build_setup.py --arch 86  # RTX 3090
```

---

## 🚀 推荐的开发流程

### 学习阶段（Colab）：
1. 上传代码到Colab
2. 使用PyTorch JIT编译（方案1）
3. 快速迭代开发
4. 使用PyTorch profiler分析性能

### 生产阶段（本地）：
1. 安装CUDA Toolkit
2. 设置CUDA_HOME（方案2）
3. 使用CMake构建（最佳性能）
4. 使用Nsight Compute深度分析

---

## 🛠️ 常见问题解决

### Q1: 找不到CUDA Toolkit
```bash
# 检查CUDA是否安装
nvidia-smi
nvcc --version

# Windows下载: https://developer.nvidia.com/cuda-downloads
# Linux安装: sudo apt install nvidia-cuda-toolkit
```

### Q2: Python import错误
```bash
# 确保安装了所有依赖
pip install torch numpy pybind11 pytest

# 如果使用PyTorch JIT，确保PyTorch是GPU版本
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True
```

### Q3: 编译错误
```bash
# 清理构建缓存
python setup.py clean --all
rm -rf build/
rm -rf *.egg-info/

# 重新构建
python scripts/build_setup.py
```

---

## 📊 环境对比

| 特性 | Colab JIT | 本地setup.py | 本地CMake |
|------|-----------|--------------|-----------|
| 安装难度 | 🟢 无需安装 | 🟡 需要CUDA | 🟡 需要CUDA+CMake |
| 编译速度 | 🟡 1-2分钟 | 🟢 30秒 | 🟢 30秒 |
| 性能 | 🟢 优秀 | 🟢 优秀 | 🟢 最优 |
| 调试便利性 | 🟢 浏览器 | 🟡 本地 | 🟡 本地 |
| 适用场景 | 学习、实验 | 开发、测试 | 生产、部署 |

---

## 🎯 快速开始推荐

### 如果你是初学者：
```python
# 直接在Colab中运行
!git clone https://github.com/your-repo/cuda_operators.git
%cd cuda_operators

# 使用提供的notebook
%run colab_layernorm.ipynb
```

### 如果你有本地GPU：
```bash
# 1. 设置CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# 2. 使用智能构建脚本
python scripts/build_setup.py

# 3. 运行测试
python tests/test_layernorm.py
```

---

## 📞 获取帮助

如果遇到问题：
1. 检查CUDA是否正确安装: `nvidia-smi`
2. 检查PyTorch CUDA支持: `python -c "import torch; print(torch.cuda.is_available())"`
3. 查看项目文档: README.md, QUICKSTART.md
4. 提交Issue: 包含错误信息和系统信息

---

**提示**: setup.py已经修复为智能检测环境，应该可以正常工作了！