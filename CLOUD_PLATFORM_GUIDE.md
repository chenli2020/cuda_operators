# 云平台使用指南（智星云/AutoDL）

## 快速开始

### 方法 1: SSH 连接 + 命令行

```bash
# 1. 克隆或上传代码
git clone <你的仓库地址>
cd cuda_operators

# 2. 运行启动脚本
bash cloud_quickstart.sh

# 3. 启动 Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 方法 2: 直接上传文件

1. 通过平台的 Web 界面上传代码
2. 打开终端，运行：
```bash
bash cloud_quickstart.sh
```

## 平台特定说明

### AutoDL
- ✅ 预装 PyTorch + CUDA
- ✅ 提供终端和 SSH
- ✅ 支持持久化磁盘
- 💡 建议：使用 "社区镜像" 选择 PyTorch 环境

### 智星云
- ✅ 预装深度学习环境
- ✅ 提供桌面和 SSH
- 💡 建议：选择 PyTorch 镜像

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8
- GCC >= 9.0

## 故障排除

### 问题 1: 编译错误
```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 GCC 版本
gcc --version
```

### 问题 2: PyTorch CUDA 不可用
```bash
# 重新安装 PyTorch（匹配 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题 3: 权限问题
```bash
# 添加执行权限
chmod +x cloud_quickstart.sh
```

## 性能优化

### 选择合适的 GPU
- RTX 3090: 性价比高，24GB 显存
- RTX 4090: 最快性能，24GB 显存
- A100: 大模型训练，40/80GB 显存

### 监控 GPU 使用
```bash
# 实时监控
watch -n 1 nvidia-smi
```

## 文件说明

- `cloud_quickstart.sh` - Linux/Mac 启动脚本
- `cloud_quickstart.bat` - Windows 启动脚本
- `run_python_tests.sh` - 运行所有测试（Linux/Mac）
- `run_python_tests.bat` - 运行所有测试（Windows）
- `setup.py` - 编译脚本
- `CLOUD_PLATFORM_GUIDE.md` - 本文档

## 支持的平台

- ✅ AutoDL
- ✅ 智星云
- ✅ 阿里云 PAI
- ✅ 腾讯云 GPU
- ✅ 百度云 AI
