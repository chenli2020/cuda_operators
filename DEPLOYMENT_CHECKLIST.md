# 云平台部署检查清单

## 部署前检查

- [ ] 已注册 AutoDL 或 智星云 账号
- [ ] 已完成实名认证
- [ ] 账户已充值（建议 ¥50+）
- [ ] 本地代码已准备完成

## 平台选择检查

- [ ] 选择合适的 GPU（推荐 RTX 3090）
- [ ] 选择 PyTorch 镜像（推荐 2.0+ 版本）
- [ ] 确认 CUDA 版本（>= 11.8）
- [ ] 确认 Python 版本（>= 3.8）

## 代码部署检查

### 方法 1: Git Clone
- [ ] 代码已推送到 GitHub/Gitee
- [ ] 平台 Terminal 中执行 `git clone`
- [ ] 进入项目目录 `cd cuda_operators`

### 方法 2: 直接上传
- [ ] 本地代码打包为 zip
- [ ] 平台 Web 界面上传 zip 文件
- [ ] Terminal 中解压 `unzip project.zip`

## 环境配置检查

- [ ] 运行 `bash cloud_quickstart.sh`
- [ ] 检查 NVIDIA 驱动：`nvidia-smi`
- [ ] 检查 Python 版本：`python --version`
- [ ] 检查 PyTorch：`python -c "import torch; print(torch.__version__)"`
- [ ] 检查 CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 安装依赖完成：`pip install -r requirements.txt`

## 编译检查

- [ ] 编译 CUDA 扩展：`python setup.py build_ext --inplace`
- [ ] 编译无错误
- [ ] 生成了 `.pyd` (Windows) 或 `.so` (Linux) 文件

## 测试检查

- [ ] 运行单元测试：`python -m pytest tests/ -v`
- [ ] 所有测试通过
- [ ] 性能基准测试正常

## 使用检查

- [ ] 选择一个测试开始：`python tests/test_layernorm.py`
- [ ] 查看测试输出和性能对比
- [ ] 结果正确
- [ ] GPU 使用率正常

## 常见问题排查

### 编译失败
- [ ] 检查 CUDA 版本匹配
- [ ] 检查 GCC 版本（>= 9.0）
- [ ] 检查 CMake 版本（>= 3.18）

### 导入错误
- [ ] 确认编译生成的文件在正确位置
- [ ] 检查 PYTHONPATH 环境变量
- [ ] 重新编译扩展

### CUDA 不可用
- [ ] 检查 PyTorch CUDA 版本匹配
- [ ] 重新安装 PyTorch
- [ ] 检查 GPU 驱动

### 性能问题
- [ ] 检查 GPU 使用率：`nvidia-smi`
- [ ] 检查是否使用正确实现（如 warp, vectorized）
- [ ] 调整 batch size 和数据大小

## 关机前检查

- [ ] 保存所有代码修改
- [ ] 下载重要代码到本地
- [ ] 提交代码到 Git 仓库
- [ ] 停止 GPU 租用（省钱！）

## 下次使用快速恢复

- [ ] 重新租用 GPU
- [ ] Git clone 最新代码
- [ ] 运行 `bash cloud_quickstart.sh`
- [ ] 开始测试：`python tests/test_layernorm.py`

---

**💡 提示**：每次使用前建议运行 `bash cloud_quickstart.sh` 确保环境正确配置。

**💰 省钱提醒**：用完记得关机！每小时都要计费。
