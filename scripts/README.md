# 自动化脚本使用说明

本目录包含多个自动化脚本，帮助你快速将项目部署到 Colab 环境。

## 🚀 快速开始（推荐）

### Windows 用户

```cmd
# 一键完成所有操作
cd scripts
quickstart_colab.bat
```

这个脚本会自动：
1. ✅ 初始化 Git 仓库（如果需要）
2. ✅ 添加所有文件并提交
3. ✅ 引导你创建 GitHub 仓库
4. ✅ 推送代码到 GitHub
5. ✅ 生成 Colab 打开链接

### Linux/Mac 用户

```bash
cd scripts
chmod +x sync_to_github.sh
./sync_to_github.sh
```

## 📁 可用脚本

### 1. `quickstart_colab.bat` - 一键启动（Windows）

**最简单的方式，适合第一次使用**

功能：
- 自动初始化 Git 仓库
- 自动添加和提交文件
- 引导创建 GitHub 仓库
- 推送代码并生成 Colab 链接

**使用：**
```cmd
cd scripts
quickstart_colab.bat
```

**要求：**
- Windows 系统
- Git 已安装
- GitHub 账号

---

### 2. `sync_to_github.sh` / `sync_to_github.bat` - GitHub 同步

**适合已配置好 GitHub 仓库的项目**

功能：
- 自动添加 Colab 相关文件
- 提交更改
- 推送到 GitHub

**使用：**
```cmd
# Windows
cd scripts
sync_to_github.bat

# Linux/Mac
cd scripts
chmod +x sync_to_github.sh
./sync_to_github.sh
```

---

### 3. `upload_to_gdrive.py` - Google Drive 上传

**直接上传到 Google Drive，然后从 Drive 打开**

功能：
- 通过 Google Drive API 上传 notebook
- 自动创建文件夹
- 生成 Colab 打开链接

**首次配置：**
1. 启用 Google Drive API
2. 创建 OAuth 2.0 凭证
3. 下载 `credentials.json` 到 `scripts/` 目录

**使用：**
```bash
# 安装依赖
pip install google-api-python-client google-auth-oauthlib

# 运行脚本
python scripts/upload_to_gdrive.py
```

**详细配置步骤：**

#### 步骤 1：启用 Google Drive API

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 访问 [Google Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com)
4. 点击 "启用"

#### 步骤 2：创建 OAuth 凭证

1. 访问 [Credentials 页面](https://console.cloud.google.com/apis/credentials)
2. 点击 "创建凭证" → "OAuth 客户端 ID"
3. 应用类型选择 "桌面应用"
4. 输入名称（如 "Colab Uploader"）
5. 点击 "创建"
6. 下载 JSON 文件，重命名为 `credentials.json`
7. 将文件放到 `scripts/` 目录

#### 步骤 3：运行上传脚本

```bash
python scripts/upload_to_gdrive.py
```

首次运行会打开浏览器进行认证。

---

## 🔄 完整工作流程

### 方法 1：通过 GitHub（推荐）

```cmd
# 1. 准备环境
cd scripts
quickstart_colab.bat

# 2. 脚本会引导你完成所有步骤

# 3. 在 Colab 中打开
# 访问生成的链接，或：
# https://colab.research.google.com/github/YOUR_USERNAME/cuda_operators/blob/main/colab_layernorm.ipynb
```

### 方法 2：通过 Google Drive

```bash
# 1. 配置 Google Drive API（首次）
# 按照上面的步骤创建 credentials.json

# 2. 上传文件
python scripts/upload_to_gdrive.py

# 3. 脚本会生成 Colab 打开链接
```

### 方法 3：手动更新（已配置）

```cmd
# 修改代码后，快速同步
cd scripts
sync_to_github.bat  # Windows
# 或
./sync_to_github.sh  # Linux/Mac
```

## 🛠️ 故障排除

### 问题 1：Git 命令未找到

**Windows:**
```cmd
# 下载并安装 Git
# https://git-scm.com/download/win
```

**Mac:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt-get install git
```

### 问题 2：GitHub 认证失败

```
# 使用 GitHub CLI（推荐）
# https://cli.github.com/

gh auth login

# 或使用 SSH 密钥
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### 问题 3：推送到 GitHub 失败

```cmd
# 检查远程仓库
git remote -v

# 如果配置错误，重新设置
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/cuda_operators.git

# 推送
git push -u origin main
```

### 问题 4：Python 脚本依赖缺失

```bash
# 安装所需包
pip install google-api-python-client google-auth-oauthlib
```

## 📊 脚本对比

| 脚本 | 难度 | 速度 | 适用场景 |
|------|------|------|----------|
| `quickstart_colab.bat` | ⭐ 最简单 | 快 | 首次使用 |
| `sync_to_github.sh/bat` | ⭐⭐ 简单 | 快 | 日常更新 |
| `upload_to_gdrive.py` | ⭐⭐⭐ 复杂 | 中等 | 需要隐私保护 |

## 💡 最佳实践

1. **首次使用**：运行 `quickstart_colab.bat`
2. **日常更新**：运行 `sync_to_github.bat`
3. **团队协作**：使用 GitHub 方式
4. **私人项目**：使用 Google Drive 方式

## 🔗 有用链接

- [GitHub 创建仓库](https://github.com/new)
- [Google Cloud Console](https://console.cloud.google.com/)
- [Google Colab](https://colab.research.google.com/)
- [Git 下载](https://git-scm.com/downloads)
- [GitHub CLI](https://cli.github.com/)

## 📞 获取帮助

如果遇到问题：

1. 查看项目主 README：`../README.md`
2. 查看 Colab 指南：`../COLAB_GUIDE.md`
3. 提交 Issue：[项目 Issues](https://github.com/yourusername/cuda_operators/issues)
