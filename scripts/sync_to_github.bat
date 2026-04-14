@echo off
REM 自动同步到 GitHub，方便在 Colab 中使用
REM Windows 版本

echo ==========================================
echo   CUDA Operators - GitHub 同步工具
echo ==========================================
echo.

REM 检查是否是 git 仓库
if not exist .git (
    echo ❌ 错误：当前目录不是 git 仓库
    echo    请先运行: git init
    pause
    exit /b 1
)

REM 检查是否有未提交的更改
for /f %%i in ('git status --porcelain') do set HAS_CHANGES=%%i
defined HAS_CHANGES (
    echo 📝 检测到未提交的更改...
    echo    添加 Colab notebooks...
    git add colab_*.ipynb
    git add COLAB_GUIDE.md
    git add colab\
    git add scripts\colab_test_runner.py
    git add README.md

    echo    提交更改...
    git commit -m "Update Colab notebooks and tools"
) else (
    echo ✓ 没有新的更改需要提交
)

REM 检查是否配置了远程仓库
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  未配置 GitHub 远程仓库
    echo.
    echo 请按以下步骤操作：
    echo 1. 在 GitHub 上创建新仓库：https://github.com/new
    echo 2. 运行以下命令（替换 YOUR_USERNAME）：
    echo    git remote add origin https://github.com/YOUR_USERNAME/cuda_operators.git
    echo    git branch -M main
    echo    git push -u origin main
    echo.
    pause
    exit /b 1
)

REM 推送到 GitHub
echo.
echo 🚀 推送到 GitHub...
git push origin main

if errorlevel 1 (
    echo.
    echo ❌ 推送失败，可能需要认证
    echo 请尝试：git push origin main
    echo.
) else (
    echo.
    echo ✓ 同步完成！
    echo.
    echo ==========================================
    echo   在 Colab 中打开：
    echo ==========================================
    echo.
    echo 1. 访问：https://colab.research.google.com/
    echo 2. 点击：文件 → 在 GitHub 中打开笔记本
    echo 3. 粘贴仓库 URL
    echo 4. 选择：colab_layernorm.ipynb
    echo.
)

pause
