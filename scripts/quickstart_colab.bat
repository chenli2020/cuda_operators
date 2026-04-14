@echo off
REM 一键准备 Colab 环境 - 自动创建 GitHub 仓库并推送

echo ==========================================
echo   CUDA Operators - Colab 快速启动
echo ==========================================
echo.

REM 检查是否是 git 仓库
if not exist .git (
    echo 🔧 初始化 Git 仓库...
    git init
    git branch -M main
)

REM 添加所有文件
echo 📝 添加文件到 Git...
git add . >nul 2>&1

REM 提交
echo 💾 提交更改...
git commit -m "Add CUDA operators with Colab support" >nul 2>&1

REM 检查远程仓库
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  未配置 GitHub 远程仓库
    echo.
    echo ==========================================
    echo   自动创建 GitHub 仓库
    echo ==========================================
    echo.
    echo 请选择操作：
    echo.
    echo [1] 手动创建仓库（推荐）
    echo [2] 自动创建（需要 GitHub CLI）
    echo [3] 取消
    echo.
    choice /c 123 /n /m "请选择 (1-3): "

    if errorlevel 3 (
        echo 已取消
        pause
        exit /b 0
    )

    if errorlevel 2 (
        echo.
        echo 🔧 使用 GitHub CLI 创建仓库...
        where gh >nul 2>&1
        if errorlevel 1 (
            echo ❌ 未安装 GitHub CLI
            echo 请安装：https://cli.github.com/
            echo 或选择选项 [1] 手动创建
            pause
            exit /b 1
        )

        gh repo create cuda_operators --public --source=. --remote=origin --push
        if errorlevel 1 (
            echo ❌ 创建仓库失败
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo 请按以下步骤手动创建 GitHub 仓库：
        echo.
        echo 1. 访问：https://github.com/new
        echo 2. 仓库名称：cuda_operators
        echo 3. 选择 Public
        echo 4. 不要初始化 README（.gitignore 或 license）
        echo 5. 点击 "Create repository"
        echo.
        echo 创建后，运行以下命令：
        echo.
        echo     git remote add origin https://github.com/YOUR_USERNAME/cuda_operators.git
        echo     git push -u origin main
        echo.
        echo 替换 YOUR_USERNAME 为你的 GitHub 用户名
        echo.
        pause
        exit /b 0
    )
)

REM 推送到 GitHub
echo.
echo 🚀 推送到 GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ 推送失败
    echo 可能需要先运行：git push -u origin main
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   ✅ 准备完成！
echo ==========================================
echo.

REM 获取仓库 URL
for /f "tokens=*" %%i in ('git remote get-url origin') do set REPO_URL=%%i

echo 📚 在 Colab 中打开：
echo.
echo 1. 访问：https://colab.research.google.com/
echo 2. 点击：文件 → 在 GitHub 中打开笔记本
echo 3. 输入仓库地址：%REPO_URL%
echo 4. 选择：colab_layernorm.ipynb
echo.
echo 或者直接访问：
echo.
echo https://colab.research.google.com/github/%REPO_URL:*/blob/main/colab_layernorm.ipynb
echo.
echo 💡 提示：
echo    - 在 Colab 中启用 GPU（运行时 → 更改运行时类型 → T4 GPU）
echo    - 从上到下依次运行所有单元格
echo.
echo ==========================================
echo.

pause
