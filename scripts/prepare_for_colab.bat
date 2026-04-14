@echo off
REM 一键打包 Colab notebooks
REM 将所有 Colab 相关文件打包成一个 zip 文件
REM 然后直接在 Colab 中上传解压

echo ==========================================
echo   📦 打包 Colab Notebooks
echo ==========================================
echo.

cd /d "%~dp0.."

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python
    echo 请先安装 Python：https://www.python.org/
    pause
    exit /b 1
)

REM 运行打包脚本
echo 📦 正在打包...
echo.
python scripts\prepare_for_colab.py

if errorlevel 1 (
    echo.
    echo ❌ 打包失败
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   ✅ 打包完成！
echo ==========================================
echo.
echo 📁 文件位置：当前目录
echo 📦 文件名：colab_notebooks.zip
echo.
echo ==========================================
echo   📤 上传到 Colab 的步骤
echo ==========================================
echo.
echo 方法 1 - 直接上传（推荐）：
echo   1. 打开 https://colab.research.google.com/
echo   2. 点击左侧 📁 文件图标
echo   3. 点击 📤 上传文件
echo   4. 选择 colab_notebooks.zip
echo   5. 运行：!unzip -o colab_notebooks.zip
echo   6. 打开任意 notebook 开始学习！
echo.
echo 方法 2 - 查看详细说明：
echo   查看 UPLOAD_TO_COLAB.md 文件
echo.
echo ==========================================
echo.

REM 询问是否打开上传指南
set /p OPEN_GUIDE="是否打开上传指南？(Y/N): "
if /i "%OPEN_GUIDE%"=="Y" (
    if exist UPLOAD_TO_COLAB.md (
        start UPLOAD_TO_COLAB.md
    )
)

REM 询问是否打开 Colab
set /p OPEN_COLAB="是否打开 Colab？(Y/N): "
if /i "%OPEN_COLAB%"=="Y" (
    start https://colab.research.google.com/
)

echo.
echo ✅ 准备完成！
echo.
pause
