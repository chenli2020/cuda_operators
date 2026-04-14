@echo off
REM 直接运行 Python 学习脚本 - 不需要 Jupyter

echo ==========================================
echo   CUDA Operators - Python 学习模式
echo ==========================================
echo.

REM 1. 编译
echo 🔨 编译 CUDA 扩展...
python setup.py build_ext --inplace

REM 2. 运行各算子的测试和学习
echo.
echo 📚 开始学习...
echo.

echo ==========================================
echo 1️⃣ LayerNorm 学习
echo ==========================================
python tests\test_layernorm.py
echo.

echo ==========================================
echo 2️⃣ RMSNorm 学习
echo ==========================================
python tests\test_rmsnorm.py
echo.

echo ==========================================
echo 3️⃣ Softmax 学习
echo ==========================================
python tests\test_softmax.py
echo.

echo ==========================================
echo 4️⃣ Reduce 学习
echo ==========================================
python tests\test_reduce.py
echo.

echo ==========================================
echo 5️⃣ MatMul 学习
echo ==========================================
python tests\test_matmul.py
echo.

echo ==========================================
echo 🎉 所有学习完成！
echo ==========================================
echo.
echo 💡 下一步：
echo    - 修改测试文件中的参数重新实验
echo    - 查看 benchmark\ 了解性能对比
echo    - 阅读 src\ops\ 中的实现代码
echo.

pause
