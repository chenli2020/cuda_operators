#!/bin/bash
# 自动同步到 GitHub，方便在 Colab 中使用

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  CUDA Operators - GitHub 同步工具"
echo "=========================================="
echo ""

# 检查是否是 git 仓库
if [ ! -d .git ]; then
    echo "❌ 错误：当前目录不是 git 仓库"
    echo "   请先运行: git init"
    exit 1
fi

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 检测到未提交的更改..."

    # 添加所有 Colab 相关文件
    echo "   添加 Colab notebooks..."
    git add colab_*.ipynb
    git add COLAB_GUIDE.md
    git add colab/
    git add scripts/colab_test_runner.py
    git add README.md

    # 提交
    echo "   提交更改..."
    git commit -m "Update Colab notebooks and tools

- Add/update Colab notebooks for all operators
- Add Colab usage guide
- Add performance profiling tools
- Update README with Colab quick start"
else
    echo "✓ 没有新的更改需要提交"
fi

# 检查是否配置了远程仓库
if ! git remote get-url origin >/dev/null 2>&1; then
    echo ""
    echo "⚠️  未配置 GitHub 远程仓库"
    echo ""
    echo "请按以下步骤操作："
    echo "1. 在 GitHub 上创建新仓库：https://github.com/new"
    echo "2. 运行以下命令（替换 YOUR_USERNAME）："
    echo "   git remote add origin https://github.com/YOUR_USERNAME/cuda_operators.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    exit 1
fi

# 推送到 GitHub
echo ""
echo "🚀 推送到 GitHub..."
git push origin main

echo ""
echo "✓ 同步完成！"
echo ""
echo "=========================================="
echo "  在 Colab 中打开："
echo "=========================================="
echo ""
REMOTE_URL=$(git remote get-url origin)
echo "1. 访问：https://colab.research.google.com/"
echo "2. 点击：文件 → 在 GitHub 中打开笔记本"
echo "3. 粘贴仓库 URL："
echo "   $REMOTE_URL"
echo "4. 选择：colab_layernorm.ipynb（或其他 notebook）"
echo ""
echo "或使用直接链接："
echo "https://colab.research.google.com/github/${REMOTE_URL#*.com/}/blob/main/colab_layernorm.ipynb"
echo ""
