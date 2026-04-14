#!/usr/bin/env python3
"""
打包 Colab 文件 - 一键上传到 Colab

将所有 Colab 相关文件打包成单个 zip 文件，
然后直接在 Colab 中解压使用。
"""

import os
import zipfile
from pathlib import Path
import sys

# 配置
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_ZIP = "colab_notebooks.zip"
FILES_TO_INCLUDE = [
    'colab_layernorm.ipynb',
    'colab_rmsnorm.ipynb',
    'colab_softmax.ipynb',
    'colab_reduce.ipynb',
    'colab_matmul.ipynb',
    'COLAB_GUIDE.md',
    'QUICKSTART_COLAB.md',
]


def create_zip():
    """创建 zip 文件"""
    print("="*70)
    print("  📦 打包 Colab Notebooks")
    print("="*70)
    print()

    # 切换到项目目录
    os.chdir(PROJECT_DIR)

    # 检查文件是否存在
    missing_files = []
    existing_files = []

    for filename in FILES_TO_INCLUDE:
        filepath = Path(filename)
        if filepath.exists():
            existing_files.append(filename)
            print(f"  ✓ {filename}")
        else:
            missing_files.append(filename)
            print(f"  ✗ {filename} (不存在)")

    if missing_files:
        print()
        print(f"⚠️  警告：{len(missing_files)} 个文件不存在")
        for f in missing_files:
            print(f"     - {f}")

    if not existing_files:
        print()
        print("❌ 没有可打包的文件")
        return False

    # 创建 zip 文件
    print()
    print(f"📦 正在创建 {OUTPUT_ZIP}...")

    try:
        with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in existing_files:
                zipf.write(filename)

        file_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)  # MB

        print()
        print("="*70)
        print("  ✅ 打包完成！")
        print("="*70)
        print()
        print(f"📁 文件：{OUTPUT_ZIP}")
        print(f"📊 大小：{file_size:.2f} MB")
        print(f"📝 包含 {len(existing_files)} 个文件")
        print()

        return True

    except Exception as e:
        print()
        print(f"❌ 打包失败：{e}")
        return False


def create_upload_guide():
    """创建上传指南"""
    guide = f"""# 📤 上传到 Colab 指南

## 方法 1：文件上传（推荐，< 50MB）

1. **打开 Google Colab**
   - 访问：https://colab.research.google.com/

2. **上传 zip 文件**
   - 点击左侧 📁 文件浏览器图标
   - 点击 📤 上传文件按钮
   - 选择项目目录中的 `{OUTPUT_ZIP}`

3. **解压文件**
   - 在第一个单元格中运行：
   ```python
   !unzip -o {OUTPUT_ZIP}
   ```

4. **打开 notebook**
   - 在文件浏览器中找到解压后的文件
   - 双击打开任意 `.ipynb` 文件
   - 开始学习！

---

## 方法 2：Google Drive（大文件推荐）

1. **上传到 Google Drive**
   - 打开 https://drive.google.com/
   - 上传 `{OUTPUT_ZIP}`

2. **在 Colab 中挂载 Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **复制并解压**
   ```python
   !cp /content/drive/MyDrive/{OUTPUT_ZIP} .
   !unzip -o {OUTPUT_ZIP}
   ```

---

## 方法 3：直接运行（最简单）

复制以下代码到 Colab 第一个单元格并运行：

```python
# 上传并解压 Colab notebooks
import zipfile
from google.colab import files

print("请选择上传文件：{OUTPUT_ZIP}")
uploaded = files.upload()

# 解压
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ {{filename}} 解压完成")

# 列出文件
print("\\n可用的 notebooks：")
import os
for f in sorted(os.listdir('.')):
    if f.endswith('.ipynb'):
        print(f"  - {{f}}")

print("\\n在文件浏览器中双击任意 notebook 开始学习！")
```

---

## ✅ 验证上传

解压后，你应该看到以下文件：
"""

    for filename in FILES_TO_INCLUDE:
        guide += f"- `{filename}`\\n"

    guide += """
运行以下命令验证：
```python
!ls -la *.ipynb
```

---

## 🚀 下一步

1. **启用 GPU**
   - 运行时 → 更改运行时类型 → T4 GPU

2. **打开第一个 notebook**
   - 推荐：`colab_layernorm.ipynb`

3. **开始学习**
   - 从上到下依次运行单元格
   - 理解每个优化步骤

---

## 💡 提示

- 如果 zip 文件很大，建议使用 Google Drive 方法
- 解压后可以删除 zip 文件节省空间
- 记得定期保存修改到 Google Drive
"""

    with open('UPLOAD_TO_COLAB.md', 'w', encoding='utf-8') as f:
        f.write(guide)

    print(f"📖 已创建上传指南：UPLOAD_TO_COLAB.md")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='打包 Colab notebooks')
    parser.add_argument('--no-guide', action='store_true',
                       help='不创建上传指南')

    args = parser.parse_args()

    # 创建 zip 文件
    if create_zip():
        # 创建上传指南
        if not args.no_guide:
            create_upload_guide()

        print()
        print("🚀 下一步：")
        print()
        print("1. 打开 Colab：https://colab.research.google.com/")
        print("2. 上传文件到 Colab（拖拽或点击上传）")
        print(f"3. 在第一个单元格运行：!unzip -o {OUTPUT_ZIP}")
        print("4. 打开任意 notebook 开始学习！")
        print()
        print("📖 详细说明：查看 UPLOAD_TO_COLAB.md")
        print()
        print("="*70)
    else:
        print()
        print("❌ 打包失败，请检查文件是否存在")
        sys.exit(1)


if __name__ == '__main__':
    main()
