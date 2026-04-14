#!/usr/bin/env python3
"""
本地文件服务器 - 直接上传到 Colab

在本地运行此脚本，然后在 Colab 中下载文件
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path
import socket
from functools import partial

# 配置
PORT = 8000
PROJECT_DIR = Path(__file__).parent.parent
FILES_TO_SERVE = [
    'colab_layernorm.ipynb',
    'colab_rmsnorm.ipynb',
    'colab_softmax.ipynb',
    'colab_reduce.ipynb',
    'colab_matmul.ipynb',
    'COLAB_GUIDE.md',
]


def get_local_ip():
    """获取本机 IP 地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def create_colab_notebook():
    """创建用于下载文件的 Colab notebook"""
    colab_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 从本地服务器下载 Colab Notebooks\n",
                    "\n",
                    "## 使用步骤：\n",
                    "\n",
                    "1. **在本地运行上传脚本**：\n",
                    "   ```bash\n",
                    "   python scripts/upload_to_colab.py\n",
                    "   ```\n",
                    "\n",
                    "2. **输入本地服务器的地址**（脚本会显示）：\n",
                    "   - 格式：`http://YOUR_LOCAL_IP:8000`\n",
                    "   - 例如：`http://192.168.1.100:8000`\n",
                    "\n",
                    "3. **运行下面的单元格下载文件**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 安装必要的包\n",
                    "!pip install -q torch numpy"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import requests\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# 配置本地服务器地址\n",
                    "LOCAL_SERVER = \"http://YOUR_LOCAL_IP:8000\"  # 替换为脚本显示的地址\n",
                    "\n",
                    "# 要下载的文件\n",
                    "FILES_TO_DOWNLOAD = [\n",
                    "    'colab_layernorm.ipynb',\n",
                    "    'colab_rmsnorm.ipynb',\n",
                    "    'colab_softmax.ipynb',\n",
                    "    'colab_reduce.ipynb',\n",
                    "    'colab_matmul.ipynb',\n",
                    "]\n",
                    "\n",
                    "print(f\"从本地服务器下载文件：{LOCAL_SERVER}\")\n",
                    "print(\"=\"*60)\n",
                    "\n",
                    "# 下载文件\n",
                    "for filename in FILES_TO_DOWNLOAD:\n",
                    "    url = f\"{LOCAL_SERVER}/{filename}\"\n",
                    "    print(f\"下载 {filename}...\")\n",
                    "    \n",
                    "    try:\n",
                    "        response = requests.get(url, timeout=10)\n",
                    "        if response.status_code == 200:\n",
                    "            with open(filename, 'wb') as f:\n",
                    "                f.write(response.content)\n",
                    "            print(f\"  ✓ {filename} 下载成功\")\n",
                    "        else:\n",
                    "            print(f\"  ✗ {filename} 下载失败 (状态码: {response.status_code})\")\n",
                    "    except Exception as e:\n",
                    "        print(f\"  ✗ {filename} 下载失败: {e}\")\n",
                    "\n",
                    "print(\"\\n\" + \"=\"*60)\n",
                    "print(\"下载完成！\")\n",
                    "print(\"\\n现在可以打开 notebook：\")\n",
                    "for filename in FILES_TO_DOWNLOAD:\n",
                    "    if os.path.exists(filename):\n",
                    "        print(f\"  ✓ {filename}\")\n",
                    "        \n",
                    "print(\"\\n\" + \"=\"*60)\n",
                    "print(\"下一步：\")\n",
                    "print(\"1. 在文件浏览器中找到下载的 notebook\")\n",
                    "print(\"2. 双击打开，开始学习！\")\n",
                    "print(\"3. 记得启用 GPU（运行时 → 更改运行时类型 → T4 GPU）\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 或者：直接在 Colab 中打开单个文件\n",
                    "\n",
                    "如果你只想下载特定的文件，可以使用下面的代码："
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 下载单个文件（例如 LayerNorm）\n",
                    "import requests\n",
                    "\n",
                    "LOCAL_SERVER = \"http://YOUR_LOCAL_IP:8000\"  # 替换为你的地址\n",
                    "filename = \"colab_layernorm.ipynb\"  # 可以改成其他文件\n",
                    "\n",
                    "url = f\"{LOCAL_SERVER}/{filename}\"\n",
                    "response = requests.get(url)\n",
                    "\n",
                    "with open(filename, 'wb') as f:\n",
                    "    f.write(response.content)\n",
                    "\n",
                    "print(f\"✓ {filename} 下载完成！\")\n",
                    "print(f\"\\n现在可以在文件浏览器中打开它了\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # 保存为 JSON
    import json
    with open('download_from_local.ipynb', 'w', encoding='utf-8') as f:
        json.dump(colab_content, f, indent=2, ensure_ascii=False)

    print("✓ 已创建 download_from_local.ipynb")


def start_server():
    """启动 HTTP 服务器"""
    # 切换到项目目录
    os.chdir(PROJECT_DIR)

    # 检查文件是否存在
    missing_files = [f for f in FILES_TO_SERVE if not Path(f).exists()]
    if missing_files:
        print("⚠️  以下文件不存在：")
        for f in missing_files:
            print(f"   - {f}")
        print("\n将只提供存在的文件")

    # 获取本机 IP
    local_ip = get_local_ip()

    # 创建自定义请求处理器
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # 只打印重要信息
            if 'GET' in args[0] or 'POST' in args[0]:
                print(f"  📥 {args[0]}")

        def end_headers(self):
            # 添加 CORS 头，允许跨域访问
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()

    # 启动服务器
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print("="*70)
            print("  🚀 本地文件服务器 - 用于上传到 Colab")
            print("="*70)
            print()
            print("📡 服务器信息：")
            print(f"   本地访问：http://localhost:{PORT}")
            print(f"   网络访问：http://{local_ip}:{PORT}")
            print()
            print("📁 提供的文件：")
            for f in FILES_TO_SERVE:
                if Path(f).exists():
                    print(f"   ✓ {f}")
                else:
                    print(f"   ✗ {f} (不存在)")
            print()
            print("📝 在 Colab 中使用：")
            print("   1. 打开 https://colab.research.google.com/")
            print("   2. 新建 notebook 或打开 download_from_local.ipynb")
            print(f"   3. 设置 LOCAL_SERVER = \"http://{local_ip}:{PORT}\"")
            print("   4. 运行下载代码")
            print()
            print("💡 提示：")
            print("   - 确保电脑和 Colab 运行在同一网络")
            print("   - 如果 Colab 在云端，可能需要使用 ngrok 等工具")
            print("   - 按 Ctrl+C 停止服务器")
            print()
            print("="*70)
            print()
            print("🚀 服务器启动中...")
            print(f"   在 Colab 中使用：http://{local_ip}:{PORT}")
            print()
            print("按 Ctrl+C 停止服务器")
            print("-"*70)
            print()

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n")
        print("="*70)
        print("  ✓ 服务器已停止")
        print("="*70)
    except OSError as e:
        if e.errno == 48 or e.errno == 10048:
            print(f"\n❌ 端口 {PORT} 已被占用")
            print("   请尝试：")
            print("   1. 关闭占用端口的程序")
            print("   2. 或修改脚本中的 PORT 变量")
        else:
            print(f"\n❌ 错误：{e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='本地文件服务器 - 用于上传到 Colab')
    parser.add_argument('--create-notebook', action='store_true',
                       help='创建 Colab 下载 notebook')
    parser.add_argument('--port', type=int, default=PORT,
                       help=f'服务器端口（默认：{PORT}）')

    args = parser.parse_args()

    if args.create_notebook:
        create_colab_notebook()
        print("\n✓ 已创建 download_from_local.ipynb")
        print("  将此文件上传到 Colab 并运行")
    else:
        # 先创建 notebook
        create_colab_notebook()
        print("\n" + "="*70)
        print()

        # 启动服务器
        PORT = args.port
        start_server()
