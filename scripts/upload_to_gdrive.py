#!/usr/bin/env python3
"""
自动上传 Colab notebooks 到 Google Drive

使用 Google Drive API 将文件上传到 Google Drive，
然后在 Google Colab 中直接打开。

首次使用需要配置：
1. 启用 Google Drive API
2. 下载 credentials.json
3. 运行此脚本进行认证
"""

import os
import sys
import pickle
from pathlib import Path

# 检查依赖
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("❌ 缺少必要的依赖包")
    print("请运行：pip install google-api-python-client google-auth-oauthlib")
    sys.exit(1)

# 配置
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
NOTEBOOKS = [
    'colab_layernorm.ipynb',
    'colab_rmsnorm.ipynb',
    'colab_softmax.ipynb',
    'colab_reduce.ipynb',
    'colab_matmul.ipynb',
]


def authenticate():
    """Google Drive API 认证"""
    creds = None

    # 加载已保存的凭证
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)

    # 如果没有有效凭证，进行认证
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print("\n" + "="*60)
                print("  首次使用需要配置 Google Drive API")
                print("="*60)
                print("\n请按以下步骤操作：")
                print("\n1. 访问 Google Cloud Console:")
                print("   https://console.cloud.google.com/")
                print("\n2. 创建新项目或选择现有项目")
                print("\n3. 启用 Google Drive API:")
                print("   https://console.cloud.google.com/apis/library/drive.googleapis.com")
                print("\n4. 创建 OAuth 2.0 凭证:")
                print("   https://console.cloud.google.com/apis/credentials")
                print("   - 点击 '创建凭证' → 'OAuth 客户端 ID'")
                print("   - 应用类型选择 '桌面应用'")
                print("   - 下载凭证，重命名为 'credentials.json'")
                print("   - 将 'credentials.json' 放在当前目录")
                print("\n5. 重新运行此脚本")
                print("\n" + "="*60)
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # 保存凭证
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    return creds


def upload_file(service, filename, folder_id=None):
    """上传文件到 Google Drive"""
    file_metadata = {
        'name': os.path.basename(filename),
    }

    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(filename, resumable=True)

    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        print(f"  ✓ 上传成功: {os.path.basename(filename)}")
        print(f"    ID: {file.get('id')}")
        return file.get('id')

    except Exception as e:
        print(f"  ✗ 上传失败: {os.path.basename(filename)}")
        print(f"    错误: {e}")
        return None


def find_or_create_folder(service, folder_name='CUDA_Operators'):
    """查找或创建文件夹"""
    # 搜索现有文件夹
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        fields='files(id, name)'
    ).execute()

    files = results.get('files', [])

    if files:
        print(f"  ✓ 使用现有文件夹: {folder_name}")
        return files[0]['id']

    # 创建新文件夹
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }

    file = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()

    print(f"  ✓ 创建文件夹: {folder_name}")
    return file.get('id')


def main():
    """主函数"""
    print("="*60)
    print("  CUDA Operators - Google Drive 上传工具")
    print("="*60)
    print()

    # 检查 notebook 文件是否存在
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    missing_files = [nb for nb in NOTEBOOKS if not os.path.exists(nb)]
    if missing_files:
        print(f"⚠️  以下文件不存在：")
        for f in missing_files:
            print(f"   - {f}")
        print()

        # 检查是否有任何 notebook 文件
        existing_notebooks = list(Path('.').glob('colab_*.ipynb'))
        if not existing_notebooks:
            print("❌ 未找到任何 Colab notebook 文件")
            sys.exit(1)

        print(f"将上传现有的 {len(existing_notebooks)} 个文件\n")
        NOTEBOOKS[:] = [str(nb) for nb in existing_notebooks]

    # 认证
    print("🔐 正在进行 Google Drive 认证...")
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    # 创建或查找文件夹
    print("\n📁 准备 Google Drive 文件夹...")
    folder_id = find_or_create_folder(service, 'CUDA_Operators')

    # 上传文件
    print("\n📤 上传 Colab notebooks...")
    print()

    uploaded_ids = []
    for notebook in NOTEBOOKS:
        file_id = upload_file(service, notebook, folder_id)
        if file_id:
            uploaded_ids.append((notebook, file_id))

    # 生成 Colab 链接
    if uploaded_ids:
        print("\n" + "="*60)
        print("  上传完成！在 Colab 中打开：")
        print("="*60)
        print()

        for notebook, file_id in uploaded_ids:
            colab_url = f"https://colab.research.google.com/drive/{file_id}"
            print(f"📓 {notebook}:")
            print(f"   {colab_url}")
            print()

        print("💡 提示：")
        print("   - 点击上面的链接直接在 Colab 中打开")
        print("   - 记得在 Colab 中启用 GPU（运行时 → 更改运行时类型 → T4 GPU）")
        print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
