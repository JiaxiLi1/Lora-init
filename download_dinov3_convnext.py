#!/usr/bin/env python3
"""
快速下载 facebook/dinov3-convnext-large-pretrain-lvd1689m 模型到指定目录
支持多种下载方式和加速选项
"""

import os
import sys
import time
import argparse
from pathlib import Path

def download_with_huggingface_hub():
    """使用 huggingface_hub 库下载（推荐方法）"""
    try:
        from huggingface_hub import snapshot_download
        print("✅ 使用 huggingface_hub 进行下载...")
        
        # 目标目录
        target_dir = "/home/rtx3090/code_jiaxi/LORO-main_temp/file_jinjin/dinov3-convnext-large-pretrain-lvd1689m"
        
        print(f"📁 下载目录: {target_dir}")
        print("🚀 开始下载...")
        
        start_time = time.time()
        
        # 下载模型，使用多线程加速
        downloaded_path = snapshot_download(
            repo_id="facebook/dinov3-convnext-large-pretrain-lvd1689m",
            local_dir=target_dir,
            max_workers=8,  # 使用8个线程并行下载
            resume_download=True,  # 支持断点续传
            local_files_only=False,
            ignore_patterns=["*.msgpack", "*.h5"]  # 忽略不需要的文件格式
        )
        
        end_time = time.time()
        download_time = end_time - start_time
        
        print(f"✅ 下载完成！")
        print(f"📁 模型保存在: {downloaded_path}")
        print(f"⏱️ 下载耗时: {download_time:.2f} 秒")
        
        return True
        
    except ImportError:
        print("❌ huggingface_hub 未安装")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_with_git_lfs():
    """使用 git lfs 克隆（备用方法）"""
    print("🔧 使用 git lfs 进行下载...")
    
    target_dir = "/home/rtx3090/code_jiaxi/LORO-main_temp/file_jinjin"
    model_dir = os.path.join(target_dir, "dinov3-convnext-large-pretrain-lvd1689m")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 如果目录已存在，删除它
    if os.path.exists(model_dir):
        import shutil
        print(f"🗑️ 删除现有目录: {model_dir}")
        shutil.rmtree(model_dir)
    
    # 克隆仓库
    clone_cmd = f"""
cd {target_dir} && \
git clone https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m
"""
    
    print("🚀 开始 git clone...")
    start_time = time.time()
    
    result = os.system(clone_cmd)
    
    end_time = time.time()
    download_time = end_time - start_time
    
    if result == 0:
        print(f"✅ Git clone 完成！")
        print(f"📁 模型保存在: {model_dir}")
        print(f"⏱️ 下载耗时: {download_time:.2f} 秒")
        return True
    else:
        print("❌ Git clone 失败")
        return False

def download_with_transformers():
    """使用 transformers 库下载（备用方法）"""
    try:
        from transformers import AutoModel, AutoImageProcessor
        print("🔧 使用 transformers 库进行下载...")
        
        target_dir = "/home/rtx3090/code_jiaxi/LORO-main_temp/file_jinjin/dinov3-convnext-large-pretrain-lvd1689m"
        
        print("🚀 开始下载模型...")
        start_time = time.time()
        
        # 下载模型和处理器
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            cache_dir=target_dir
        )
        
        processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            cache_dir=target_dir
        )
        
        end_time = time.time()
        download_time = end_time - start_time
        
        print(f"✅ 下载完成！")
        print(f"📁 模型缓存在: {target_dir}")
        print(f"⏱️ 下载耗时: {download_time:.2f} 秒")
        
        return True
        
    except ImportError:
        print("❌ transformers 未安装")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    missing_deps = []
    
    try:
        import huggingface_hub
        print("✅ huggingface_hub 已安装")
    except ImportError:
        missing_deps.append("huggingface_hub")
    
    try:
        import transformers
        print("✅ transformers 已安装")
    except ImportError:
        missing_deps.append("transformers")
    
    # 检查 git lfs
    git_lfs_check = os.system("git lfs version > /dev/null 2>&1")
    if git_lfs_check == 0:
        print("✅ git lfs 已安装")
    else:
        print("⚠️ git lfs 未安装")
    
    if missing_deps:
        print(f"\n📦 需要安装的依赖: {', '.join(missing_deps)}")
        print("安装命令:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        print()
    
    return len(missing_deps) == 0

def main():
    parser = argparse.ArgumentParser(description="下载 DINOv3 ConvNeXt Large 模型")
    parser.add_argument("--method", choices=["auto", "hf_hub", "git", "transformers"], 
                       default="auto", help="选择下载方法")
    
    args = parser.parse_args()
    
    print("🤗 DINOv3 ConvNeXt Large 模型下载器")
    print("=" * 60)
    
    # 创建目标目录
    target_base = "/home/rtx3090/code_jiaxi/LORO-main_temp/file_jinjin"
    os.makedirs(target_base, exist_ok=True)
    print(f"📁 目标目录: {target_base}")
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    success = False
    
    if args.method == "auto":
        # 自动选择最佳方法
        print("\n🚀 自动选择最佳下载方法...")
        
        # 优先使用 huggingface_hub
        if not success:
            success = download_with_huggingface_hub()
        
        # 备用方法：git lfs
        if not success:
            print("\n🔄 尝试备用方法...")
            success = download_with_git_lfs()
        
        # 最后备用：transformers
        if not success:
            print("\n🔄 尝试最后的备用方法...")
            success = download_with_transformers()
            
    elif args.method == "hf_hub":
        success = download_with_huggingface_hub()
    elif args.method == "git":
        success = download_with_git_lfs()
    elif args.method == "transformers":
        success = download_with_transformers()
    
    if success:
        print("\n🎉 模型下载成功！")
        print("\n📋 下载的文件:")
        model_path = os.path.join(target_base, "dinov3-convnext-large-pretrain-lvd1689m")
        if os.path.exists(model_path):
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if not file.startswith('.'):  # 忽略隐藏文件
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        total_size += size
                        print(f"  {file} ({size / 1024 / 1024:.1f} MB)")
            print(f"\n📊 总大小: {total_size / 1024 / 1024:.1f} MB")
    else:
        print("\n❌ 所有下载方法都失败了")
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 安装缺失的依赖包")
        print("3. 确保有足够的磁盘空间")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
