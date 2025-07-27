#!/usr/bin/env python3
"""
清理IsaacLab目录中的项目文件

这个脚本会帮助删除IsaacLab根目录中的项目文件，
保持IsaacLab目录的整洁。
"""

import os
import shutil
import sys
from pathlib import Path
import argparse
import json

# 需要删除的文件列表（基于之前的迁移映射）
FILES_TO_DELETE = [
    "cabinet_rl_with_sensors_new.py",
    "cabinet_rl_BASELINE.py", 
    "run_sensor_comparison_experiment.py",
    "analyze_experiment_results.py",
    "enhanced_sensor_vs_baseline_experiment.py",
    "simple_contact_sensor_demo.py",
    "contact_sensor_diagnostic.py",
    "quick_sensor_test.py",
    "README_sensor_experiment.md",
    "RL_Contact_Sensor_Experiment_Report.md",
    "conversation_summary.md",
    "baseline_seed44_result.json",
    "baseline_seed43_result.json",
    "baseline_seed42_result.json",
    "sensor_seed44_result.json",
    "sensor_seed43_result.json",
    "sensor_seed42_result.json",
    "comparison_results_20250707_225934.json",
    "comparison_results_20250707_233038.json",
    "comparison_results_20250708_003119.json",
    "SENSOR_enhanced_real_200.log",
    "SENSOR_reference_200.log",
    "baseline_training.log",
    "tensorboard.log",
    "train_baseline.sh",
    "tactile_vs_baseline_experiment.sh",
    "start_conversation.sh",
]

# 需要删除的目录列表
DIRS_TO_DELETE = [
    "experiments",
    "experiment_analysis",
    "plots",
    "outputs",
]

def cleanup_isaaclab_directory(isaaclab_dir: str, dry_run: bool = False):
    """清理IsaacLab目录"""
    isaaclab_path = Path(isaaclab_dir)
    
    if not isaaclab_path.exists():
        print(f"❌ IsaacLab目录不存在: {isaaclab_path}")
        return False
    
    print(f"🧹 清理IsaacLab目录: {isaaclab_path}")
    
    deleted_files = []
    deleted_dirs = []
    errors = []
    
    # 删除文件
    print("\n📄 删除文件:")
    for filename in FILES_TO_DELETE:
        file_path = isaaclab_path / filename
        if file_path.exists():
            if dry_run:
                print(f"🔍 [DRY RUN] 将删除文件: {file_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"✅ 已删除文件: {file_path}")
                    deleted_files.append(str(file_path))
                except Exception as e:
                    print(f"❌ 删除文件失败: {file_path} - {e}")
                    errors.append(f"File: {file_path} - {e}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    # 删除目录
    print("\n📁 删除目录:")
    for dirname in DIRS_TO_DELETE:
        dir_path = isaaclab_path / dirname
        if dir_path.exists() and dir_path.is_dir():
            if dry_run:
                print(f"🔍 [DRY RUN] 将删除目录: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    print(f"✅ 已删除目录: {dir_path}")
                    deleted_dirs.append(str(dir_path))
                except Exception as e:
                    print(f"❌ 删除目录失败: {dir_path} - {e}")
                    errors.append(f"Directory: {dir_path} - {e}")
        else:
            print(f"⚠️  目录不存在: {dir_path}")
    
    # 显示清理结果
    print(f"\n📊 清理结果:")
    print(f"✅ 已删除文件: {len(deleted_files)}")
    print(f"✅ 已删除目录: {len(deleted_dirs)}")
    print(f"❌ 错误: {len(errors)}")
    
    if errors:
        print("\n❌ 错误详情:")
        for error in errors:
            print(f"  - {error}")
    
    # 保存清理日志
    if not dry_run:
        cleanup_log = {
            "deleted_files": deleted_files,
            "deleted_dirs": deleted_dirs,
            "errors": errors,
            "timestamp": None
        }
        
        log_file = Path("cleanup_log.json")
        with open(log_file, 'w') as f:
            json.dump(cleanup_log, f, indent=2, default=str)
        
        print(f"\n📋 清理日志已保存到: {log_file}")
    
    return len(errors) == 0

def main():
    parser = argparse.ArgumentParser(description="清理IsaacLab目录中的项目文件")
    parser.add_argument(
        "--isaaclab-dir",
        default="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab",
        help="IsaacLab目录路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要删除的文件，不实际删除"
    )
    
    args = parser.parse_args()
    
    # 验证IsaacLab目录
    isaaclab_path = Path(args.isaaclab_dir)
    if not isaaclab_path.exists():
        print(f"❌ IsaacLab目录不存在: {isaaclab_path}")
        sys.exit(1)
    
    # 确认操作
    if not args.dry_run:
        print("⚠️  警告: 这将永久删除IsaacLab目录中的项目文件!")
        print("⚠️  请确保你已经成功迁移了所有重要文件!")
        print(f"📁 目标目录: {isaaclab_path}")
        
        confirm = input("确认继续删除? (输入 'DELETE' 确认): ").strip()
        if confirm != 'DELETE':
            print("❌ 操作已取消")
            sys.exit(0)
    
    # 执行清理
    success = cleanup_isaaclab_directory(args.isaaclab_dir, args.dry_run)
    
    if success:
        print("\n🎉 清理完成!")
        print("✅ IsaacLab目录已恢复整洁")
    else:
        print("\n⚠️  清理过程中遇到一些问题")
        print("📋 请检查错误日志并手动处理")
        sys.exit(1)

if __name__ == "__main__":
    main()
