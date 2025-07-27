#!/usr/bin/env python3
"""
统一的训练启动脚本

这个脚本提供了一个统一的入口点来运行不同版本的训练，
并正确设置IsaacLab环境和项目路径。
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(config: dict):
    """设置环境变量"""
    # 设置IsaacLab路径
    isaaclab_root = config['paths']['isaaclab_root']
    os.environ['ISAACLAB_ROOT'] = isaaclab_root
    
    # 设置项目路径
    project_root = config['paths']['project_root']
    os.environ['PROJECT_ROOT'] = project_root
    
    # 设置Python路径
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{isaaclab_root}:{project_root}:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = f"{isaaclab_root}:{project_root}"

def run_training(version: str, args: argparse.Namespace):
    """运行训练"""
    # 加载项目配置
    config = load_config('config/project_config.yaml')
    setup_environment(config)
    
    # 根据版本选择配置和脚本
    if version == 'sensor':
        config_file = 'config/sensor_config.yaml'
        script_name = 'scripts/train_sensors.py'
    elif version == 'baseline':
        config_file = 'config/baseline_config.yaml'
        script_name = 'scripts/train_baseline.py'
    else:
        raise ValueError(f"Unknown version: {version}")
    
    # 加载版本特定配置
    version_config = load_config(config_file)
    
    # 构建命令
    isaaclab_root = config['paths']['isaaclab_root']
    
    cmd = [
        f"{isaaclab_root}/isaaclab.sh",
        "-p", script_name,
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.max_iterations),
        "--seed", str(args.seed)
    ]
    
    if args.headless:
        cmd.append("--headless")
    
    if args.disable_fabric:
        cmd.append("--disable_fabric")
    
    print(f"🚀 Running {version} training...")
    print(f"📄 Config: {config_file}")
    print(f"💻 Command: {' '.join(cmd)}")
    
    # 运行训练
    try:
        result = subprocess.run(
            cmd,
            cwd=isaaclab_root,
            check=True
        )
        print(f"✅ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"💥 Training failed with exception: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="统一训练启动脚本")
    parser.add_argument("version", choices=["sensor", "baseline"], help="训练版本")
    parser.add_argument("--num_envs", type=int, default=64, help="环境数量")
    parser.add_argument("--max_iterations", type=int, default=1000, help="最大迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用fabric")
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 运行训练
    exit_code = run_training(args.version, args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
