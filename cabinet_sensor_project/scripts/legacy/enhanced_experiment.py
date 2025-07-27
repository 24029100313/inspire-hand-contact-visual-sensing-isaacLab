#!/usr/bin/env python3
"""
增强版传感器 vs 基线版本 RL 性能对比实验

新增功能：
- 🎯 成功率统计
- 📏 开启程度记录 
- 🎬 关键环节视频记录
- 📁 每次实验建立特定的文件夹
- 📊 详细的性能分析

用法:
    python enhanced_sensor_vs_baseline_experiment.py --num_envs 64 --max_iterations 100
"""

import argparse
import subprocess
import time
import os
import psutil
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import re


class ExperimentManager:
    """实验管理器 - 负责创建实验文件夹和组织输出"""
    
    def __init__(self, base_name: str = "cabinet_experiment"):
        self.base_name = base_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_root = Path(f"experiments/{base_name}_{self.timestamp}")
        
        # 创建实验目录结构
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.baseline_dir = self.experiment_root / "baseline"
        self.sensor_dir = self.experiment_root / "sensor"
        self.videos_dir = self.experiment_root / "videos"
        self.logs_dir = self.experiment_root / "logs"
        self.results_dir = self.experiment_root / "results"
        
        # 创建所有子目录
        for dir_path in [self.baseline_dir, self.sensor_dir, self.videos_dir, 
                        self.logs_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"[实验管理] 实验目录已创建: {self.experiment_root}")
        print(f"[实验管理] 目录结构:")
        print(f"├── baseline/     (基线版本输出)")
        print(f"├── sensor/       (传感器版本输出)")
        print(f"├── videos/       (关键环节视频)")
        print(f"├── logs/         (训练日志)")
        print(f"└── results/      (分析结果)")
    
    def get_experiment_path(self, version_type: str) -> Path:
        """获取特定版本的实验路径"""
        if version_type == "baseline":
            return self.baseline_dir
        elif version_type == "sensors":
            return self.sensor_dir
        else:
            raise ValueError(f"Unknown version type: {version_type}")
    
    def get_video_path(self, version_type: str, video_type: str) -> Path:
        """获取视频保存路径"""
        return self.videos_dir / f"{version_type}_{video_type}"
    
    def get_log_path(self, version_type: str) -> Path:
        """获取日志保存路径"""
        return self.logs_dir / f"{version_type}_training.log"
    
    def get_results_path(self, filename: str) -> Path:
        """获取结果文件路径"""
        return self.results_dir / filename


class TaskAnalyzer:
    """任务分析器 - 专门用于分析抽屉开启任务"""
    
    def __init__(self):
        self.success_episodes = []
        self.failure_episodes = []
        self.opening_progress = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.contact_events = []
        
        # 成功标准
        self.success_threshold = 0.25  # 抽屉开启距离阈值（米）
        self.reward_threshold = 50.0   # 奖励阈值
        
    def parse_episode_from_log(self, log_line: str) -> Optional[Dict]:
        """从日志行中解析episode信息"""
        try:
            # 解析成功的episode
            if "episode" in log_line.lower() and "reward" in log_line.lower():
                # 提取episode信息
                episode_match = re.search(r'episode[:\s]*(\d+)', log_line.lower())
                reward_match = re.search(r'reward[:\s]*([+-]?\d+\.?\d*)', log_line.lower())
                
                if episode_match and reward_match:
                    episode_id = int(episode_match.group(1))
                    total_reward = float(reward_match.group(1))
                    
                    # 估算开启距离（基于奖励）
                    estimated_opening = max(0, (total_reward - 10) / 100)  # 简化估算
                    
                    return {
                        'episode_id': episode_id,
                        'total_reward': total_reward,
                        'estimated_opening': estimated_opening,
                        'is_success': total_reward > self.reward_threshold
                    }
            
            return None
            
        except Exception:
            return None
    
    def analyze_training_log(self, log_path: Path):
        """分析训练日志"""
        if not log_path.exists():
            return
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                episode_data = self.parse_episode_from_log(line)
                if episode_data:
                    self.episode_rewards.append(episode_data['total_reward'])
                    self.opening_progress.append(episode_data['estimated_opening'])
                    
                    if episode_data['is_success']:
                        self.success_episodes.append(episode_data)
                    else:
                        self.failure_episodes.append(episode_data)
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        total_episodes = len(self.success_episodes) + len(self.failure_episodes)
        if total_episodes == 0:
            return 0.0
        return len(self.success_episodes) / total_episodes
    
    def get_average_opening_distance(self) -> float:
        """获取平均开启距离"""
        if not self.opening_progress:
            return 0.0
        return np.mean(self.opening_progress)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取任务分析摘要"""
        total_episodes = len(self.success_episodes) + len(self.failure_episodes)
        
        if total_episodes == 0:
            return {
                'total_episodes': 0,
                'success_rate': 0.0,
                'average_opening_distance': 0.0,
                'average_episode_reward': 0.0,
                'max_opening_distance': 0.0,
                'successful_episodes': 0,
                'failed_episodes': 0
            }
        
        return {
            'total_episodes': total_episodes,
            'successful_episodes': len(self.success_episodes),
            'failed_episodes': len(self.failure_episodes),
            'success_rate': self.get_success_rate(),
            'average_opening_distance': self.get_average_opening_distance(),
            'max_opening_distance': np.max(self.opening_progress) if self.opening_progress else 0.0,
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'opening_distance_std': np.std(self.opening_progress) if self.opening_progress else 0.0,
        }


class PerformanceMonitor:
    """系统性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self.system_memory_samples = []
        self.monitor_thread = None
        self.start_time = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # 系统内存使用率
                memory = psutil.virtual_memory()
                self.system_memory_samples.append(memory.percent)
                
                # GPU监控（如果可用）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_gb = memory_info.used / (1024 ** 3)
                    self.gpu_memory_samples.append(gpu_memory_gb)
                except:
                    self.gpu_memory_samples.append(0.0)
                
                time.sleep(2.0)
                
            except Exception:
                continue
    
    def get_summary(self) -> Dict[str, float]:
        """获取监控摘要"""
        if not self.cpu_samples:
            return {}
        
        return {
            'avg_cpu_percent': np.mean(self.cpu_samples),
            'max_cpu_percent': np.max(self.cpu_samples),
            'avg_system_memory_percent': np.mean(self.system_memory_samples),
            'avg_gpu_memory_gb': np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0.0,
            'max_gpu_memory_gb': np.max(self.gpu_memory_samples) if self.gpu_memory_samples else 0.0,
        }


def parse_training_output(line: str) -> Optional[Dict]:
    """解析训练输出"""
    result = {}
    
    try:
        # 解析迭代信息
        if "it/" in line.lower():
            it_match = re.search(r'it/(\d+)', line.lower())
            if it_match:
                result['iteration'] = int(it_match.group(1))
        
        # 解析奖励信息
        reward_match = re.search(r'reward[:\s]*([+-]?\d+\.?\d*)', line.lower())
        if reward_match:
            result['reward'] = float(reward_match.group(1))
        
        # 解析FPS信息
        fps_match = re.search(r'(\d+\.?\d*)\s*fps', line.lower())
        if fps_match:
            result['fps'] = float(fps_match.group(1))
        
        return result if result else None
        
    except Exception:
        return None


def run_enhanced_experiment(version_type: str, num_envs: int, max_iterations: int, 
                           seed: int, experiment_manager: ExperimentManager) -> Dict[str, Any]:
    """运行增强版实验"""
    
    # 确定使用的脚本
    if version_type == "baseline":
        script = "cabinet_rl_BASELINE.py"
        name = "基线版本（无传感器）"
    elif version_type == "sensors":
        script = "cabinet_rl_WITH_SENSORS.py"
        name = "传感器版本"
    else:
        raise ValueError(f"Unknown version type: {version_type}")
    
    print(f"\n{'='*70}")
    print(f"   🚀 开始增强实验: {name}")
    print(f"   🌍 环境数: {num_envs}")
    print(f"   📊 最大迭代: {max_iterations}")
    print(f"   🎲 随机种子: {seed}")
    print(f"   🕐 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    # 获取实验路径
    experiment_path = experiment_manager.get_experiment_path(version_type)
    log_path = experiment_manager.get_log_path(version_type)
    
    # 启动性能监控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 构建命令，添加视频录制
    video_dir = experiment_manager.get_video_path(version_type, "training")
    video_dir.mkdir(exist_ok=True)
    
    cmd = [
        "./isaaclab.sh", "-p", script,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--seed", str(seed),
        "--video",  # 启用视频录制
        "--video_length", "200",  # 视频长度
        "--video_interval", "500",  # 录制间隔
        "--headless"
    ]
    
    print(f"📹 视频录制已启用: {video_dir}")
    print(f"📊 性能监控已启动")
    
    start_time = time.time()
    training_results = {
        'final_reward': 0.0,
        'final_steps_per_sec': 0.0,
        'total_steps': 0,
        'reward_history': [],
        'iteration_history': [],
        'fps_history': [],
        'max_reward_achieved': -float('inf'),
        'min_reward_achieved': float('inf')
    }
    
    # 保存训练日志
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"实验开始: {datetime.now().isoformat()}\n")
    log_file.write(f"配置: {version_type}, {num_envs} envs, {max_iterations} iters, seed {seed}\n")
    log_file.write("-" * 50 + "\n")
    
    try:
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("📊 实时监控训练进度:")
        print("   格式: [时间] 迭代 | 奖励 | 步数/秒")
        print("-" * 60)
        
        iteration_count = 0
        last_reward = 0.0
        last_fps = 0.0
        last_progress_time = time.time()
        timeout_duration = 3600  # 60分钟超时
        
        # 实时读取输出
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            line = line.strip()
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # 保存到日志文件
            log_file.write(f"[{current_time}] {line}\n")
            log_file.flush()
            
            # 解析训练信息
            parsed_info = parse_training_output(line)
            if parsed_info:
                if 'iteration' in parsed_info:
                    iteration_count = parsed_info['iteration']
                    last_progress_time = time.time()
                
                if 'reward' in parsed_info:
                    last_reward = parsed_info['reward']
                    training_results['reward_history'].append(last_reward)
                    training_results['iteration_history'].append(iteration_count)
                    
                    # 更新最大最小奖励
                    training_results['max_reward_achieved'] = max(
                        training_results['max_reward_achieved'], last_reward
                    )
                    training_results['min_reward_achieved'] = min(
                        training_results['min_reward_achieved'], last_reward
                    )
                
                if 'fps' in parsed_info:
                    last_fps = parsed_info['fps']
                    training_results['fps_history'].append(last_fps)
            
            # 显示进度（每10个迭代显示一次）
            if iteration_count > 0 and iteration_count % 10 == 0:
                print(f"[{current_time}] 迭代 {iteration_count:3d} | "
                      f"奖励 {last_reward:6.2f} | "
                      f"FPS {last_fps:4.0f}")
            
            # 检查超时
            if time.time() - last_progress_time > timeout_duration:
                print(f"⚠️ 警告: {timeout_duration}秒内无进展，终止进程")
                process.kill()
                break
        
        # 等待进程完成
        process.wait()
        
    except Exception as e:
        print(f"❌ 实验执行错误: {e}")
        log_file.write(f"错误: {e}\n")
    
    finally:
        log_file.close()
        monitor.stop_monitoring()
    
    # 分析任务性能
    task_analyzer = TaskAnalyzer()
    task_analyzer.analyze_training_log(log_path)
    
    # 完成训练结果
    total_duration = time.time() - start_time
    training_results['final_reward'] = last_reward
    training_results['final_steps_per_sec'] = last_fps
    training_results['total_steps'] = iteration_count * num_envs
    
    # 获取性能摘要
    performance_summary = monitor.get_summary()
    
    # 获取任务分析摘要
    task_summary = task_analyzer.get_summary()
    
    # 构建完整结果
    result = {
        'version': version_type,
        'name': name,
        'duration_min': total_duration / 60,
        'training_results': training_results,
        'performance': performance_summary,
        'task_analysis': task_summary,
        'config': {
            'num_envs': num_envs,
            'max_iterations': max_iterations,
            'seed': seed
        },
        'paths': {
            'experiment_dir': str(experiment_path),
            'log_file': str(log_path),
            'video_dir': str(video_dir)
        }
    }
    
    # 显示结果摘要
    print(f"\n✅ {name} 实验完成!")
    print(f"   ⏱️ 总用时: {total_duration/60:.1f} 分钟")
    print(f"   🎯 最终奖励: {training_results['final_reward']:.2f}")
    print(f"   🚀 最终FPS: {training_results['final_steps_per_sec']:.0f}")
    print(f"   📊 成功率: {task_summary['success_rate']:.1%}")
    print(f"   📏 平均开启距离: {task_summary['average_opening_distance']:.3f}m")
    print(f"   💻 平均CPU: {performance_summary.get('avg_cpu_percent', 0):.1f}%")
    print(f"   🎮 平均GPU内存: {performance_summary.get('avg_gpu_memory_gb', 0):.1f}GB")
    
    return result


def compare_enhanced_results(baseline_result: Dict, sensor_result: Dict):
    """对比分析增强实验结果"""
    
    print(f"\n{'='*70}")
    print(" 📊 增强实验结果对比分析")
    print(f"{'='*70}")
    
    # 基本信息
    print("\n📋 实验配置:")
    print(f"├── 环境数: {baseline_result['config']['num_envs']}")
    print(f"├── 最大迭代: {baseline_result['config']['max_iterations']}")
    print(f"└── 随机种子: {baseline_result['config']['seed']}")
    
    # 任务性能对比
    print("\n🎯 任务性能对比:")
    baseline_task = baseline_result['task_analysis']
    sensor_task = sensor_result['task_analysis']
    
    # 成功率对比
    success_rate_diff = sensor_task['success_rate'] - baseline_task['success_rate']
    print(f"├── 成功率:")
    print(f"│   ├── 基线版本: {baseline_task['success_rate']:.1%}")
    print(f"│   ├── 传感器版本: {sensor_task['success_rate']:.1%}")
    print(f"│   └── 差异: {success_rate_diff:+.1%}")
    
    # 开启距离对比
    opening_diff = sensor_task['average_opening_distance'] - baseline_task['average_opening_distance']
    print(f"├── 平均开启距离:")
    print(f"│   ├── 基线版本: {baseline_task['average_opening_distance']:.3f}m")
    print(f"│   ├── 传感器版本: {sensor_task['average_opening_distance']:.3f}m")
    print(f"│   └── 差异: {opening_diff:+.3f}m")
    
    # 奖励对比
    baseline_training = baseline_result['training_results']
    sensor_training = sensor_result['training_results']
    
    reward_diff = sensor_training['final_reward'] - baseline_training['final_reward']
    reward_diff_pct = (reward_diff / baseline_training['final_reward']) * 100 if baseline_training['final_reward'] != 0 else 0
    
    print(f"└── 最终奖励:")
    print(f"    ├── 基线版本: {baseline_training['final_reward']:.3f}")
    print(f"    ├── 传感器版本: {sensor_training['final_reward']:.3f}")
    print(f"    └── 差异: {reward_diff:+.3f} ({reward_diff_pct:+.1f}%)")
    
    # 综合结论
    print("\n📈 综合结论:")
    
    if success_rate_diff > 0.1:
        task_conclusion = "✅ 传感器显著提高了任务成功率"
    elif success_rate_diff > 0.05:
        task_conclusion = "✅ 传感器轻微提高了任务成功率"
    elif success_rate_diff > -0.05:
        task_conclusion = "➖ 传感器对任务成功率影响很小"
    else:
        task_conclusion = "❌ 传感器降低了任务成功率"
    
    print(f"└── 任务性能: {task_conclusion}")


def save_enhanced_results(baseline_result: Dict, sensor_result: Dict, 
                         experiment_manager: ExperimentManager):
    """保存增强实验结果"""
    
    # 主要结果文件
    results_file = experiment_manager.get_results_path("experiment_results.json")
    
    # 构建完整结果
    enhanced_results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'description': '增强版传感器vs基线版本RL性能对比实验',
            'features': [
                '成功率统计',
                '开启程度记录',
                '关键环节视频记录',
                '详细性能分析',
                '独立实验文件夹'
            ]
        },
        'baseline_result': baseline_result,
        'sensor_result': sensor_result,
        'comparison': {
            'success_rate_difference': (sensor_result['task_analysis']['success_rate'] - 
                                      baseline_result['task_analysis']['success_rate']),
            'opening_distance_difference': (sensor_result['task_analysis']['average_opening_distance'] - 
                                          baseline_result['task_analysis']['average_opening_distance']),
            'reward_difference_pct': ((sensor_result['training_results']['final_reward'] - 
                                     baseline_result['training_results']['final_reward']) / 
                                    baseline_result['training_results']['final_reward'] * 100) if baseline_result['training_results']['final_reward'] != 0 else 0
        },
        'experiment_paths': {
            'experiment_root': str(experiment_manager.experiment_root),
            'baseline_dir': str(experiment_manager.baseline_dir),
            'sensor_dir': str(experiment_manager.sensor_dir),
            'videos_dir': str(experiment_manager.videos_dir),
            'logs_dir': str(experiment_manager.logs_dir),
            'results_dir': str(experiment_manager.results_dir)
        }
    }
    
    # 保存结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
    
    # 保存简化摘要
    summary_file = experiment_manager.get_results_path("experiment_summary.json")
    summary = {
        'timestamp': enhanced_results['experiment_info']['timestamp'],
        'baseline_success_rate': baseline_result['task_analysis']['success_rate'],
        'sensor_success_rate': sensor_result['task_analysis']['success_rate'],
        'success_rate_improvement': enhanced_results['comparison']['success_rate_difference'],
        'baseline_opening_distance': baseline_result['task_analysis']['average_opening_distance'],
        'sensor_opening_distance': sensor_result['task_analysis']['average_opening_distance'],
        'opening_distance_improvement': enhanced_results['comparison']['opening_distance_difference'],
        'reward_improvement_pct': enhanced_results['comparison']['reward_difference_pct'],
        'experiment_root': str(experiment_manager.experiment_root)
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 实验结果已保存:")
    print(f"├── 详细结果: {results_file}")
    print(f"└── 摘要结果: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="增强版传感器vs基线版本RL性能对比实验")
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数")
    parser.add_argument("--max_iterations", type=int, default=100, help="最大训练迭代数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--experiment_name", type=str, default="cabinet_sensor_comparison", 
                       help="实验名称")
    
    args = parser.parse_args()
    
    # 创建实验管理器
    experiment_manager = ExperimentManager(args.experiment_name)
    
    print("🔬 增强版传感器 vs 基线版本 RL 性能对比实验")
    print("=" * 70)
    print(f"实验配置:")
    print(f"├── 并行环境数: {args.num_envs}")
    print(f"├── 最大迭代数: {args.max_iterations}")
    print(f"├── 随机种子: {args.seed}")
    print(f"├── 实验名称: {args.experiment_name}")
    print(f"└── 实验目录: {experiment_manager.experiment_root}")
    
    print(f"\n🎯 新增功能:")
    print(f"├── ✅ 成功率统计")
    print(f"├── 📏 开启程度记录")
    print(f"├── 🎬 关键环节视频记录")
    print(f"├── 📁 独立实验文件夹")
    print(f"└── 📊 详细性能分析")
    
    # 运行基线实验
    print(f"\n🚀 第一阶段: 基线版本实验")
    baseline_result = run_enhanced_experiment("baseline", args.num_envs, 
                                            args.max_iterations, args.seed, 
                                            experiment_manager)
    
    # 等待系统恢复
    print("\n⏳ 等待系统恢复...")
    time.sleep(10)
    
    # 运行传感器实验
    print(f"\n🚀 第二阶段: 传感器版本实验")
    sensor_result = run_enhanced_experiment("sensors", args.num_envs, 
                                          args.max_iterations, args.seed, 
                                          experiment_manager)
    
    # 对比分析结果
    compare_enhanced_results(baseline_result, sensor_result)
    
    # 保存结果
    save_enhanced_results(baseline_result, sensor_result, experiment_manager)
    
    print(f"\n🎉 实验完成！")
    print(f"📁 所有结果已保存到: {experiment_manager.experiment_root}")
    print(f"🎬 视频文件位于: {experiment_manager.videos_dir}")
    print(f"📋 日志文件位于: {experiment_manager.logs_dir}")
    print(f"📊 分析结果位于: {experiment_manager.results_dir}")


if __name__ == "__main__":
    main()
