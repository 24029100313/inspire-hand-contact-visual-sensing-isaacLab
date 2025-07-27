#!/usr/bin/env python3
"""
🧪 传感器对比实验脚本

自动运行传感器版本和基线版本的训练，并进行全面的性能对比分析。

主要功能：
1. 自动运行多种配置的训练实验
2. 实时监控训练进度和性能
3. 收集和解析训练数据
4. 生成详细的对比分析报告
5. 可视化训练曲线和性能指标
6. 自动清理Isaac Sim进程，防止资源占用

新功能：
- 实时训练状态监控（每N次迭代显示进度）
- 智能进程管理和GPU内存监控
- 可配置的清理策略和监控间隔

依赖包：
    pip install psutil

Usage Examples:
    # 基本使用 - 每5次迭代显示一次状态
    python run_sensor_comparison_experiment.py --num_seeds 3 --max_iterations 2000 --num_envs 64
    
    # 自定义状态显示间隔（每10次迭代显示一次）
    python run_sensor_comparison_experiment.py --num_seeds 3 --status_interval 10
    
    # 禁用实时监控，使用传统模式
    python run_sensor_comparison_experiment.py --num_seeds 3 --disable_realtime_monitoring
    
    # 快速测试（少量迭代，频繁状态更新）
    python run_sensor_comparison_experiment.py --num_seeds 1 --max_iterations 100 --status_interval 5
    
    # 长时间训练（自定义超时和清理间隔）
    python run_sensor_comparison_experiment.py --num_seeds 5 --max_iterations 5000 --timeout 14400 --cleanup_wait 120
"""

import argparse
import os
import subprocess
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import yaml
import glob
from typing import Dict, List, Tuple
import shutil
import signal
import re
import select

# 检查并导入psutil
try:
    import psutil
except ImportError:
    print("❌ [ERROR] psutil package is required for process management")
    print("📦 This package is needed to properly clean up Isaac Sim processes")
    
    user_input = input("🤔 Would you like to install psutil automatically? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        print("📥 [INSTALL] Installing psutil...")
        try:
            subprocess.run([
                "pip", "install", "psutil"
            ], check=True)
            print("✅ [SUCCESS] psutil installed successfully!")
            import psutil
        except subprocess.CalledProcessError:
            print("❌ [ERROR] Failed to install psutil automatically")
            print("🔧 Please install manually using: pip install psutil")
            exit(1)
        except Exception as e:
            print(f"❌ [ERROR] Installation failed: {e}")
            print("🔧 Please install manually using: pip install psutil")
            exit(1)
    else:
        print("🔧 Please install psutil manually using: pip install psutil")
        print("   Or if you're using conda: conda install psutil")
        exit(1)


class SensorComparisonExperiment:
    """传感器对比实验管理器"""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f"experiments/sensor_comparison_{self.timestamp}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.configs = {
            'with_sensors': {
                'script': 'cabinet_rl_with_sensors_new.py',
                'name': 'With Sensors',
                'color': '#2E86AB',
                'expected_obs_dim': 43
            },
            'baseline': {
                'script': 'cabinet_rl_BASELINE.py', 
                'name': 'Baseline (No Sensors)',
                'color': '#A23B72',
                'expected_obs_dim': 31
            }
        }
        
        # 结果存储
        self.results = {}
        self.training_logs = {}
        
        print(f"🧪 [EXPERIMENT] Sensor Comparison Experiment")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        print(f"🌱 Seeds: {list(range(args.num_seeds))}")
        print(f"🔄 Max iterations: {args.max_iterations}")
        print(f"🌍 Environments: {args.num_envs}")
    
    def cleanup_isaac_processes(self):
        """清理Isaac Sim相关进程"""
        if self.args.disable_cleanup:
            print("⚠️ [SKIP] Process cleanup disabled by user")
            return
            
        print("🧹 [CLEANUP] Cleaning up Isaac Sim processes...")
        
        # Isaac Sim相关的进程名称
        isaac_process_names = [
            'isaac-sim',
            'python.sh',
            'kit',
            'omni.isaac.sim',
            'nvidia-isaac-sim',
            'libwayland-egl.so'
        ]
        
        # 首先尝试优雅地结束进程
        killed_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] is None:
                    continue
                    
                cmdline_str = ' '.join(proc.info['cmdline']).lower()
                
                # 检查是否是Isaac相关进程
                if any(name in cmdline_str for name in isaac_process_names):
                    print(f"🔍 Found Isaac process: PID={proc.info['pid']}, CMD={cmdline_str[:100]}...")
                    try:
                        proc.terminate()  # 发送SIGTERM
                        killed_processes.append(proc.info['pid'])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # 等待进程结束
        if killed_processes:
            print(f"⏱️ [CLEANUP] Waiting for {len(killed_processes)} processes to terminate...")
            time.sleep(5)
            
            # 强制杀死仍然存在的进程
            for pid in killed_processes:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        print(f"💀 [CLEANUP] Force killing process {pid}")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        # 使用系统命令作为备用方案
        try:
            subprocess.run(["pkill", "-f", "isaac"], capture_output=True, timeout=10)
            subprocess.run(["pkill", "-f", "omni"], capture_output=True, timeout=10)
            subprocess.run(["pkill", "-f", "kit"], capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️ [CLEANUP] pkill command timed out")
        except Exception as e:
            print(f"⚠️ [CLEANUP] pkill failed: {e}")
        
        print("✅ [CLEANUP] Process cleanup completed")
    
    def check_gpu_memory(self):
        """检查GPU内存使用情况"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    used, total = line.split(', ')
                    used_mb = int(used)
                    total_mb = int(total)
                    usage_percent = (used_mb / total_mb) * 100
                    print(f"🖥️ [GPU {i}] Memory: {used_mb}MB / {total_mb}MB ({usage_percent:.1f}%)")
                    
                    if usage_percent > self.args.gpu_memory_threshold:
                        print(f"⚠️ [WARNING] GPU {i} memory usage is high ({usage_percent:.1f}%)")
                        return False
                return True
            else:
                print("⚠️ [WARNING] Could not check GPU memory")
                return True
                
        except Exception as e:
            print(f"⚠️ [WARNING] GPU memory check failed: {e}")
            return True
    
    def wait_for_system_ready(self):
        """等待系统准备就绪"""
        print("⏳ [WAIT] Waiting for system to be ready...")
        
        # 基本等待时间
        time.sleep(10)
        
        # 检查GPU内存，如果占用过高则继续等待
        max_wait_attempts = 6  # 最多等待60秒
        for attempt in range(max_wait_attempts):
            if self.check_gpu_memory():
                break
            else:
                print(f"⏳ [WAIT] GPU memory still high, waiting... (attempt {attempt+1}/{max_wait_attempts})")
                time.sleep(10)
        
        print("✅ [READY] System ready for next training")
    
    def parse_training_output_line(self, line: str) -> Dict:
        """解析训练输出行，提取关键指标"""
        metrics = {}
        
        # 常见的训练指标模式
        patterns = {
            'iteration': r'(?:Iteration|iter|step)[\s:=]+(\d+)',
            'reward': r'(?:reward|episode_reward|mean_reward)[\s:=]+([-+]?\d*\.?\d+)',
            'loss': r'(?:loss|policy_loss|value_loss)[\s:=]+([-+]?\d*\.?\d+)',
            'episode_length': r'(?:episode_length|ep_len)[\s:=]+([-+]?\d*\.?\d+)',
            'success_rate': r'(?:success_rate|success)[\s:=]+([-+]?\d*\.?\d+)',
            'lr': r'(?:learning_rate|lr)[\s:=]+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)',
            'fps': r'(?:fps|FPS)[\s:=]+([-+]?\d*\.?\d+)',
            'time_elapsed': r'(?:time|elapsed)[\s:=]+([-+]?\d*\.?\d+)'
        }
        
        line_lower = line.lower()
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, line_lower)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics
    
    def display_training_status(self, config_name: str, seed: int, iteration: int, metrics: Dict, start_time: float):
        """显示训练状态"""
        config = self.configs[config_name]
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 [STATUS] {config['name']} (seed={seed}) - Iteration {iteration}")
        print(f"⏱️  Elapsed: {elapsed_time:.1f}s")
        
        if metrics:
            # 显示可用的指标
            if 'reward' in metrics:
                print(f"🎯 Reward: {metrics['reward']:.3f}")
            if 'loss' in metrics:
                print(f"📉 Loss: {metrics['loss']:.6f}")
            if 'episode_length' in metrics:
                print(f"📏 Episode Length: {metrics['episode_length']:.1f}")
            if 'success_rate' in metrics:
                print(f"✅ Success Rate: {metrics['success_rate']:.1%}")
            if 'lr' in metrics:
                print(f"📚 Learning Rate: {metrics['lr']:.2e}")
            if 'fps' in metrics:
                print(f"🚀 FPS: {metrics['fps']:.1f}")
        else:
            print("📋 Waiting for metrics...")
        
        # 进度条
        if self.args.max_iterations > 0:
            progress = iteration / self.args.max_iterations
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"📈 Progress: |{bar}| {progress:.1%} ({iteration}/{self.args.max_iterations})")

    def run_single_training(self, config_name: str, seed: int) -> Dict:
        """运行单个训练实验"""
        config = self.configs[config_name]
        script_name = config['script']
        
        print(f"\n🚀 [TRAINING] Starting {config['name']} (seed={seed})")
        print(f"📄 Script: {script_name}")
        
        # 训练前清理
        self.cleanup_isaac_processes()
        self.wait_for_system_ready()
        
        # 构建命令
        cmd = [
            "./isaaclab.sh", "-p", script_name,
            "--num_envs", str(self.args.num_envs),
            "--max_iterations", str(self.args.max_iterations),
            "--seed", str(seed),
            "--headless"
        ]
        
        if self.args.disable_fabric:
            cmd.append("--disable_fabric")
        
        print(f"💻 Command: {' '.join(cmd)}")
        
        # 选择运行模式
        if getattr(self.args, 'disable_realtime_monitoring', False):
            return self._run_training_traditional_mode(config_name, seed, cmd)
        else:
            return self._run_training_realtime_mode(config_name, seed, cmd)
    
    def _run_training_traditional_mode(self, config_name: str, seed: int, cmd: list) -> Dict:
        """传统模式：等待训练完成后处理输出"""
        config = self.configs[config_name]
        start_time = time.time()
        
        print("📊 [TRADITIONAL MODE] Training without real-time monitoring...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab",
                capture_output=True,
                text=True,
                timeout=getattr(self.args, 'timeout', 3600)
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ [SUCCESS] {config['name']} (seed={seed}) completed in {training_time:.1f}s")
                success = True
                error_msg = None
            else:
                print(f"❌ [ERROR] {config['name']} (seed={seed}) failed")
                print(f"stderr: {result.stderr[-500:]}")
                success = False
                error_msg = result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ [TIMEOUT] {config['name']} (seed={seed}) timed out")
            success = False
            error_msg = "Training timed out"
            training_time = getattr(self.args, 'timeout', 3600)
            result = None
            
        except Exception as e:
            print(f"💥 [EXCEPTION] {config['name']} (seed={seed}) failed: {e}")
            success = False
            error_msg = str(e)
            training_time = time.time() - start_time
            result = None
        
        # 训练后清理
        print("🧹 [POST-TRAINING] Cleaning up processes...")
        self.cleanup_isaac_processes()
        
        return {
            'config_name': config_name,
            'seed': seed,
            'success': success,
            'training_time': training_time,
            'error_msg': error_msg,
            'stdout': result.stdout if result else "",
            'stderr': result.stderr if result else error_msg or "",
            'final_iteration': None,
            'final_metrics': {}
        }
    
    def _run_training_realtime_mode(self, config_name: str, seed: int, cmd: list) -> Dict:
        """实时监控模式：实时显示训练进度"""
        config = self.configs[config_name]
        start_time = time.time()
        
        # 实时监控变量
        current_iteration = 0
        status_interval = getattr(self.args, 'status_interval', 5)
        last_displayed_iteration = -status_interval  # 确保第一次迭代就显示
        latest_metrics = {}
        output_lines = []
        error_lines = []
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"🔄 [REALTIME MODE] Training started (PID: {process.pid})")
            print(f"📊 Status updates every {status_interval} iterations")
            
            # 实时读取输出
            while True:
                # 检查进程是否结束
                if process.poll() is not None:
                    break
                
                # 读取stdout
                if process.stdout:
                    try:
                        ready, _, _ = select.select([process.stdout], [], [], 0.1)
                        if ready:
                            line = process.stdout.readline()
                            if line:
                                output_lines.append(line)
                                
                                # 调试：显示所有输出行
                                if "iter" in line.lower() or "episode" in line.lower() or "reward" in line.lower():
                                    print(f"🔍 [DEBUG] Training output: {line.strip()}")
                                
                                # 解析当前行的指标
                                line_metrics = self.parse_training_output_line(line)
                                if line_metrics:
                                    latest_metrics.update(line_metrics)
                                    print(f"📊 [PARSED] Found metrics: {line_metrics}")
                                    
                                    # 更新迭代计数
                                    if 'iteration' in line_metrics:
                                        current_iteration = int(line_metrics['iteration'])
                                
                                # 每N次迭代显示一次状态
                                if (current_iteration > 0 and 
                                    current_iteration % status_interval == 0 and 
                                    current_iteration > last_displayed_iteration):
                                    
                                    self.display_training_status(
                                        config_name, seed, current_iteration, 
                                        latest_metrics, start_time
                                    )
                                    last_displayed_iteration = current_iteration
                    except (OSError, ValueError):
                        # 处理select或读取错误
                        pass
                
                # 读取stderr
                if process.stderr:
                    try:
                        ready, _, _ = select.select([process.stderr], [], [], 0.1)
                        if ready:
                            error_line = process.stderr.readline()
                            if error_line:
                                error_lines.append(error_line)
                    except (OSError, ValueError):
                        pass
                
                # 检查超时
                elapsed = time.time() - start_time
                timeout = getattr(self.args, 'timeout', 3600)
                if elapsed > timeout:
                    print(f"\n⏰ [TIMEOUT] Training exceeded {timeout}s, terminating...")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    break
            
            # 读取剩余输出
            try:
                remaining_stdout, remaining_stderr = process.communicate(timeout=30)
                if remaining_stdout:
                    output_lines.extend(remaining_stdout.splitlines())
                if remaining_stderr:
                    error_lines.extend(remaining_stderr.splitlines())
            except subprocess.TimeoutExpired:
                print("⚠️ [WARNING] Timeout while reading final output")
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # 检查训练是否成功
            return_code = process.returncode
            if return_code == 0:
                print(f"\n✅ [SUCCESS] {config['name']} (seed={seed}) completed in {training_time:.1f}s")
                print(f"🎯 Final iteration: {current_iteration}")
                if latest_metrics:
                    print(f"📈 Final metrics: {latest_metrics}")
                success = True
                error_msg = None
            else:
                print(f"\n❌ [ERROR] {config['name']} (seed={seed}) failed (exit code: {return_code})")
                if error_lines:
                    print(f"stderr: {''.join(error_lines[-10:])}")
                success = False
                error_msg = '\n'.join(error_lines)
            
        except Exception as e:
            print(f"\n💥 [EXCEPTION] {config['name']} (seed={seed}) failed: {e}")
            success = False
            error_msg = str(e)
            training_time = time.time() - start_time
            output_lines = []
            error_lines = [str(e)]
        
        # 训练后清理
        print("🧹 [POST-TRAINING] Cleaning up processes...")
        self.cleanup_isaac_processes()
        
        return {
            'config_name': config_name,
            'seed': seed,
            'success': success,
            'training_time': training_time,
            'error_msg': error_msg,
            'stdout': '\n'.join(output_lines),
            'stderr': '\n'.join(error_lines),
            'final_iteration': current_iteration,
            'final_metrics': latest_metrics
        }
    
    def parse_training_logs(self, config_name: str, seed: int) -> Dict:
        """解析训练日志获取性能指标"""
        config = self.configs[config_name]
        
        # 查找对应的日志目录
        if config_name == 'with_sensors':
            log_pattern = f"logs/rsl_rl/cabinet_with_sensors_new/*/seed{seed}"
        else:
            log_pattern = f"logs/rsl_rl/cabinet_baseline/*/seed{seed}"
        
        log_dirs = glob.glob(log_pattern)
        if not log_dirs:
            print(f"⚠️ [WARNING] No log directory found for {config['name']} seed={seed}")
            return {}
        
        log_dir = Path(log_dirs[-1])  # 使用最新的日志目录
        
        # 解析tensorboard日志或其他日志文件
        # 这里可以根据实际的日志格式进行解析
        parsed_data = {
            'final_reward': None,
            'convergence_iteration': None,
            'success_rate': None,
            'episode_length': None
        }
        
        # 尝试解析summaries.json文件（如果存在）
        summaries_file = log_dir / "summaries.json"
        if summaries_file.exists():
            try:
                with open(summaries_file, 'r') as f:
                    summaries = json.load(f)
                    # 提取最终性能指标
                    if 'Episode_Reward' in summaries:
                        parsed_data['final_reward'] = summaries['Episode_Reward'][-1] if summaries['Episode_Reward'] else None
            except Exception as e:
                print(f"📄 [LOG] Error parsing summaries.json: {e}")
        
        return parsed_data
    
    def run_all_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 [EXPERIMENT] Starting full comparison experiment")
        
        total_runs = len(self.configs) * self.args.num_seeds
        current_run = 0
        
        for config_name in self.configs.keys():
            self.results[config_name] = []
            
            for seed in range(self.args.num_seeds):
                current_run += 1
                print(f"\n📊 [PROGRESS] Run {current_run}/{total_runs}")
                
                # 运行训练
                result = self.run_single_training(config_name, seed)
                
                # 解析日志
                if result['success']:
                    log_data = self.parse_training_logs(config_name, seed)
                    result.update(log_data)
                
                # 保存结果
                self.results[config_name].append(result)
                
                # 保存中间结果
                self.save_intermediate_results()
                
                # 训练间隔 - 更长的等待时间和彻底清理
                if current_run < total_runs:
                    print("🧹 [INTERVAL] Performing thorough cleanup between training runs...")
                    
                    # 强制清理所有可能的Isaac进程
                    self.cleanup_isaac_processes()
                    
                    # 额外等待，让GPU内存完全释放
                    half_wait = self.args.cleanup_wait // 2
                    print(f"⏱️ [INTERVAL] Waiting {half_wait}s for complete resource cleanup...")
                    time.sleep(half_wait)
                    
                    # 再次检查和清理
                    self.cleanup_isaac_processes()
                    self.wait_for_system_ready()
                    
                    print(f"⏱️ [INTERVAL] Final {half_wait}s wait before next training...")
                    time.sleep(half_wait)
        
        print(f"\n🎉 [COMPLETE] All experiments completed!")
    
    def save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.experiment_dir / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def analyze_results(self):
        """分析实验结果"""
        print(f"\n📈 [ANALYSIS] Analyzing experiment results...")
        
        # 创建DataFrame便于分析
        all_data = []
        for config_name, results in self.results.items():
            for result in results:
                row = {
                    'config': config_name,
                    'config_name': self.configs[config_name]['name'],
                    'seed': result['seed'],
                    'success': result['success'],
                    'training_time': result['training_time'],
                    'final_reward': result.get('final_reward'),
                    'convergence_iteration': result.get('convergence_iteration'),
                    'success_rate': result.get('success_rate'),
                    'episode_length': result.get('episode_length')
                }
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        # 计算统计指标
        stats = {}
        for config_name in self.configs.keys():
            config_data = df[df['config'] == config_name]
            successful_runs = config_data[config_data['success'] == True]
            
            stats[config_name] = {
                'total_runs': len(config_data),
                'successful_runs': len(successful_runs),
                'success_rate': len(successful_runs) / len(config_data),
                'avg_training_time': successful_runs['training_time'].mean() if len(successful_runs) > 0 else None,
                'std_training_time': successful_runs['training_time'].std() if len(successful_runs) > 0 else None,
                'avg_final_reward': successful_runs['final_reward'].mean() if len(successful_runs) > 0 else None,
                'std_final_reward': successful_runs['final_reward'].std() if len(successful_runs) > 0 else None
            }
        
        # 保存分析结果
        analysis_file = self.experiment_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        df.to_csv(self.experiment_dir / "experiment_data.csv", index=False)
        
        return df, stats
    
    def generate_visualizations(self, df: pd.DataFrame, stats: Dict):
        """生成可视化图表"""
        print(f"📊 [VISUALIZATION] Generating comparison plots...")
        
        # 设置绘图风格 - 兼容不同版本的seaborn
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
                print("⚠️ [WARNING] Using default matplotlib style (seaborn not available)")
        
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sensor vs Baseline Comparison', fontsize=16, fontweight='bold')
        
        # 1. 训练时间对比
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='config_name', y='training_time', ax=axes[0,0])
            axes[0,0].set_title('Training Time Comparison')
            axes[0,0].set_ylabel('Training Time (seconds)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 成功率对比
        success_rates = []
        config_names = []
        for config_name, stat in stats.items():
            success_rates.append(stat['success_rate'] * 100)
            config_names.append(self.configs[config_name]['name'])
        
        bars = axes[0,1].bar(config_names, success_rates, 
                            color=[self.configs[k]['color'] for k in stats.keys()])
        axes[0,1].set_title('Training Success Rate')
        axes[0,1].set_ylabel('Success Rate (%)')
        axes[0,1].set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 最终奖励对比（如果有数据）
        reward_data = successful_df.dropna(subset=['final_reward'])
        if len(reward_data) > 0:
            sns.boxplot(data=reward_data, x='config_name', y='final_reward', ax=axes[1,0])
            axes[1,0].set_title('Final Reward Comparison')
            axes[1,0].set_ylabel('Final Reward')
            axes[1,0].tick_params(axis='x', rotation=45)
        else:
            axes[1,0].text(0.5, 0.5, 'No reward data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Final Reward Comparison')
        
        # 4. 统计摘要表
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        
        # 创建统计表格
        table_data = []
        headers = ['Metric', 'With Sensors', 'Baseline']
        
        # 成功率
        table_data.append([
            'Success Rate (%)',
            f"{stats['with_sensors']['success_rate']*100:.1f}",
            f"{stats['baseline']['success_rate']*100:.1f}"
        ])
        
        # 平均训练时间
        sensor_time = stats['with_sensors']['avg_training_time']
        baseline_time = stats['baseline']['avg_training_time']
        if sensor_time and baseline_time:
            table_data.append([
                'Avg Training Time (s)',
                f"{sensor_time:.1f} ± {stats['with_sensors']['std_training_time']:.1f}",
                f"{baseline_time:.1f} ± {stats['baseline']['std_training_time']:.1f}"
            ])
            
            # 计算相对差异
            time_diff = ((sensor_time - baseline_time) / baseline_time) * 100
            table_data.append([
                'Time Difference (%)',
                f"{time_diff:+.1f}",
                "baseline"
            ])
        
        table = axes[1,1].table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,1].set_title('Statistical Summary')
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = self.experiment_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 [SAVE] Plots saved to {plot_file}")
        
        plt.show()
    
    def generate_report(self, stats: Dict):
        """生成详细的实验报告"""
        print(f"📋 [REPORT] Generating experiment report...")
        
        report_file = self.experiment_dir / "experiment_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# 传感器对比实验报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**实验目录**: `{self.experiment_dir}`\n\n")
            
            f.write(f"## 实验配置\n\n")
            f.write(f"- **随机种子数量**: {self.args.num_seeds}\n")
            f.write(f"- **最大迭代次数**: {self.args.max_iterations}\n")
            f.write(f"- **环境数量**: {self.args.num_envs}\n")
            f.write(f"- **对比版本**:\n")
            for config_name, config in self.configs.items():
                f.write(f"  - {config['name']}: `{config['script']}`\n")
            
            f.write(f"\n## 实验结果\n\n")
            
            for config_name, stat in stats.items():
                config = self.configs[config_name]
                f.write(f"### {config['name']}\n\n")
                f.write(f"- **总运行次数**: {stat['total_runs']}\n")
                f.write(f"- **成功运行次数**: {stat['successful_runs']}\n")
                f.write(f"- **成功率**: {stat['success_rate']*100:.1f}%\n")
                
                if stat['avg_training_time']:
                    f.write(f"- **平均训练时间**: {stat['avg_training_time']:.1f} ± {stat['std_training_time']:.1f} 秒\n")
                
                if stat['avg_final_reward']:
                    f.write(f"- **平均最终奖励**: {stat['avg_final_reward']:.3f} ± {stat['std_final_reward']:.3f}\n")
                
                f.write(f"\n")
            
            # 添加结论
            f.write(f"## 结论\n\n")
            
            sensor_stats = stats['with_sensors']
            baseline_stats = stats['baseline']
            
            if sensor_stats['avg_training_time'] and baseline_stats['avg_training_time']:
                time_diff = ((sensor_stats['avg_training_time'] - baseline_stats['avg_training_time']) 
                           / baseline_stats['avg_training_time']) * 100
                
                if time_diff > 5:
                    f.write(f"- 🐌 传感器版本的训练时间比基线版本长 {time_diff:.1f}%，这可能是由于额外的传感器数据处理开销\n")
                elif time_diff < -5:
                    f.write(f"- 🚀 传感器版本的训练时间比基线版本短 {abs(time_diff):.1f}%，传感器信息可能有助于更快收敛\n")
                else:
                    f.write(f"- ⚖️ 两个版本的训练时间基本相当（差异 {time_diff:+.1f}%）\n")
            
            success_diff = (sensor_stats['success_rate'] - baseline_stats['success_rate']) * 100
            if success_diff > 10:
                f.write(f"- ✅ 传感器版本的成功率明显更高（+{success_diff:.1f}%），传感器信息显著提升了训练稳定性\n")
            elif success_diff < -10:
                f.write(f"- ❌ 传感器版本的成功率较低（{success_diff:.1f}%），可能存在配置问题\n")
            else:
                f.write(f"- 📊 两个版本的成功率相当（差异 {success_diff:+.1f}%）\n")
        
        print(f"📄 [SAVE] Report saved to {report_file}")
    
    def run_complete_experiment(self):
        """运行完整的对比实验"""
        try:
            # 运行所有实验
            self.run_all_experiments()
            
            # 分析结果
            df, stats = self.analyze_results()
            
            # 生成可视化
            self.generate_visualizations(df, stats)
            
            # 生成报告
            self.generate_report(stats)
            
            print(f"\n🎉 [COMPLETE] Experiment completed successfully!")
            print(f"📁 Results saved in: {self.experiment_dir}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ [INTERRUPTED] Experiment interrupted by user")
            print(f"📁 Partial results saved in: {self.experiment_dir}")
        
        except Exception as e:
            print(f"\n💥 [ERROR] Experiment failed: {e}")
            print(f"📁 Partial results saved in: {self.experiment_dir}")
            raise


def main():
    parser = argparse.ArgumentParser(description="传感器对比实验")
    parser.add_argument("--num_seeds", type=int, default=3, help="随机种子数量")
    parser.add_argument("--max_iterations", type=int, default=1000, help="最大训练迭代次数")
    parser.add_argument("--num_envs", type=int, default=32, help="环境数量")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用fabric")
    parser.add_argument("--timeout", type=int, default=7200, help="单个训练的超时时间(秒)")
    parser.add_argument("--cleanup_wait", type=int, default=60, help="训练间隔的清理等待时间(秒)")
    parser.add_argument("--disable_cleanup", action="store_true", help="禁用自动进程清理(不推荐)")
    parser.add_argument("--gpu_memory_threshold", type=int, default=80, help="GPU内存使用率阈值百分比，超过此值将等待")
    parser.add_argument("--status_interval", type=int, default=5, help="状态显示间隔(每N次迭代显示一次)")
    parser.add_argument("--disable_realtime_monitoring", action="store_true", help="禁用实时监控，使用传统模式")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.status_interval <= 0:
        print("❌ [ERROR] Status interval must be positive")
        exit(1)
    
    # 如果禁用清理，给出警告
    if args.disable_cleanup:
        print("⚠️ [WARNING] Process cleanup is disabled!")
        print("⚠️ This may cause resource conflicts between training runs")
        user_confirm = input("🤔 Are you sure you want to continue? (y/n): ").lower().strip()
        if user_confirm not in ['y', 'yes']:
            print("❌ Experiment cancelled")
            exit(0)
    
    # 如果禁用实时监控，给出提示
    if args.disable_realtime_monitoring:
        print("📊 [INFO] Real-time monitoring disabled, using traditional mode")
    else:
        print(f"📊 [INFO] Real-time monitoring enabled (status every {args.status_interval} iterations)")
    
    # 创建并运行实验
    experiment = SensorComparisonExperiment(args)
    experiment.run_complete_experiment()


if __name__ == "__main__":
    main() 