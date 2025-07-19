#!/usr/bin/env python3
"""
分析传感器对比实验结果

分析sensor_comparison_20250709_234139实验的结果，提取关键指标：
1. 两个版本的成功率对比
2. 学习曲线和奖励变化
3. 第一次学会开抽屉的时间
4. 详细的性能分析
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from pathlib import Path

class ExperimentAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.results = None
        self.training_data = {}
        
    def load_results(self):
        """加载实验结果"""
        print(f"📂 [LOAD] Loading experiment results from {self.results_file}")
        
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"✅ [SUCCESS] Results loaded successfully")
            
            # 打印基本信息
            print(f"📊 [INFO] Available configurations: {list(self.results.keys())}")
            for config_name, results in self.results.items():
                print(f"  - {config_name}: {len(results)} runs")
                
        except Exception as e:
            print(f"❌ [ERROR] Failed to load results: {e}")
            return False
            
        return True
    
    def parse_training_logs(self):
        """解析训练日志，提取学习曲线数据"""
        print("📊 [PARSE] Parsing training logs...")
        
        for config_name, results in self.results.items():
            self.training_data[config_name] = []
            
            for run_idx, result in enumerate(results):
                print(f"🔍 [PARSE] Processing {config_name} run {run_idx + 1}")
                
                # 解析训练输出日志
                training_log = result.get('stdout', '')
                
                # 提取训练指标
                iterations = []
                rewards = []
                losses = []
                times = []
                
                # 正则表达式模式来匹配训练输出
                iteration_pattern = r'Learning iteration (\d+)/\d+'
                reward_pattern = r'Mean reward: ([\d.-]+)'
                loss_pattern = r'Mean entropy loss: ([-+]?\d*\.?\d+)'
                time_pattern = r'Time elapsed: (\d{2}:\d{2}:\d{2})'
                
                lines = training_log.split('\n')
                current_iteration = None
                
                for line in lines:
                    # 查找迭代数
                    iter_match = re.search(iteration_pattern, line)
                    if iter_match:
                        current_iteration = int(iter_match.group(1))
                    
                    # 查找奖励
                    reward_match = re.search(reward_pattern, line)
                    if reward_match and current_iteration:
                        reward = float(reward_match.group(1))
                        iterations.append(current_iteration)
                        rewards.append(reward)
                    
                    # 查找损失
                    loss_match = re.search(loss_pattern, line)
                    if loss_match and current_iteration and len(losses) < len(rewards):
                        loss = float(loss_match.group(1))
                        losses.append(loss)
                
                # 存储解析的数据
                run_data = {
                    'seed': result.get('seed', run_idx),
                    'success': result.get('success', False),
                    'training_time': result.get('training_time', 0),
                    'iterations': iterations,
                    'rewards': rewards,
                    'losses': losses,
                    'final_iteration': result.get('final_iteration', 0),
                    'final_metrics': result.get('final_metrics', {})
                }
                
                self.training_data[config_name].append(run_data)
                
                print(f"  ✅ Found {len(iterations)} training points for seed {run_data['seed']}")
    
    def analyze_success_rates(self):
        """分析成功率"""
        print("📈 [ANALYSIS] Analyzing success rates...")
        
        success_analysis = {}
        
        for config_name, results in self.results.items():
            total_runs = len(results)
            successful_runs = sum(1 for r in results if r.get('success', False))
            success_rate = successful_runs / total_runs if total_runs > 0 else 0
            
            success_analysis[config_name] = {
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': success_rate,
                'success_percentage': success_rate * 100
            }
            
            print(f"🎯 [RESULT] {config_name}:")
            print(f"   总运行次数: {total_runs}")
            print(f"   成功次数: {successful_runs}")
            print(f"   成功率: {success_rate:.1%}")
        
        return success_analysis
    
    def analyze_learning_performance(self):
        """分析学习性能"""
        print("📚 [ANALYSIS] Analyzing learning performance...")
        
        performance_analysis = {}
        
        for config_name, runs in self.training_data.items():
            successful_runs = [run for run in runs if run['success']]
            
            if not successful_runs:
                print(f"⚠️ [WARNING] No successful runs for {config_name}")
                continue
            
            # 计算平均学习曲线
            all_rewards = []
            all_iterations = []
            
            for run in successful_runs:
                if run['rewards']:
                    all_rewards.extend(run['rewards'])
                    all_iterations.extend(run['iterations'])
            
            if all_rewards:
                # 计算首次达到良好性能的时间
                good_performance_threshold = 80  # 奖励阈值
                first_success_iterations = []
                
                for run in successful_runs:
                    rewards = run['rewards']
                    iterations = run['iterations']
                    
                    for i, reward in enumerate(rewards):
                        if reward >= good_performance_threshold:
                            first_success_iterations.append(iterations[i])
                            break
                
                avg_first_success = np.mean(first_success_iterations) if first_success_iterations else None
                
                performance_analysis[config_name] = {
                    'avg_final_reward': np.mean([run['rewards'][-1] for run in successful_runs if run['rewards']]),
                    'std_final_reward': np.std([run['rewards'][-1] for run in successful_runs if run['rewards']]),
                    'avg_training_time': np.mean([run['training_time'] for run in successful_runs]),
                    'std_training_time': np.std([run['training_time'] for run in successful_runs]),
                    'avg_first_success_iteration': avg_first_success,
                    'num_successful_learning': len(first_success_iterations),
                    'all_rewards': all_rewards,
                    'all_iterations': all_iterations
                }
                
                print(f"🚀 [RESULT] {config_name}:")
                print(f"   平均最终奖励: {performance_analysis[config_name]['avg_final_reward']:.2f} ± {performance_analysis[config_name]['std_final_reward']:.2f}")
                print(f"   平均训练时间: {performance_analysis[config_name]['avg_training_time']:.1f} ± {performance_analysis[config_name]['std_training_time']:.1f}秒")
                if avg_first_success:
                    print(f"   首次学会时间: 平均第{avg_first_success:.0f}次迭代")
                print(f"   成功学会次数: {len(first_success_iterations)}/{len(successful_runs)}")
        
        return performance_analysis
    
    def create_visualizations(self, success_analysis, performance_analysis):
        """创建可视化图表"""
        print("📊 [VISUALIZATION] Creating comparison plots...")
        
        # 设置matplotlib字体以支持中文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sensor vs Baseline Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. 成功率对比
        config_names = list(success_analysis.keys())
        success_rates = [success_analysis[name]['success_percentage'] for name in config_names]
        
        colors = ['#2E86AB', '#A23B72']  # 蓝色for传感器, 紫色for基线
        bars = axes[0,0].bar(config_names, success_rates, color=colors[:len(config_names)])
        axes[0,0].set_title('Training Success Rate Comparison')
        axes[0,0].set_ylabel('Success Rate (%)')
        axes[0,0].set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 最终奖励对比
        if len(performance_analysis) >= 2:
            config_names_perf = list(performance_analysis.keys())
            final_rewards = [performance_analysis[name]['avg_final_reward'] for name in config_names_perf]
            final_rewards_std = [performance_analysis[name]['std_final_reward'] for name in config_names_perf]
            
            bars2 = axes[0,1].bar(config_names_perf, final_rewards, 
                                 yerr=final_rewards_std, capsize=5, color=colors[:len(config_names_perf)])
            axes[0,1].set_title('Final Reward Comparison')
            axes[0,1].set_ylabel('Mean Final Reward')
            
            # 添加数值标签
            for bar, reward, std in zip(bars2, final_rewards, final_rewards_std):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                              f'{reward:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 学习曲线
        if performance_analysis:
            for i, (config_name, data) in enumerate(performance_analysis.items()):
                if data['all_rewards'] and data['all_iterations']:
                    # 创建平滑的学习曲线
                    iterations = np.array(data['all_iterations'])
                    rewards = np.array(data['all_rewards'])
                    
                    # 按迭代次数排序
                    sorted_indices = np.argsort(iterations)
                    iterations_sorted = iterations[sorted_indices]
                    rewards_sorted = rewards[sorted_indices]
                    
                    # 使用滑动平均平滑曲线
                    window_size = max(1, len(rewards_sorted) // 50)
                    if len(rewards_sorted) > window_size:
                        smoothed_rewards = pd.Series(rewards_sorted).rolling(window=window_size, min_periods=1).mean()
                        axes[1,0].plot(iterations_sorted, smoothed_rewards, 
                                      label=config_name, color=colors[i], linewidth=2)
            
            axes[1,0].set_title('Learning Curves (Smoothed)')
            axes[1,0].set_xlabel('Training Iteration')
            axes[1,0].set_ylabel('Mean Reward')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 学习速度对比
        if len(performance_analysis) >= 2:
            config_names_speed = list(performance_analysis.keys())
            first_success_iters = [performance_analysis[name]['avg_first_success_iteration'] 
                                  for name in config_names_speed 
                                  if performance_analysis[name]['avg_first_success_iteration'] is not None]
            valid_names = [name for name in config_names_speed 
                          if performance_analysis[name]['avg_first_success_iteration'] is not None]
            
            if first_success_iters:
                bars3 = axes[1,1].bar(valid_names, first_success_iters, color=colors[:len(valid_names)])
                axes[1,1].set_title('Time to First Success')
                axes[1,1].set_ylabel('Iterations to Reach Good Performance')
                
                # 添加数值标签
                for bar, iters in zip(bars3, first_success_iters):
                    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                                  f'{iters:.0f}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1,1].text(0.5, 0.5, 'No successful learning data', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Time to First Success')
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path("experiment_analysis")
        output_dir.mkdir(exist_ok=True)
        
        plot_file = output_dir / "sensor_comparison_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 [SAVE] Analysis plots saved to {plot_file}")
        
        plt.show()
    
    def generate_report(self, success_analysis, performance_analysis):
        """生成详细分析报告"""
        print("📋 [REPORT] Generating analysis report...")
        
        output_dir = Path("experiment_analysis")
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / "sensor_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 传感器对比实验分析报告\n\n")
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**实验数据**: sensor_comparison_20250709_234139\n\n")
            
            # 成功率分析
            f.write("## 🎯 成功率分析\n\n")
            for config_name, data in success_analysis.items():
                f.write(f"### {config_name}\n")
                f.write(f"- **总运行次数**: {data['total_runs']}\n")
                f.write(f"- **成功运行次数**: {data['successful_runs']}\n")
                f.write(f"- **成功率**: {data['success_percentage']:.1f}%\n\n")
            
            # 性能分析
            f.write("## 📈 性能分析\n\n")
            for config_name, data in performance_analysis.items():
                f.write(f"### {config_name}\n")
                f.write(f"- **平均最终奖励**: {data['avg_final_reward']:.2f} ± {data['std_final_reward']:.2f}\n")
                f.write(f"- **平均训练时间**: {data['avg_training_time']:.1f} ± {data['std_training_time']:.1f} 秒\n")
                if data['avg_first_success_iteration']:
                    f.write(f"- **首次学会时间**: 平均第 {data['avg_first_success_iteration']:.0f} 次迭代\n")
                f.write(f"- **成功学会率**: {data['num_successful_learning']}/{len(self.training_data[config_name])} ({data['num_successful_learning']/len(self.training_data[config_name])*100:.1f}%)\n\n")
            
            # 对比结论
            f.write("## 🔍 对比结论\n\n")
            if len(success_analysis) >= 2:
                configs = list(success_analysis.keys())
                sensor_config = next((c for c in configs if 'sensor' in c.lower()), configs[0])
                baseline_config = next((c for c in configs if 'baseline' in c.lower()), configs[1] if len(configs) > 1 else configs[0])
                
                if sensor_config in success_analysis and baseline_config in success_analysis:
                    sensor_success = success_analysis[sensor_config]['success_percentage']
                    baseline_success = success_analysis[baseline_config]['success_percentage']
                    
                    f.write(f"### 成功率对比\n")
                    f.write(f"- **传感器版本**: {sensor_success:.1f}%\n")
                    f.write(f"- **基线版本**: {baseline_success:.1f}%\n")
                    f.write(f"- **差异**: {sensor_success - baseline_success:+.1f}%\n\n")
                    
                    if sensor_config in performance_analysis and baseline_config in performance_analysis:
                        sensor_perf = performance_analysis[sensor_config]
                        baseline_perf = performance_analysis[baseline_config]
                        
                        f.write(f"### 性能对比\n")
                        f.write(f"- **传感器版本最终奖励**: {sensor_perf['avg_final_reward']:.2f}\n")
                        f.write(f"- **基线版本最终奖励**: {baseline_perf['avg_final_reward']:.2f}\n")
                        f.write(f"- **奖励差异**: {sensor_perf['avg_final_reward'] - baseline_perf['avg_final_reward']:+.2f}\n\n")
                        
                        if sensor_perf['avg_first_success_iteration'] and baseline_perf['avg_first_success_iteration']:
                            f.write(f"### 学习速度对比\n")
                            f.write(f"- **传感器版本首次学会**: 第 {sensor_perf['avg_first_success_iteration']:.0f} 次迭代\n")
                            f.write(f"- **基线版本首次学会**: 第 {baseline_perf['avg_first_success_iteration']:.0f} 次迭代\n")
                            f.write(f"- **学习速度差异**: {baseline_perf['avg_first_success_iteration'] - sensor_perf['avg_first_success_iteration']:+.0f} 迭代\n\n")
            
            # 总结
            f.write("## 📝 总结\n\n")
            f.write("基于以上分析，可以得出以下结论：\n\n")
            f.write("1. **稳定性**: 比较两个版本的训练成功率，评估哪个版本更稳定\n")
            f.write("2. **性能**: 比较最终奖励值，评估哪个版本能达到更好的性能\n")
            f.write("3. **学习效率**: 比较首次学会的时间，评估哪个版本学习更快\n")
            f.write("4. **训练时间**: 比较总训练时间，评估计算效率\n\n")
            
        print(f"📄 [SAVE] Analysis report saved to {report_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        if not self.load_results():
            return
        
        self.parse_training_logs()
        success_analysis = self.analyze_success_rates()
        performance_analysis = self.analyze_learning_performance()
        
        self.create_visualizations(success_analysis, performance_analysis)
        self.generate_report(success_analysis, performance_analysis)
        
        print("\n🎉 [COMPLETE] Analysis completed successfully!")
        print(f"📁 Results saved in experiment_analysis/ directory")

def main():
    # 实验结果文件路径
    results_file = "experiments/sensor_comparison_20250709_234139/intermediate_results.json"
    
    # 创建分析器并运行分析
    analyzer = ExperimentAnalyzer(results_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 