#!/usr/bin/env python3
"""
🚀 快速传感器对比测试

在运行完整实验前，先用少量迭代测试两个版本是否能正常工作。

Usage:
    python quick_sensor_test.py
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime


def run_quick_test():
    """运行快速测试"""
    print("🧪 [QUICK TEST] Starting sensor comparison quick test")
    print("⚡ This will run both versions with minimal iterations to verify functionality\n")
    
    configs = {
        'sensor': {
            'script': 'cabinet_rl_with_sensors_new.py',
            'name': 'With Sensors'
        },
        'baseline': {
            'script': 'cabinet_rl_BASELINE.py',
            'name': 'Baseline'
        }
    }
    
    # 测试参数
    test_params = {
        'num_envs': 4,
        'max_iterations': 5,  # 仅5次迭代用于快速测试
        'seed': 42
    }
    
    results = {}
    
    for config_key, config in configs.items():
        print(f"🚀 [TEST] Running {config['name']}...")
        
        # 构建命令
        cmd = [
            "./isaaclab.sh", "-p", config['script'],
            "--num_envs", str(test_params['num_envs']),
            "--max_iterations", str(test_params['max_iterations']),
            "--seed", str(test_params['seed']),
            "--headless"
        ]
        
        print(f"💻 Command: {' '.join(cmd)}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行训练
            result = subprocess.run(
                cmd,
                cwd="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab",
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ [SUCCESS] {config['name']} completed in {duration:.1f}s")
                
                # 解析输出获取观测空间信息
                obs_space = None
                for line in result.stdout.split('\n'):
                    if 'Environment observation space:' in line:
                        obs_space = int(line.split(':')[-1].strip())
                        break
                
                results[config_key] = {
                    'success': True,
                    'duration': duration,
                    'obs_space': obs_space,
                    'stdout_lines': len(result.stdout.split('\n')),
                    'stderr_lines': len(result.stderr.split('\n'))
                }
                
                print(f"  📊 Observation space: {obs_space} dimensions")
                print(f"  ⏱️ Duration: {duration:.1f} seconds")
                
            else:
                print(f"❌ [FAILED] {config['name']} failed!")
                print(f"  🔍 Return code: {result.returncode}")
                print(f"  📝 Last stderr lines:")
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines[-5:]:
                    if line.strip():
                        print(f"    {line}")
                
                results[config_key] = {
                    'success': False,
                    'duration': duration,
                    'error': result.stderr[-500:],  # 最后500字符
                    'return_code': result.returncode
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏰ [TIMEOUT] {config['name']} timed out after 10 minutes")
            results[config_key] = {
                'success': False,
                'duration': 600,
                'error': "Training timed out"
            }
        
        except Exception as e:
            print(f"💥 [EXCEPTION] {config['name']} failed: {e}")
            results[config_key] = {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
        
        print()  # 空行分隔
    
    # 生成测试报告
    print("📊 [REPORT] Quick test results:")
    print("=" * 60)
    
    for config_key, result in results.items():
        config = configs[config_key]
        print(f"\n🔧 {config['name']}:")
        
        if result['success']:
            print(f"  ✅ Status: SUCCESS")
            print(f"  ⏱️ Duration: {result['duration']:.1f} seconds")
            print(f"  📊 Observation space: {result['obs_space']} dimensions")
            
            # 验证观测空间
            expected_dims = {'sensor': 43, 'baseline': 31}
            if result['obs_space'] == expected_dims[config_key]:
                print(f"  ✅ Observation space: CORRECT")
            else:
                print(f"  ⚠️ Observation space: UNEXPECTED (expected {expected_dims[config_key]})")
        else:
            print(f"  ❌ Status: FAILED")
            print(f"  ⏱️ Duration: {result['duration']:.1f} seconds")
            print(f"  📝 Error: {result.get('error', 'Unknown error')}")
    
    # 对比分析
    if results['sensor']['success'] and results['baseline']['success']:
        print(f"\n🎯 [COMPARISON] Performance comparison:")
        sensor_time = results['sensor']['duration']
        baseline_time = results['baseline']['duration']
        time_diff = ((sensor_time - baseline_time) / baseline_time) * 100
        
        print(f"  📈 Time difference: {time_diff:+.1f}%")
        if abs(time_diff) < 10:
            print(f"  ✅ Similar performance")
        elif time_diff > 0:
            print(f"  🐌 Sensor version is slower")
        else:
            print(f"  🚀 Sensor version is faster")
        
        print(f"\n🎉 [READY] Both versions work correctly!")
        print(f"✨ You can now run the full experiment with:")
        print(f"   python run_sensor_comparison_experiment.py --num_seeds 3 --max_iterations 2000")
    
    else:
        print(f"\n⚠️ [WARNING] Some tests failed. Please fix issues before running full experiment.")
        
        # 建议修复措施
        if not results['sensor']['success']:
            print(f"  🔧 Check sensor version configuration")
        if not results['baseline']['success']:
            print(f"  🔧 Check baseline version configuration")
    
    # 保存测试结果
    test_dir = Path("experiments/quick_tests")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(test_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 [SAVE] Test results saved to: {test_file}")


if __name__ == "__main__":
    run_quick_test() 