# 使用说明

## 快速开始

### 1. 环境准备

确保已经安装了IsaacLab和项目依赖：

```bash
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project
pip install -r requirements.txt
pip install -e .
```

### 2. 基本训练

#### 训练传感器版本
```bash
python scripts/run_training.py sensor --num_envs 64 --max_iterations 1000 --headless
```

#### 训练基线版本
```bash
python scripts/run_training.py baseline --num_envs 64 --max_iterations 1000 --headless
```

#### 使用原始脚本（如果需要）
```bash
# 传感器版本
python scripts/train_sensors.py --num_envs 64 --max_iterations 1000 --headless

# 基线版本
python scripts/train_baseline.py --num_envs 64 --max_iterations 1000 --headless
```

### 3. 对比实验

运行完整的对比实验：
```bash
python scripts/run_comparison.py --num_seeds 3 --max_iterations 2000 --num_envs 64
```

### 4. 结果分析

分析实验结果：
```bash
python scripts/analyze_results.py --experiment_dir data/results/latest
```

### 5. 演示和测试

运行传感器功能演示：
```bash
python scripts/demo/sensor_demo.py
```

运行系统诊断：
```bash
python scripts/demo/diagnostic.py
```

## 配置文件

### 项目配置
- `config/project_config.yaml` - 项目整体配置
- `config/base_config.yaml` - 基础配置
- `config/sensor_config.yaml` - 传感器版本配置
- `config/baseline_config.yaml` - 基线版本配置

### 修改配置

1. **修改环境数量**:
   ```yaml
   # config/base_config.yaml
   env:
     num_envs: 128  # 修改为你想要的数量
   ```

2. **修改训练参数**:
   ```yaml
   # config/base_config.yaml
   training:
     max_iterations: 2000
     ppo:
       learning_rate: 1e-4
   ```

3. **修改IsaacLab路径**:
   ```yaml
   # config/project_config.yaml
   paths:
     isaaclab_root: "/your/isaaclab/path"
   ```

## 常见问题

### Q1: 训练失败，提示找不到模块

**解决方案**: 确保已经安装项目依赖并设置了正确的Python路径：
```bash
pip install -e .
export PYTHONPATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab:$PYTHONPATH"
```

### Q2: GPU内存不足

**解决方案**: 
1. 减少环境数量：`--num_envs 32`
2. 在配置文件中设置内存限制
3. 使用进程清理脚本

### Q3: Isaac Sim进程残留

**解决方案**: 
```bash
# 使用内置的清理功能
python scripts/run_comparison.py --cleanup_processes

# 或者手动清理
pkill -f isaac
pkill -f omni
```

### Q4: 训练中断后如何恢复

**解决方案**:
1. 检查 `logs/training/` 目录找到最新的检查点
2. 使用 `--resume` 参数继续训练（如果脚本支持）
3. 或者从保存的模型重新开始

## 高级使用

### 自定义实验

1. 创建新的配置文件：
   ```bash
   cp config/sensor_config.yaml config/my_experiment.yaml
   ```

2. 修改配置参数

3. 运行实验：
   ```bash
   python scripts/run_training.py sensor --config config/my_experiment.yaml
   ```

### 批量实验

使用shell脚本运行多个实验：
```bash
#!/bin/bash
for seed in 42 43 44 45 46; do
    python scripts/run_training.py sensor --seed $seed --num_envs 64
    python scripts/run_training.py baseline --seed $seed --num_envs 64
done
```

### 结果可视化

查看训练曲线：
```bash
# 如果有tensorboard日志
tensorboard --logdir logs/training/

# 或者使用项目的可视化工具
python scripts/visualize_results.py --experiment_dir data/results/your_experiment
```

## 项目维护

### 清理IsaacLab目录

迁移完成后，清理原始IsaacLab目录：
```bash
# 预览要删除的文件
python scripts/cleanup_isaaclab.py --dry-run

# 实际删除（请谨慎）
python scripts/cleanup_isaaclab.py
```

### 更新依赖

```bash
pip install -r requirements.txt --upgrade
```

### 运行测试

```bash
python -m pytest tests/
```

## 性能优化

### 提高训练速度

1. **增加环境数量**:
   ```bash
   python scripts/run_training.py sensor --num_envs 128
   ```

2. **使用多GPU**（如果支持）:
   ```bash
   python scripts/run_training.py sensor --gpu_ids 0,1
   ```

3. **调整渲染设置**:
   ```yaml
   # config/base_config.yaml
   visualization:
     enable_rendering: false
   ```

### 内存优化

1. **减少环境数量**
2. **启用内存清理**:
   ```yaml
   # config/project_config.yaml
   system:
     process_management:
       enable_cleanup: true
   ```

3. **使用内存限制**:
   ```yaml
   # config/base_config.yaml
   system:
     memory_limit: 0.8
   ```

## 故障排除

### 日志文件位置

- 训练日志: `logs/training/`
- 实验日志: `logs/experiments/`
- 调试日志: `logs/debug/`

### 常用调试命令

```bash
# 检查GPU使用情况
nvidia-smi

# 检查进程
ps aux | grep isaac

# 检查磁盘空间
df -h

# 检查内存使用
free -h
```

### 获取帮助

```bash
# 查看脚本帮助
python scripts/run_training.py --help
python scripts/run_comparison.py --help

# 查看配置文件模板
cat config/base_config.yaml
```
