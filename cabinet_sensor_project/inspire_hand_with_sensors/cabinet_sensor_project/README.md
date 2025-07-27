# Cabinet Sensor Project

一个基于IsaacLab的机器人抓取控制项目，比较传感器增强版本与基线版本的性能差异。

## 项目结构

```
cabinet_sensor_project/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包
├── setup.py                           # 项目安装配置
├── config/                            # 配置文件目录
│   ├── base_config.yaml              # 基础配置
│   ├── sensor_config.yaml            # 传感器版本配置
│   └── baseline_config.yaml          # 基线版本配置
├── src/                               # 源代码目录
│   ├── __init__.py
│   ├── envs/                          # 环境相关代码
│   │   ├── __init__.py
│   │   ├── cabinet_env_base.py       # 基础环境
│   │   ├── cabinet_env_sensors.py    # 传感器增强环境
│   │   └── cabinet_env_baseline.py   # 基线环境
│   ├── tasks/                         # 任务定义
│   │   ├── __init__.py
│   │   ├── cabinet_task_base.py      # 基础任务
│   │   └── cabinet_task_sensors.py   # 传感器任务
│   ├── utils/                         # 工具函数
│   │   ├── __init__.py
│   │   ├── sensors.py                # 传感器工具
│   │   ├── visualization.py          # 可视化工具
│   │   └── metrics.py                # 评估指标
│   └── experiments/                   # 实验相关代码
│       ├── __init__.py
│       ├── base_experiment.py        # 基础实验类
│       ├── sensor_comparison.py      # 传感器对比实验
│       └── analysis.py               # 结果分析
├── scripts/                           # 脚本目录
│   ├── train_baseline.py             # 基线训练脚本
│   ├── train_sensors.py              # 传感器训练脚本
│   ├── run_comparison.py             # 对比实验脚本
│   ├── analyze_results.py            # 结果分析脚本
│   └── demo/                          # 演示脚本
│       ├── sensor_demo.py            # 传感器演示
│       └── diagnostic.py             # 诊断工具
├── tests/                             # 测试代码
│   ├── __init__.py
│   ├── test_environments.py          # 环境测试
│   ├── test_sensors.py               # 传感器测试
│   └── test_experiments.py           # 实验测试
├── data/                              # 数据目录
│   ├── configs/                       # 配置文件
│   ├── models/                        # 训练好的模型
│   └── results/                       # 实验结果
├── logs/                              # 日志目录
│   ├── training/                      # 训练日志
│   ├── experiments/                   # 实验日志
│   └── debug/                         # 调试日志
├── docs/                              # 文档目录
│   ├── setup.md                       # 安装说明
│   ├── usage.md                       # 使用说明
│   ├── experiments.md                 # 实验说明
│   └── api/                           # API文档
└── examples/                          # 示例代码
    ├── basic_usage.py                 # 基础使用示例
    ├── sensor_comparison.py           # 传感器对比示例
    └── custom_experiment.py           # 自定义实验示例
```

## 环境要求

- Python 3.8+
- IsaacLab (安装在: `/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab`)
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Pandas
- psutil

## 安装

1. 克隆项目：
```bash
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws
git clone <your-repo-url> cabinet_sensor_project
cd cabinet_sensor_project
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. 安装依赖：
```bash
pip install -r requirements.txt
pip install -e .  # 以开发模式安装项目
```

## 快速开始

### 基本训练

```bash
# 训练基线版本
python scripts/train_baseline.py --num_envs 64 --max_iterations 1000

# 训练传感器版本
python scripts/train_sensors.py --num_envs 64 --max_iterations 1000
```

### 对比实验

```bash
# 运行完整对比实验
python scripts/run_comparison.py --num_seeds 3 --max_iterations 2000

# 分析实验结果
python scripts/analyze_results.py --experiment_dir data/results/latest
```

### 演示

```bash
# 传感器功能演示
python scripts/demo/sensor_demo.py

# 系统诊断
python scripts/demo/diagnostic.py
```

## 主要功能

1. **传感器增强环境** - 集成接触传感器的机器人抓取环境
2. **基线环境** - 不使用传感器的标准环境
3. **自动化对比实验** - 批量运行实验并自动分析结果
4. **可视化分析** - 生成训练曲线和性能对比图表
5. **进程管理** - 智能的Isaac Sim进程清理和资源管理

## 配置说明

项目使用YAML配置文件管理参数：

- `config/base_config.yaml` - 基础配置（环境、训练参数等）
- `config/sensor_config.yaml` - 传感器特定配置
- `config/baseline_config.yaml` - 基线版本配置

## 实验流程

1. **环境设置** - 配置IsaacLab环境和项目依赖
2. **训练准备** - 设置训练参数和随机种子
3. **批量训练** - 自动运行多个配置的训练任务
4. **结果收集** - 收集训练日志和性能指标
5. **分析比较** - 生成对比分析报告和可视化图表

## 开发指南

### 添加新环境

1. 在 `src/envs/` 中创建新的环境文件
2. 继承 `cabinet_env_base.py` 中的基础类
3. 实现必要的方法和配置
4. 在 `config/` 中添加相应的配置文件

### 添加新实验

1. 在 `src/experiments/` 中创建实验类
2. 继承 `base_experiment.py` 中的基础实验类
3. 实现实验逻辑和结果分析
4. 在 `scripts/` 中添加启动脚本

### 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_environments.py
```

## 贡献指南

1. Fork项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送到分支：`git push origin feature/new-feature`
5. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请创建issue或联系项目维护者。

## 更新日志

### v1.0.0
- 初始版本
- 基本的传感器vs基线对比功能
- 自动化实验流程
- 结果分析和可视化 