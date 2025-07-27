#!/bin/bash

echo "🎯 =============================================="
echo "   触觉感知 vs 基线版本 RL 对比实验"
echo "=============================================="
echo ""
echo "📊 实验设计:"
echo "├── 版本1: 基线版本 (标准奖励函数)"
echo "├── 版本2: 触觉版本 (传感器增强奖励函数)"
echo "├── 训练迭代: 500次 (充分学习触觉控制)"
echo "├── 并行环境: 64个"
echo "└── 预计时间: ~12分钟"
echo ""
echo "🔬 关键差异:"
echo "├── 基线版本总奖励权重: ~12.5"
echo "│   ├── approach_ee_handle: 2.0"
echo "│   ├── open_drawer_bonus: 7.5"
echo "│   └── 其他: 3.0"
echo "├── 触觉版本总奖励权重: ~15.7 (更高!)"
echo "│   ├── 传统奖励: 6.7 (减少权重)"
echo "│   ├── gentle_contact: 3.0 (新增)"
echo "│   ├── contact_detection: 2.0 (新增)" 
echo "│   └── progressive_contact: 4.0 (新增)"
echo "└── 预期: 触觉版本应获得 10-20% 性能提升"
echo ""
echo "开始实验? (按 Ctrl+C 取消, 按 Enter 继续)"
read

# 激活虚拟环境
source venv/bin/activate

echo ""
echo "🔄 第1阶段: 基线版本训练 (传统奖励函数)"
echo "================================================"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Open-Drawer-Franka-Baseline-v0 \
    --num_envs 64 \
    --max_iterations 500 \
    --headless \
    --experiment_name tactile_baseline_500 \
    2>&1 | tee tactile_baseline_500.log

echo ""
echo "🤖 第2阶段: 触觉感知版本训练 (传感器增强奖励)"
echo "================================================"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Open-Drawer-Franka-Tactile-v0 \
    --num_envs 64 \
    --max_iterations 500 \
    --headless \
    --experiment_name tactile_enhanced_500 \
    2>&1 | tee tactile_enhanced_500.log

echo ""
echo "📊 实验结果分析"
echo "================================================"
echo "基线版本最终性能:"
grep -E "Episode_Reward|mean_reward" tactile_baseline_500.log | tail -5
echo ""
echo "触觉版本最终性能:"
grep -E "Episode_Reward|mean_reward" tactile_enhanced_500.log | tail -5
echo ""
echo "🎉 实验完成! 日志文件:"
echo "├── tactile_baseline_500.log"
echo "└── tactile_enhanced_500.log"
echo ""
echo "预期结果:"
echo "├── 基线版本: ~67分 (与之前100次迭代相似)"
echo "└── 触觉版本: ~75-80分 (传感器辅助提升10-20%)"
