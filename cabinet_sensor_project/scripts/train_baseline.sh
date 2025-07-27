#!/bin/bash

echo "🔵 开始基线任务训练..."
echo "任务: Isaac-Open-Drawer-Franka-v0 (无传感器)"
echo "参数: 64环境 × 100次迭代"
echo ""

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Open-Drawer-Franka-v0 \
    --num_envs 64 \
    --max_iterations 100 \
    --headless \
    --seed 42

echo ""
echo "✅ 基线训练完成！"
