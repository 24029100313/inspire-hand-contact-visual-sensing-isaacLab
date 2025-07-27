#!/bin/bash

echo "🤖 Franka抓取任务 - 夹爪传感器实时监控（USD集成传感器版本）"
echo "📁 项目目录: $(pwd)"

# !!重要!! 请将此路径修改为您自己电脑上IsaacLab的实际路径
ISAACLAB_PATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎯 Isaac Lab: $ISAACLAB_PATH"
echo "📦 抓取任务：Franka机械臂抓取立方体"
echo "🔍 传感器监控：夹爪8个传感器实时检测"
echo "🟢 传感器位置：USD集成传感器（左右夹爪各4个）"
echo "🔧 改进：传感器直接集成在USD中，无需动态创建"
echo "📐 USD文件：panda_arm_hand_with_sensors.usd"
echo "🔬 输出格式: 每个传感器输出接触力矢量[fx,fy,fz]"
echo "📊 终端实时显示接触状态和力分布"
echo "💾 数据自动保存至: gripper_sensor_data/gripper_contact_data_*.csv"
echo ""

# 检查Isaac Lab路径是否存在
if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "❌ 错误: Isaac Lab路径未找到或不正确: $ISAACLAB_PATH"
    echo "请修改脚本中的 ISAACLAB_PATH 变量。"
    exit 1
fi

# 检查USD文件是否存在
if [ ! -f "$SCRIPT_DIR/panda_arm_hand_with_sensors.usd" ]; then
    echo "❌ 错误：找不到USD文件 panda_arm_hand_with_sensors.usd"
    echo "请确保文件在项目目录中"
    exit 1
fi

echo "✅ 找到Isaac Lab: $ISAACLAB_PATH/isaaclab.sh"
echo "✅ 找到USD文件: $SCRIPT_DIR/panda_arm_hand_with_sensors.usd"
echo ""

echo "启动参数说明："
echo "  --num_envs 4     # 并行环境数量（默认：4）"
echo "  --headless       # 无头模式运行"
echo "  --device cuda    # 运行设备（cuda/cpu）"
echo ""

# 运行抓取任务Python脚本
# "$@" 会将所有传递给此脚本的额外参数（如 --headless）传递给Python脚本
echo "🚀 启动带传感器的Franka抓取任务..."
"$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/lift_cube_sm_with_sensors.py" --num_envs 4 --device cuda "$@"

echo ""
echo "✅ 程序已结束" 