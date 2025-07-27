#!/bin/bash

echo "🤖 Franka抓取任务 - 夹爪传感器实时监控（正确的变换矩阵方法）"
echo "📁 项目目录: $(pwd)"

# !!重要!! 请将此路径修改为您自己电脑上IsaacLab的实际路径
ISAACLAB_PATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎯 Isaac Lab: $ISAACLAB_PATH"
echo "📦 抓取任务：Franka机械臂抓取立方体"
echo "🔍 传感器监控：夹爪指尖接触力实时检测"
echo "🟢 传感器位置：左右夹爪指尖 (panda_leftfinger, panda_rightfinger)"
echo "🔧 改进：使用quat_apply()进行正确的位姿变换"
echo "📐 变换方法：局部偏移 → 四元数旋转 → 世界坐标"
echo "🔬 输出格式: 每个接触点输出 位置[x,y,z], 四元数[w,x,y,z], 力矢量[fx,fy,fz]"
echo "📊 终端实时显示接触状态和力分布"
echo "💾 数据自动保存至: gripper_sensor_data/gripper_contact_data_*.csv"
echo ""

# 检查Isaac Lab路径是否存在
if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "❌ 错误: Isaac Lab路径未找到或不正确: $ISAACLAB_PATH"
    echo "请修改脚本中的 ISAACLAB_PATH 变量。"
    exit 1
fi

# 运行抓取任务Python脚本
# "$@" 会将所有传递给此脚本的额外参数（如 --headless）传递给Python脚本
"$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/lift_cube_with_sensors_precise.py" "$@" 
