#!/bin/bash

echo "🔬 静态力平衡测试 - 逐接触点详细数据输出"
echo "📁 项目目录: $(pwd)"

# !!重要!! 请将此路径修改为您自己电脑上IsaacLab的实际路径
ISAACLAB_PATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎯 Isaac Lab: $ISAACLAB_PATH"
echo "📦 静态测试：1kg红色立方体直接放在4个绿色传感器上"
echo "🔍 验证物理定律：ΣF = mg ≈ 9.81N"
echo "🟢 传感器pad设为kinematic，以作为纯粹的静态测量平台"
echo "🔬 输出格式: 每个接触点输出 位置[x,y,z], 法线[nx,ny,nz], 力矢量[fx,fy,fz]"
echo "📊 终端实时显示力分布和误差分析"
echo "💾 数据自动保存至: data/contact_data.csv"
echo ""

# 检查Isaac Lab路径是否存在
if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "❌ 错误: Isaac Lab路径未找到或不正确: $ISAACLAB_PATH"
    echo "请修改脚本中的 ISAACLAB_PATH 变量。"
    exit 1
fi

# 运行静态力测试Python脚本
# "$@" 会将所有传递给此脚本的额外参数（如 --headless）传递给Python脚本
"$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/static_force_test.py" "$@"
