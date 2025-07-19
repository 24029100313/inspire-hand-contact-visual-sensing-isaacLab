#!/bin/bash
# Contact Sensor Experiment Launcher for Isaac Lab

echo "🔬 CONTACT SENSOR LEARNING EXPERIMENT"
echo "======================================"
echo
echo "📋 实验概述："
echo "• 四个contact sensor排列成正方形"
echo "• 1kg立方体从2m高度自由落下"
echo "• 验证合力 = 重力 (9.81N)"
echo "• 学习多传感器设置"
echo
echo "🚀 启动实验..."
echo

# 确保在正确的目录
cd "$(dirname "$0")/.."

# 检查IsaacLab是否存在
if [ ! -d "../IsaacLab" ]; then
    echo "❌ IsaacLab目录未找到"
    echo "请确保您在正确的项目目录中"
    exit 1
fi

# 复制实验脚本到IsaacLab目录，这样可以直接运行
cp scripts/contact_sensor_simple.py ../IsaacLab/

# 切换到IsaacLab目录并运行
cd ../IsaacLab

echo "🎯 使用Isaac Lab环境运行实验..."
echo "   ./isaaclab.sh -p contact_sensor_simple.py"
echo

# 运行实验
./isaaclab.sh -p contact_sensor_simple.py

# 清理
rm -f contact_sensor_simple.py

echo "✅ 实验完成！"
