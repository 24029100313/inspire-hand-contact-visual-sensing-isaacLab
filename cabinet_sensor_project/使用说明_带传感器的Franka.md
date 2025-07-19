# 带传感器的Franka Panda机器人使用说明

## 概述
本项目将原始的Franka Panda URDF文件扩展，添加了8个触觉传感器，并创建了一个新的Isaac Lab环境来演示传感器功能。

## 文件结构
```
cabinet_sensor_project/
├── panda_arm_hand_with_sensors.urdf          # 修改后的URDF文件（含8个传感器）
├── lift_cube_sm_with_sensors.py              # 带传感器的抓取演示程序
├── run_lift_with_sensors.sh                  # 启动脚本
├── URDF_Sensor_Modification_Guide.md         # URDF修改说明
└── 使用说明_带传感器的Franka.md               # 本文档
```

## 传感器配置详情

### 🎯 传感器位置
- **左夹爪**: 4个传感器，排列为2×2正方形
- **右夹爪**: 4个传感器，排列为2×2正方形
- **位置精度**: 基于你的Python代码中的确切坐标

### 📏 传感器特性
- **尺寸**: 8mm × 8mm × 2mm
- **颜色**: 绿色（便于可视化）
- **质量**: 0.001kg（极轻，不影响动力学）
- **连接**: 通过固定关节连接到夹爪

### 🔗 传感器跟随运动
**是的！** 8个传感器会完全跟随夹爪运动：
- 当夹爪开合时，传感器保持相对位置
- 当机器人移动时，传感器跟随整个机器人运动
- 传感器通过固定关节直接连接到夹爪链接

## 使用方法

### 1. 配置Isaac Lab路径
**重要：** 首先需要修改启动脚本中的Isaac Lab路径：

```bash
# 编辑启动脚本
nano run_lift_with_sensors.sh

# 找到并修改这一行为你的实际路径
ISAACLAB_PATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab"
```

### 2. 快速启动
```bash
# 使用启动脚本（推荐）
./run_lift_with_sensors.sh

# 带参数运行
./run_lift_with_sensors.sh --headless --num_envs 8
```

### 3. 手动运行（高级用户）
```bash
# 确保在Isaac Lab环境中
source /path/to/IsaacLab/isaaclab.sh

# 运行程序
./isaaclab.sh -p lift_cube_sm_with_sensors.py --num_envs 4 --device cuda
```

### 4. 启动参数
```bash
# 基本参数
--num_envs 4          # 并行环境数量
--device cuda         # 运行设备（cuda/cpu）
--headless           # 无头模式运行

# 示例：运行16个环境，无头模式
./run_lift_with_sensors.sh --num_envs 16 --headless
```

### 5. 验证配置
启动脚本会自动检查：
- ✅ Isaac Lab路径是否正确
- ✅ URDF文件是否存在
- ✅ 所有必要文件是否就位

成功启动后会看到：
```
🤖 Franka抓取任务 - 夹爪传感器实时监控（URDF集成传感器版本）
📁 项目目录: /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project
🎯 Isaac Lab: /home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab
✅ 找到Isaac Lab: /home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab/isaaclab.sh
✅ 找到URDF文件: /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/panda_arm_hand_with_sensors.urdf
🚀 启动带传感器的Franka抓取任务...
```

## 关于USD转换

### 🤔 需要转换为USD吗？
**不需要手动转换！** Isaac Lab会自动处理：

1. **自动转换**: Isaac Lab会自动将URDF转换为USD格式
2. **缓存机制**: 转换后的USD文件会被缓存，提高后续加载速度
3. **透明处理**: 你只需要提供URDF文件路径即可

### 📁 USD文件位置
转换后的USD文件通常保存在：
```
~/.local/share/ov/pkg/isaac_sim-*/cache/
```

## 传感器数据访问

### 🔍 实时传感器数据
程序会每100步打印一次传感器数据：
```
=== Contact Sensor Data (Environment 0) ===
panda_leftfinger_sensor_1: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_leftfinger_sensor_2: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_leftfinger_sensor_3: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_leftfinger_sensor_4: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_rightfinger_sensor_1: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_rightfinger_sensor_2: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_rightfinger_sensor_3: Force=[0.000 0.000 0.000], Magnitude=0.000
panda_rightfinger_sensor_4: Force=[0.000 0.000 0.000], Magnitude=0.000
```

### 📊 在代码中访问传感器数据
```python
# 获取特定传感器数据
force_data = sensor_manager.get_sensor_data("panda_leftfinger_sensor_1")

# 获取所有传感器数据
all_sensor_data = sensor_manager.get_all_sensor_data()

# 打印调试信息
sensor_manager.print_sensor_data(env_id=0)
```

## 主要改进

### ✅ 相比原始Python方法的优势
1. **简化配置**: 无需动态创建传感器pad
2. **更好性能**: 减少运行时计算开销
3. **标准化**: 符合ROS/URDF标准
4. **易于集成**: 与其他机器人工具兼容

### 🔧 技术特点
1. **URDF直接集成**: 传感器直接定义在URDF中
2. **自动跟随**: 传感器随夹爪运动
3. **接触检测**: 支持实时力反馈
4. **可视化**: 绿色传感器便于调试

## 故障排除

### 🐛 常见问题

**问题1**: 找不到URDF文件
```bash
错误：找不到URDF文件 panda_arm_hand_with_sensors.urdf
```
**解决**: 确保在正确的目录中运行，且URDF文件存在

**问题2**: Isaac Lab环境未激活
```bash
警告：未检测到Isaac Lab环境变量
```
**解决**: 激活Isaac Lab环境
```bash
source /path/to/IsaacLab/isaaclab.sh
```

**问题3**: 传感器数据为零
- 检查是否有物体与传感器接触
- 确保传感器配置正确
- 查看可视化确认传感器位置

### 🔍 调试技巧
1. **启用可视化**: 传感器会显示为绿色小方块
2. **检查控制台**: 注意传感器加载信息
3. **调整打印频率**: 修改`print_interval`参数

## 扩展功能

### 🚀 可能的扩展
1. **传感器数据记录**: 保存传感器数据到文件
2. **力反馈控制**: 基于传感器数据调整抓取力度
3. **多物体抓取**: 使用传感器数据进行更复杂的抓取任务
4. **传感器融合**: 结合多个传感器数据进行决策

### 🎯 应用场景
- **精密抓取**: 基于力反馈的精确抓取
- **物体检测**: 通过接触检测识别物体
- **安全控制**: 防止过度挤压脆弱物体
- **学习算法**: 为强化学习提供额外的感知信息

## 版本信息
- **创建日期**: 2024年1月
- **Isaac Lab版本**: 基于当前Isaac Lab版本
- **Python版本**: 3.8+
- **依赖项**: Isaac Lab, PyTorch, Warp

---

## 🎉 开始使用

1. 确保所有文件都在同一目录中
2. 激活Isaac Lab环境
3. 运行启动脚本：`./run_lift_with_sensors.sh`
4. 观察传感器数据输出
5. 在Isaac Sim中查看绿色传感器可视化

**祝你使用愉快！**🤖 