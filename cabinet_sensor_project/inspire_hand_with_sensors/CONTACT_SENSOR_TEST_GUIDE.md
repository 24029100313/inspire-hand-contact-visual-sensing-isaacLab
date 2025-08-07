# Inspire Hand Contact Sensor Test Guide

## 📋 概述

本指南详细说明如何运行和验证10个特定传感器pad的接触力检测功能。测试环境会让Inspire Hand执行抓取动作，并实时监测传感器数据。

## 🚀 快速开始

### 1. 运行测试脚本

```bash
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors

# 使用IsaacLab运行（推荐）
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab
./isaaclab.sh -p /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/test_contact_sensors_specific_pads.py --num_envs 1

# 如果需要无头模式（不显示GUI）
./isaaclab.sh -p /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/test_contact_sensors_specific_pads.py --num_envs 1 --headless
```

## 📦 测试环境说明

### 场景配置
- **手部模型**: 使用`inspire_hand_processed_with_specific_pads.usd`（包含10个特定传感器pad）
- **目标物体**: 6cm x 6cm x 6cm的红色立方体，质量100g
- **手部初始位置**: (0, 0, 0.3) - 位于立方体上方
- **手部朝向**: 手掌朝下（四元数: 0.7071, 0, 0, 0.7071）
- **立方体位置**: (0, -0.05, 0.03) - 在地面上，稍微靠前

### 监测的传感器pad
```
食指传感器2区：
- index_sensor_2_pad_045
- index_sensor_2_pad_046
- index_sensor_2_pad_052
- index_sensor_2_pad_053

食指传感器3区：
- index_sensor_3_pad_005

拇指传感器3区：
- thumb_sensor_3_pad_042
- thumb_sensor_3_pad_043
- thumb_sensor_3_pad_054
- thumb_sensor_3_pad_055

拇指传感器4区：
- thumb_sensor_4_pad_005
```

## 🎮 控制说明

### Inspire Hand控制范围

**重要**: Inspire Hand使用**0-1000的专有数值范围**进行关节控制，而不是标准的弧度制：

- **0**: 完全伸展/打开
- **1000**: 完全弯曲/闭合
- **中间值**: 线性插值

### 关节控制值映射

测试脚本中使用的控制值（0-1000范围）：

```python
# 拇指关节 - 较小的运动范围以实现对掌抓取
thumb_1: 0-300   # 适度弯曲
thumb_2: 0-500   # 中等弯曲
thumb_3: 0-400   # 中等弯曲
thumb_4: 0-300   # 适度弯曲

# 食指关节 - 主要抓取手指
index_1: 0-800   # 较大弯曲
index_2: 0-1000  # 完全弯曲

# 其他手指 - 辅助抓取
middle_1: 0-700  # 较大弯曲
middle_2: 0-900  # 接近完全弯曲
ring_1: 0-600    # 中等弯曲
ring_2: 0-800    # 较大弯曲
little_1: 0-500  # 中等弯曲
little_2: 0-700  # 较大弯曲
```

### 与Isaac Sim的接口

由于Isaac Sim期望弧度制输入，测试脚本会自动进行转换：
1. 在内部使用0-1000的Inspire Hand原生范围生成控制命令
2. 通过`_convert_to_radians()`方法转换为弧度
3. 将弧度值发送给Isaac Sim的关节控制器

## 📊 测试序列

测试分为三个阶段循环：

1. **打开阶段 (0-5秒)**
   - 所有关节位置 = 0
   - 手完全张开

2. **闭合阶段 (5-10秒)**
   - 逐渐闭合手指
   - 拇指和食指形成对掌抓取
   - 其他手指提供支撑

3. **保持阶段 (10-20秒)**
   - 维持抓取姿态
   - 监测稳定的接触力

4. **循环重复**

## 📈 数据输出解释

### 实时日志格式
```
⏱️  Time: 7.5s | Phase: Closing
------------------------------------------------------------
🔵 Index Sensor 2 Pads:
   index_sensor_2_pad_045: 0.000 N ⭕
   index_sensor_2_pad_046: 0.125 N ✅
   index_sensor_2_pad_052: 0.000 N ⭕
   index_sensor_2_pad_053: 0.102 N ✅

🟦 Thumb Sensor 3 Pads:
   thumb_sensor_3_pad_042: 0.098 N ⭕
   thumb_sensor_3_pad_043: 0.156 N ✅
   ...

📊 Active pads: 4/10
============================================================
```

### 符号说明
- ✅ = 接触力 > 10g (0.098N) - 有效接触
- ⭕ = 接触力 < 10g - 无有效接触
- 力值单位：牛顿(N)

## 🔧 故障排除

### 1. 内存不足错误
```bash
# 使用无头模式减少内存使用
./isaaclab.sh -p test_contact_sensors_specific_pads.py --headless

# 或减少环境数量（默认为1）
./isaaclab.sh -p test_contact_sensors_specific_pads.py --num_envs 1
```

### 2. 传感器无响应
- 检查USD文件是否正确加载
- 确认`activate_contact_sensors=True`
- 验证传感器pad名称是否匹配

### 3. 手指运动异常
- 确保使用正确的0-1000控制范围
- 检查是否误用了弧度制（错误）
- 验证关节限位是否合理

### 4. 接触力过大/过小
调整执行器参数：
```python
# 减小刚度以获得更柔和的抓取
stiffness=50.0  # 默认100.0

# 调整阻尼
damping=5.0     # 默认10.0
```

## 📝 自定义测试

### 修改抓取模式（使用0-1000范围）
编辑`_generate_grasp_motion()`函数：
```python
# 示例：只用拇指和食指
inspire_pos[:, 0] = 500 * close_ratio  # thumb_1
inspire_pos[:, 4] = 1000 * close_ratio  # index_1
# 其他手指保持打开（值为0）
```

### 改变物体属性
```python
# 修改立方体大小
size=(0.08, 0.08, 0.08)  # 8cm立方体

# 修改质量
mass=0.2  # 200g

# 修改位置
pos=(0.0, -0.08, 0.04)  # 更远一点
```

### 调整传感器阈值
```python
# 修改触发阈值（默认10g）
force_threshold: 0.049  # 5g
# 或
force_threshold: 0.196  # 20g
```

## 🎯 期望结果

成功的抓取测试应该显示：
1. **闭合阶段**: 逐渐有传感器检测到接触（2-4个pad）
2. **保持阶段**: 稳定的接触力（4-6个pad激活）
3. **力分布**: 拇指和食指传感器应显示最大的力

典型的成功抓取：
- 食指传感器2区: 1-2个pad激活
- 食指传感器3区: 可能激活
- 拇指传感器3区: 2-3个pad激活
- 拇指传感器4区: 可能激活

## 🔍 深入分析

### 导出数据
可以修改代码添加数据记录：
```python
# 在_log_contact_forces()中添加
import csv
with open('contact_forces.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([self.phase_timer] + list(self.contact_forces.values()))
```

### 可视化
- 使用`debug_vis=True`显示接触点
- 在Isaac Sim中使用Physics Inspector查看详细信息
- 使用matplotlib绘制力-时间曲线

## 📚 相关资源

- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/)
- [ContactSensor API](https://isaac-sim.github.io/IsaacLab/source/api/isaaclab.sensors.html#contact-sensor)
- [Articulation Control](https://isaac-sim.github.io/IsaacLab/source/api/isaaclab.assets.html#articulation)
- [Inspire Hand Controller源码](inspire_hand_controller.py) - 查看0-1000控制范围的实现

---

**注意**: 
1. Inspire Hand使用0-1000的专有控制范围，不是弧度制
2. 本测试环境专门为验证10个特定传感器pad设计
3. 如需测试更多传感器，请使用完整的1061传感器版本（需要更多内存） 