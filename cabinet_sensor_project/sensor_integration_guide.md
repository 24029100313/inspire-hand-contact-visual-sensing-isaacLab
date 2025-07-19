# Franka机器人传感器集成使用指南

## 🎯 完成的工作

✅ 在Franka夹爪的黑色橡胶块上添加了8个传感器pad（每个夹爪4个）
✅ 创建了带传感器的Franka机器人文件
✅ 避免了运行时位置更新的延迟问题

## 📁 文件位置

### 主要文件
- **机器人文件**: `/home/larry/NVIDIA_DEV/isaac-sim/Assets/Isaac/4.5/Isaac/Robots/Franka/franka_with_sensors.usda`
- **左夹爪**: `/home/larry/NVIDIA_DEV/isaac-sim/Assets/Isaac/4.5/Isaac/Robots/Franka/Props/panda_leftfinger_with_sensors.usda`
- **右夹爪**: `/home/larry/NVIDIA_DEV/isaac-sim/Assets/Isaac/4.5/Isaac/Robots/Franka/Props/panda_rightfinger_with_sensors.usda`

### 备份文件
- 原始文件已备份为 `*_original.usd`

## 🔧 如何使用

### 1. 在Isaac Lab中使用新的机器人

修改你的机器人配置，使用新的USDA文件：

```python
# 原来的配置
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd",
        # ...
    ),
)

# 修改为带传感器的配置
FRANKA_PANDA_WITH_SENSORS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/larry/NVIDIA_DEV/isaac-sim/Assets/Isaac/4.5/Isaac/Robots/Franka/franka_with_sensors.usda",
        activate_contact_sensors=True,  # 重要：启用接触传感器
    ),
)
```

### 2. 配置传感器

现在可以直接在传感器pad上添加ContactSensor：

```python
# 左夹爪传感器配置
left_sensors_cfg = {
    "gripper_left_sensor_1": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger/sensor_pad_1",
        track_pose=True,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    ),
    "gripper_left_sensor_2": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger/sensor_pad_2",
        track_pose=True,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    ),
    # ... 其他传感器
}

# 右夹爪传感器配置
right_sensors_cfg = {
    "gripper_right_sensor_1": ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger/sensor_pad_1",
        track_pose=True,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    ),
    # ... 其他传感器
}
```

### 3. 传感器pad规格

- **尺寸**: 8mm × 8mm × 2mm
- **材质**: PlasticGreen（绿色，易于识别）
- **碰撞检测**: 已启用
- **质量**: 0.001kg
- **排列**: 正方形布局，间距9.52mm

### 4. 传感器pad位置（相对于夹爪局部坐标）

| Pad | X (mm) | Y (mm) | Z (mm) |
|-----|--------|--------|--------|
| 1   | -4.76  | 0.00   | 43.19  |
| 2   | +4.76  | 0.00   | 43.19  |
| 3   | -4.76  | 0.00   | 52.71  |
| 4   | +4.76  | 0.00   | 52.71  |

## ⚙️ 传感器数据处理

传感器pad会自动跟随夹爪运动，无需额外的位置更新代码。你可以直接使用Isaac Lab的ContactSensor API来获取：

- 实时接触力
- 接触位置
- 接触方向
- 力的大小和方向

## 🔄 下一步

1. 测试新的机器人文件加载是否正常
2. 验证8个传感器pad是否正确显示和工作
3. 根据需要调整传感器配置参数
4. 在你的抓取任务中集成传感器反馈

## 📞 故障排除

如果遇到问题：
1. 确认文件路径正确
2. 检查 `activate_contact_sensors=True` 是否设置
3. 验证传感器pad的prim路径是否匹配
4. 查看Isaac Sim的日志输出

---
*集成完成时间: 2025年7月17日*
