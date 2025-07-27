#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
静态力学测试 - 一个方块静置在4个接触式传感器上，用于验证力的平衡。
测试目标: ΣF = mg ≈ 9.81N，并使用正确的碰撞过滤。
此版本增强为为每个传感器输出详细的、逐接触点的数据，并自动记录到CSV文件。
"""

# --- 标准库 ---
import argparse  # 用于解析从命令行传入的参数，如 --headless
import os        # 用于操作系统功能，如此处的创建文件夹 (mkdir)
import csv       # 用于读写CSV（逗号分隔值）文件，用于数据记录

# --- Isaac Lab 核心库 ---
from isaaclab.app import AppLauncher                 # Isaac Lab 应用程序启动器，管理仿真应用的生命周期

# add argparse arguments
parser = argparse.ArgumentParser(description="Static force test using contact sensors.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils                   # Isaac Lab 仿真核心工具集，如 SimulationContext, DomeLightCfg 等
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg # 用于定义场景中的资源，如机器人、物体等
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg # 用于构建和管理整个交互式场景
from isaaclab.sensors import ContactSensor, ContactSensorCfg # 用于定义和配置接触传感器
from isaaclab.utils import configclass               # 一个好用的工具，让类可以像字典一样被配置

# --- 第三方库 ---
import torch     # PyTorch库，Isaac Lab 底层使用它进行高效的张量运算

# ================================= #
#       2. 定义场景配置 (SceneCfg)     #
# ================================= #

@configclass
class StaticForceSceneCfg(InteractiveSceneCfg):
    """
    静态力学测试场景的配置类。
    这里定义了场景中所有的物体、灯光、传感器及其初始状态。
    """

    # --- 场景设置 ---
    env_spacing = 4.0  # 环境间距，对于单环境测试设为4.0就足够了

    # --- 环境设置 ---
    # 定义一个地面
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 定义一个穹顶灯光，用于提供场景的整体照明
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # --- 基础平板定义 ---
    # 定义一个大的基础平板，作为传感器系统的底座
    base_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BasePlatform",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.05),  # 大平板尺寸：120cm x 120cm x 5cm
            # 物理属性:
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True  # 设置为运动学物体，位置固定
            ),
            # 质量属性
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            # 碰撞属性 - 使用默认设置
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 视觉材质: 灰色
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7), metallic=0.1),
        ),
        # 初始状态: 基础平板的位置
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.025), # 平板底部刚好在地面上，顶部在z=0.05
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --- 传感器平台定义 ---
    # 定义4个传感器垫，现在放置在基础平板上面
    sensor_pad_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.002, 0.002, 0.0002),  # 垫的尺寸：20cm x 20cm x 2cm
            # 物理属性:
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True  # !!核心设置!! 设置为运动学物体。这意味着它不受力影响，位置固定，是理想的静态传感器平台。
            ),
            # 质量属性 (对于运动学物体，此项不起作用，但最好还是定义一下)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            # 碰撞属性 - 使用默认设置
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 视觉材质:
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.3), # 绿色
            # !!核心设置!! 必须设为True，物理引擎才会为这个物体生成接触报告。
            activate_contact_sensors=True,
        ),
        # 初始状态: 传感器垫1的位置，现在在基础平板上面
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.25, 0.06), # 基础平板顶部(0.05) + 传感器垫半厚度(0.01) = 0.06
            rot=(1.0, 0.0, 0.0, 0.0), # 垫子的姿态（无旋转）
        ),
    )
    
    sensor_pad_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.002, 0.002, 0.0002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.3),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.25, 0.25, 0.06), # 在基础平板上面
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    sensor_pad_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_3",
        spawn=sim_utils.CuboidCfg(
            size=(0.002, 0.002, 0.0002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.3),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, -0.25, 0.06), # 在基础平板上面
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    sensor_pad_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_4",
        spawn=sim_utils.CuboidCfg(
            size=(0.002, 0.002, 0.0002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.3),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.25, -0.25, 0.06), # 在基础平板上面
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # --- 测试物体定义 ---
    # 定义一个红色的方块，它将落在传感器上
    test_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.2),  # 方块尺寸：50cm x 50cm x 20cm，确保能覆盖所有传感器
            rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 默认动态刚体
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), # !!核心设置!! 质量设为1kg，便于验证重力 F=mg=1.0*9.81=9.81N
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2), # 红色
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18) # 传感器垫顶部(0.07) + 方块半高(0.1) + 一点间隙(0.01) = 0.18
        ),
    )

    # --- 传感器定义 ---
    # 为每一个传感器垫定义一个接触传感器
    contact_sensor_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_1",  # !!核心设置!! 路径指向上面定义的第一个传感器垫的body_name
        update_period=0.0,  # 更新周期为0.0意味着每一步都更新
        debug_vis=False,     # 在仿真视窗中可视化接触点（显示为小点）
        track_pose=True,    # !!核心设置!! 启用位置和方向跟踪
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TestCube"], # !!核心设置!! 过滤器，让此传感器只关心与"TestCube"的接触
    )
    contact_sensor_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_2",
        update_period=0.0, debug_vis=False, track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TestCube"],
    )
    contact_sensor_3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_3",
        update_period=0.0, debug_vis=False, track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TestCube"],
    )
    contact_sensor_4 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorPad_4",
        update_period=0.0, debug_vis=False, track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TestCube"],
    )

def get_detailed_contact_data(sensor: ContactSensor, env_id: int = 0, sensor_name: str = ""):
    """
    为传感器上每个活动的接触点提取并格式化详细数据。
    使用Isaac Lab的实时传感器数据（pos_w, quat_w, net_forces_w）获取动态信息。

    Args:
        sensor: 要读取的 ContactSensor 实例。
        env_id: 要检查的特定环境的索引。
        sensor_name: 传感器名称（用于日志显示）。

    Returns:
        一个字典列表，每个字典代表一个接触点的完整信息。
        返回12维核心数据：位置(3) + 四元数方向(4) + 力矢量(3) + 力大小(1) + 法线分量(1)
        法线方向可以从四元数实时计算，无需单独存储。
    """
    # 访问 .data 属性会触发传感器的 "惰性更新"，从物理引擎获取最新数据
    sensor_data = sensor.data
    contact_data_list = []

    # 检查是否有有效的力数据
    if sensor_data.net_forces_w is None:
        return []
    
    # 获取指定环境的力数据
    forces_w = sensor_data.net_forces_w[env_id]  # 形状: (B, 3)
    
    # 查找有效的接触力（力的模长大于阈值）
    force_magnitudes = torch.norm(forces_w, dim=-1)  # 形状: (B,)
    active_contact_indices = torch.where(force_magnitudes > 0.01)[0]

    # 如果没有找到活动的接触点，返回空列表
    if active_contact_indices.numel() == 0:
        return []

    # 核心改进：使用Isaac Lab提供的实时传感器数据
    
    # 1. 获取传感器的实时世界位置 (3维)
    if sensor_data.pos_w is not None and len(sensor_data.pos_w) > env_id:
        sensor_pos_w = sensor_data.pos_w[env_id]  # 实时位置 [x, y, z]
    else:
        # 如果实时位置不可用，直接报错
        raise RuntimeError(f"传感器 '{sensor_name}' 的实时位置数据不可用！sensor_data.pos_w={sensor_data.pos_w}")
    
    # 2. 获取传感器的实时世界方向四元数 (4维)
    if sensor_data.quat_w is not None and len(sensor_data.quat_w) > env_id:
        sensor_quat_w = sensor_data.quat_w[env_id]  # 实时四元数 [w, x, y, z]
    else:
        # 如果实时方向不可用，直接报错
        raise RuntimeError(f"传感器 '{sensor_name}' 的实时方向数据不可用！sensor_data.quat_w={sensor_data.quat_w}")
    
    # 3. 从四元数计算法线方向（Z轴方向）
    def quat_to_normal(quat):
        """从四元数计算传感器表面的法线方向（局部Z轴在世界坐标系中的方向）"""
        # 确保quat是4维张量
        if quat.numel() != 4:
            raise RuntimeError(f"四元数张量应该有4个元素，但实际有{quat.numel()}个元素：{quat}")
        
        # 将张量展平为1D以确保正确的索引
        quat_flat = quat.flatten()
        w, x, y, z = quat_flat[0], quat_flat[1], quat_flat[2], quat_flat[3]
        
        # 四元数旋转矩阵的第三列（Z轴方向）
        normal_x = 2.0 * (x * z + w * y)
        normal_y = 2.0 * (y * z - w * x) 
        normal_z = 1.0 - 2.0 * (x * x + y * y)
        return torch.stack([normal_x, normal_y, normal_z])

    sensor_normal_w = quat_to_normal(sensor_quat_w)  # 传感器表面法线方向

    # 使用找到的活动接触点索引，从原始数据缓冲区中筛选出有效数据
    valid_forces = forces_w[active_contact_indices]  # 形状: (N_active, 3)
    valid_magnitudes = force_magnitudes[active_contact_indices]  # 形状: (N_active,)

    # 对每个接触点，使用传感器的实时位置和方向
    for i in range(len(active_contact_indices)):
        # 构建12维完整数据：位置(3) + 四元数(4) + 力矢量(3) + 力大小(1) + 法线分量(1)
        contact_data_list.append({
            # === 位置信息 (3维) ===
            "position_w": sensor_pos_w.cpu().numpy().flatten().tolist(),  # 实时传感器位置
            
            # === 方向信息 (4维) ===
            "quat_w": sensor_quat_w.cpu().numpy().flatten().tolist(),  # 实时传感器四元数 - 确保为平铺列表
            
            # === 力信息 (4维) ===
            "force_vector_w": valid_forces[i].cpu().numpy().flatten().tolist(),  # 力矢量 [fx, fy, fz]
            "force_magnitude": valid_magnitudes[i].item(),  # 力的大小
            
            # === 额外分析信息 ===
            "force_along_normal": torch.dot(valid_forces[i], sensor_normal_w).item(),  # 沿法线方向的力分量
            "contact_index": active_contact_indices[i].item(),  # 接触点索引
        })

    return contact_data_list


def get_sensor_summary_data(sensor: ContactSensor, env_id: int = 0, sensor_name: str = ""):
 
    sensor_data = sensor.data
    
    if sensor_data.net_forces_w is None:
        return None
        
    forces_w = sensor_data.net_forces_w[env_id]
    total_force = torch.sum(forces_w, dim=0)  # 总力矢量
    total_magnitude = torch.norm(total_force).item()  # 总力大小
    
    if total_magnitude < 0.01:
        return None
    
    # 获取传感器信息
    if sensor_data.pos_w is not None and len(sensor_data.pos_w) > env_id:
        sensor_pos_w = sensor_data.pos_w[env_id]
    else:
        sensor_pos_w = torch.zeros(3, device=forces_w.device)
        
    if sensor_data.quat_w is not None and len(sensor_data.quat_w) > env_id:
        sensor_quat_w = sensor_data.quat_w[env_id]
    else:
        sensor_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=forces_w.device)
    
    return {
        "sensor_name": sensor_name,
        "sensor_position_w": sensor_pos_w.cpu().numpy().tolist(),
        "sensor_orientation_quat_w": sensor_quat_w.cpu().numpy().tolist(),
        "total_force_vector_w": total_force.cpu().numpy().tolist(),
        "total_force_magnitude": total_magnitude,
        "num_active_contacts": len(torch.where(torch.norm(forces_w, dim=-1) > 0.01)[0]),
    }

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行物理仿真，打印并保存传感器数据。"""
    # 获取物理引擎的步长时间 (dt)
    sim_dt = sim.get_physics_dt()
    # 在播放前，建议先步进一次，以确保所有对象都已在物理场景中正确初始化
    sim.step()

    # 开始渲染和物理仿真
    sim.play()

    # --- CSV 数据记录设置 ---
    output_dir = "data"  # 定义保存数据文件的文件夹名称
    os.makedirs(output_dir, exist_ok=True)  # 创建这个文件夹，如果已存在则不报错
    csv_filepath = os.path.join(output_dir, "contact_data.csv") # 构建完整的文件路径
    
    # 使用 `with open(...)` 语句来安全地读写文件，它能确保文件在操作结束后被正确关闭
    with open(csv_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile) # 创建一个CSV写入器
        # 定义并写入CSV文件的表头（第一行）- 更新为12维数据格式（去掉法线存储）
        header = [
            "Timestamp", "SensorName", "ContactID", "ContactIndex",
            # 位置信息 (3维)
            "PosX", "PosY", "PosZ",
            # 方向信息 (4维：四元数)
            "QuatW", "QuatX", "QuatY", "QuatZ",
            # 力信息 (4维)
            "Fx", "Fy", "Fz", "ForceMagnitude",
            # 分析信息 (1维)
            "ForceAlongNormal"
        ]
        csv_writer.writerow(header)

        print(f"数据将实时保存到: {csv_filepath}")
        print("让方块稳定下来...")
        # 在正式开始记录前，先空跑几步，让方块因为重力自然下落并稳定在传感器上
        for _ in range(100):
            sim.step()
            scene.update(sim_dt) # scene.update() 会更新传感器等对象的内部缓冲区

        print("\n--- 开始实时传感器数据记录 (12维优化数据) ---")
        # 仿真主循环，只要仿真应用在运行就会一直执行
        while simulation_app.is_running():
            # 执行一步物理仿真
            sim.step()
            # 更新Isaac Lab场景对象的内部数据
            scene.update(sim_dt)

            # --- 数据提取、打印和保存 ---
            print("=" * 100) # 打印分割线，让输出更清晰
            total_force_z = 0.0  # 重置总力计数器
            
            # 获取当前仿真时间，用于记录
            sim_time = sim.current_time

            # 遍历场景中所有被定义的传感器 (contact_sensor_1, _2, _3, _4)
            for sensor_name, sensor_obj in scene.sensors.items():
                # 调用我们的核心函数来获取格式化后的接触点数据
                contact_points = get_detailed_contact_data(sensor_obj, env_id=0, sensor_name=sensor_name)
                
                # 如果没有接触点，就跳过这个传感器
                if not contact_points:
                    continue # 注释掉下面的打印可以减少终端刷屏
                    # print(f"传感器 '{sensor_name}': 无接触。")
                
                print(f"传感器 '{sensor_name}' 报告了 {len(contact_points)} 个接触点:")
                sensor_total_force_z = 0.0
                
                # 遍历这个传感器上的每一个接触点
                for i, point in enumerate(contact_points):
                    # 提取12维数据（去掉法线存储，改为实时计算）
                    pos = point['position_w']
                    quat = point['quat_w'] 
                    f_vec = point['force_vector_w']
                    f_mag = point['force_magnitude']
                    f_normal = point['force_along_normal']
                    contact_idx = point['contact_index']
                    
                    # 从四元数实时计算法线方向（仅用于显示）
                    def quat_to_normal_display(quat_list):
                        # 处理嵌套列表情况：如果quat_list是[[1.0, 0.0, 0.0, 0.0]]，取第一个元素
                        if isinstance(quat_list, list) and len(quat_list) == 1 and isinstance(quat_list[0], list):
                            quat_list = quat_list[0]
                        
                        # 确保quat_list有4个元素
                        if len(quat_list) != 4:
                            raise RuntimeError(f"四元数列表应该有4个元素，但实际有{len(quat_list)}个元素：{quat_list}")
                        
                        w, x, y, z = quat_list[0], quat_list[1], quat_list[2], quat_list[3]
                        normal_x = 2.0 * (x * z + w * y)
                        normal_y = 2.0 * (y * z - w * x) 
                        normal_z = 1.0 - 2.0 * (x * x + y * y)
                        return [normal_x, normal_y, normal_z]
                    
                    norm = quat_to_normal_display(quat)  # 实时计算法线（仅显示用）
                    
                    # 在终端打印每个接触点的详细信息（12维数据）
                    print(f"  接触点 {i+1} (索引:{contact_idx}):")
                    print(f"    实时位置: [x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}]")
                    print(f"    四元数方向: [w={quat[0]:.3f}, x={quat[1]:.3f}, y={quat[2]:.3f}, z={quat[3]:.3f}]")
                    print(f"    力矢量: [fx={f_vec[0]:.3f}, fy={f_vec[1]:.3f}, fz={f_vec[2]:.3f}]")
                    print(f"    力大小: {f_mag:.3f} N | 法线分量: {f_normal:.3f} N")

                    # 累加Z轴方向的力，用于后续验证
                    sensor_total_force_z += f_vec[2]

                    # --- 将12维数据写入CSV文件 ---
                    row = [
                        f"{sim_time:.4f}", sensor_name, i, contact_idx,
                        # 位置 (3维)
                        f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                        # 四元数 (4维)
                        f"{quat[0]:.4f}", f"{quat[1]:.4f}", f"{quat[2]:.4f}", f"{quat[3]:.4f}",
                        # 力 (4维)
                        f"{f_vec[0]:.4f}", f"{f_vec[1]:.4f}", f"{f_vec[2]:.4f}", f"{f_mag:.4f}",
                        # 分析 (1维)
                        f"{f_normal:.4f}"
                    ]
                    csv_writer.writerow(row) # 写入一行数据

                total_force_z += sensor_total_force_z  # 累加所有传感器的力

            gravity = 9.81
            expected_force_z = -gravity * 1.0  # 重力向下，期望的力应该是负值
            print("=" * 100)
            print(f"统一物理验证结果（所有4个传感器）:")
            print(f"   测量的总垂直力 (ΣFz): {total_force_z:.4f} N")
            print(f"   期望的垂直力 (-mg):   {expected_force_z:.4f} N")
            print(f"   力的绝对值误差:       {abs(abs(total_force_z) - abs(expected_force_z)):.4f} N")
            print(f"   力的百分比误差:       {abs((total_force_z - expected_force_z)/expected_force_z)*100:.2f}%")
            
            # 为了避免终端刷屏太快，让肉眼可以看清，这里我们每打印一次信息就多空跑几步
            for _ in range(10): 
                sim.step()

def main():
    """主函数，负责组装和启动整个流程。"""
    # 1. 根据配置类创建场景配置实例
    scene_cfg = StaticForceSceneCfg(num_envs=args_cli.num_envs)
    # 2. 创建仿真上下文管理器 (SimulationContext)，它负责与物理引擎的交互
    sim = sim_utils.SimulationContext()
    # 3. 根据场景配置创建场景实例
    scene = InteractiveScene(scene_cfg)
    # 4. 调用我们的主循环函数，开始仿真
    run_simulator(sim, scene)

# `if __name__ == "__main__":` 是Python脚本的标准入口点。
# 当你直接运行这个文件时，这部分代码会被执行。
if __name__ == "__main__":
    # 调用主函数
    main()
    # 仿真结束后，关闭Omniverse应用程序
    simulation_app.close()
