# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine with contact sensors on gripper.

This script extends the lift_cube_sm.py with contact sensors on the Franka gripper fingers.
The sensors will provide real-time force feedback during grasping operations.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_with_sensors.py --num_envs 4

"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import csv
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine with contact sensors for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence
from typing import Dict, List, Optional
import numpy as np

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
# Add proper pose transformation utilities
from isaaclab.utils.math import quat_apply, combine_frame_transforms
# 添加四元数操作相关导入
from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

# === 传感器Pad精确位置定义 ===
# 基于Franka USDA文件分析得出的黑色橡胶块位置
# 左手指传感器pad位置（世界坐标系相对于手指基准的局部坐标）
LEFT_FINGER_PAD_POSITIONS = [
    [-0.004760, 0.000000, 0.043190],  # Pad 1 (左下角)
    [0.004760, 0.000000, 0.043190],   # Pad 2 (右下角)
    [-0.004760, 0.000000, 0.052710],  # Pad 3 (左上角)
    [0.004760, 0.000000, 0.052710],   # Pad 4 (右上角)
]

# 右手指传感器pad位置（世界坐标系相对于手指基准的局部坐标）
RIGHT_FINGER_PAD_POSITIONS = [
    [-0.004760, 0.000000, 0.043190],  # Pad 1 (左下角)
    [0.004760, 0.000000, 0.043190],   # Pad 2 (右下角)
    [-0.004760, 0.000000, 0.052710],  # Pad 3 (左上角)
    [0.004760, 0.000000, 0.052710],   # Pad 4 (右上角)
]

# 位置信息说明
PAD_POSITION_INFO = {
    "description": "传感器pad正方形排列位置",
    "square_size": 0.00952,  # 正方形边长 (9.52mm)
    "rubber_block_size": [0.020, 0.012],  # 黑色橡胶块尺寸 [宽度, 长度]
    "coordinate_system": "局部坐标系（相对于手指基准）",
    "verification": "所有位置都在黑色橡胶块范围内",
    "orientation_correction": "传感器pad方向已修正为平行于黑色橡胶块表面",
    "left_finger_orientation": "左夹爪传感器pad面向右侧（正Y方向）",
    "right_finger_orientation": "右夹爪传感器pad面向左侧（负Y方向）"
}

def get_sensor_pad_orientation(finger_quat: torch.Tensor, is_left_finger: bool) -> torch.Tensor:
    """
    计算传感器pad的正确方向，使其平行于黑色橡胶块表面
    
    基于几何体分析结果：
    - 左夹爪：黑色橡胶块面向右侧（正Y方向）
    - 右夹爪：黑色橡胶块面向左侧（负Y方向）
    - 传感器pad应该平行于橡胶块表面，法线指向夹爪内侧
    
    Args:
        finger_quat: 夹爪的四元数 [N, 4] (w, x, y, z)
        is_left_finger: 是否是左夹爪
        
    Returns:
        传感器pad的正确四元数方向 [N, 4]
    """
    device = finger_quat.device
    batch_size = finger_quat.shape[0]
    
    if is_left_finger:
        # 左夹爪：传感器pad法线指向正Y方向（右侧）
        # 这确保传感器pad平行于黑色橡胶块表面
        # 推荐四元数 [w, x, y, z]: [0.7071, -0.7071, 0.0000, 0.0000]
        # 这是绕X轴旋转-90度的四元数
        pad_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], 
                               device=device, dtype=torch.float32)
    else:
        # 右夹爪：传感器pad法线指向负Y方向（左侧）
        # 这确保传感器pad平行于黑色橡胶块表面
        # 推荐四元数 [w, x, y, z]: [0.7071, 0.7071, 0.0000, 0.0000]
        # 这是绕X轴旋转90度的四元数
        pad_quat = torch.tensor([0.7071, 0.7071, 0.0000, 0.0000], 
                               device=device, dtype=torch.float32)
    
    # 扩展到批量大小
    pad_quat = pad_quat.unsqueeze(0).expand(batch_size, -1)
    
    # 将传感器pad的局部方向与夹爪的全局方向组合
    # 使用四元数乘法组合方向
    combined_quat = quat_mul(finger_quat, pad_quat)
    
    return combined_quat

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], des_object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.REST
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object.

    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str, position_threshold: float = 0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            position_threshold: The position threshold for the state machine.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, 6] = 1.0  # warp expects w-component of quaternion to be last

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset the state machine."""
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # increment wait time
        self.sm_wait_time += self.dt

        # convert all transformations from (x, y, z, qx, qy, qz, qw) to (qx, qy, qz, qw, x, y, z)
        ee_pose = ee_pose[:, [3, 4, 5, 6, 0, 1, 2]]
        object_pose = object_pose[:, [3, 4, 5, 6, 0, 1, 2]]
        des_object_pose = des_object_pose[:, [3, 4, 5, 6, 0, 1, 2]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (x, y, z, qx, qy, qz, qw)
        des_ee_pose = self.des_ee_pose[:, [4, 5, 6, 0, 1, 2, 3]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


class ContactSensorManager:
    """Manager for contact sensors on Franka gripper fingers."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the contact sensor manager."""
        self.num_envs = num_envs
        self.device = device
        self.sensors = {}
        self.csv_writers = {}
        self.csv_files = {}
        
        # Setup CSV logging
        self.setup_csv_logging()

    def setup_csv_logging(self):
        """Setup CSV logging for sensor data."""
        # Create data directory
        output_dir = "gripper_sensor_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create CSV file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filepath = os.path.join(output_dir, f"gripper_contact_data_{timestamp}.csv")
        
        # Open CSV file and create writer
        self.csv_file = open(csv_filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        header = [
            "Timestamp", "EnvID", "SensorName", "ContactID", "ContactIndex",
            # 位置信息 (3维)
            "PosX", "PosY", "PosZ",
            # 方向信息 (4维：四元数)
            "QuatW", "QuatX", "QuatY", "QuatZ",
            # 力信息 (4维)
            "Fx", "Fy", "Fz", "ForceMagnitude",
            # 分析信息 (1维)
            "ForceAlongNormal"
        ]
        self.csv_writer.writerow(header)
        
        print(f"传感器数据将实时保存到: {csv_filepath}")

    def add_sensor(self, sensor_name: str, sensor: ContactSensor):
        """Add a contact sensor to the manager."""
        self.sensors[sensor_name] = sensor

    def get_detailed_contact_data(self, sensor: ContactSensor, env_id: int = 0, sensor_name: str = ""):
        """Extract detailed contact data from a sensor (same as original implementation)."""
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

    def process_sensor_data(self, env_id: int, sim_time: float):
        """Process sensor data for all sensors and log to CSV."""
        for sensor_name, sensor in self.sensors.items():
            # 获取传感器数据
            contact_points = self.get_detailed_contact_data(sensor, env_id, sensor_name)
            
            # 如果没有接触点，跳过
            if not contact_points:
                continue
                
            print(f"环境 {env_id} - 传感器 '{sensor_name}' 报告了 {len(contact_points)} 个接触点:")
            
            # 处理每个接触点
            for i, point in enumerate(contact_points):
                # 提取12维数据
                pos = point['position_w']
                quat = point['quat_w'] 
                f_vec = point['force_vector_w']
                f_mag = point['force_magnitude']
                f_normal = point['force_along_normal']
                contact_idx = point['contact_index']
                
                # 在终端打印每个接触点的详细信息
                print(f"  接触点 {i+1} (索引:{contact_idx}):")
                print(f"    实时位置: [x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}]")
                print(f"    四元数方向: [w={quat[0]:.3f}, x={quat[1]:.3f}, y={quat[2]:.3f}, z={quat[3]:.3f}]")
                print(f"    力矢量: [fx={f_vec[0]:.3f}, fy={f_vec[1]:.3f}, fz={f_vec[2]:.3f}]")
                print(f"    力大小: {f_mag:.3f} N | 法线分量: {f_normal:.3f} N")

                # 将12维数据写入CSV文件
                row = [
                    f"{sim_time:.4f}", env_id, sensor_name, i, contact_idx,
                    # 位置 (3维)
                    f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                    # 四元数 (4维)
                    f"{quat[0]:.4f}", f"{quat[1]:.4f}", f"{quat[2]:.4f}", f"{quat[3]:.4f}",
                    # 力 (4维)
                    f"{f_vec[0]:.4f}", f"{f_vec[1]:.4f}", f"{f_vec[2]:.4f}", f"{f_mag:.4f}",
                    # 分析 (1维)
                    f"{f_normal:.4f}"
                ]
                self.csv_writer.writerow(row)

    def close(self):
        """Close CSV files."""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()


def main():
    # parse configuration
    env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0", num_envs=args_cli.num_envs)
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    
    # Enable contact sensors on the robot's gripper fingers
    # This is crucial for the contact sensors to work properly
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    
    # === 添加夹爪传感器pad ===
    # 左夹爪上的4个传感器pad - 创建为独立的rigid object
    for i in range(4):
        # 创建传感器pad
        pad_name = f"gripper_left_pad_{i+1}"
        setattr(env_cfg.scene, pad_name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{pad_name}",
            spawn=sim_utils.CuboidCfg(
                size=(0.008, 0.008, 0.002),  # 小的传感器pad：8mm x 8mm x 2mm
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True  # 运动学物体，位置将通过代码控制
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), metallic=0.3  # 绿色
                ),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),  # 初始位置，后续通过代码更新
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ))
        
        # 创建对应的接触传感器
        sensor_name = f"gripper_left_sensor_{i+1}"
        setattr(env_cfg.scene, sensor_name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{pad_name}",
            track_pose=True,
            update_period=0.0,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ))
    
    # 右夹爪上的4个传感器pad - 创建为独立的rigid object
    for i in range(4):
        # 创建传感器pad
        pad_name = f"gripper_right_pad_{i+1}"
        setattr(env_cfg.scene, pad_name, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{pad_name}",
            spawn=sim_utils.CuboidCfg(
                size=(0.008, 0.008, 0.002),  # 小的传感器pad：8mm x 8mm x 2mm
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True  # 运动学物体，位置将通过代码控制
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), metallic=0.3  # 绿色
                ),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.5),  # 初始位置，后续通过代码更新
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ))
        
        # 创建对应的接触传感器
        sensor_name = f"gripper_right_sensor_{i+1}"
        setattr(env_cfg.scene, sensor_name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{pad_name}",
            track_pose=True,
            update_period=0.0,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ))
    
    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    
    # 初始化传感器管理器
    sensor_manager = ContactSensorManager(env.unwrapped.num_envs, env.unwrapped.device)
    
    # 添加所有传感器到管理器
    for sensor_name, sensor_obj in env.unwrapped.scene.sensors.items():
        if "gripper_left_sensor_" in sensor_name or "gripper_right_sensor_" in sensor_name:
            sensor_manager.add_sensor(sensor_name, sensor_obj)

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )

    print("开始Franka抓取任务，夹爪传感器pad实时监控...")
    print(f"左夹爪传感器pad数量: 4")
    print(f"右夹爪传感器pad数量: 4")
    print(f"总传感器数量: 8")
    print(f"传感器pad排列: 正方形布局，边长 {PAD_POSITION_INFO['square_size']*1000:.1f}mm")
    print(f"传感器pad方向修正: 已修正为平行于黑色橡胶块表面")
    print(f"左夹爪传感器方向: 面向右侧（正Y方向），贴合黑色橡胶内表面")
    print(f"右夹爪传感器方向: 面向左侧（负Y方向），贴合黑色橡胶内表面")
    
    # 获取夹爪和传感器pad的引用
    robot = env.unwrapped.scene["robot"]
    sensor_pads = {}
    
    # 收集所有传感器pad的引用
    for i in range(4):
        left_pad_name = f"gripper_left_pad_{i+1}"
        right_pad_name = f"gripper_right_pad_{i+1}"
        
        if left_pad_name in env.unwrapped.scene.rigid_objects:
            sensor_pads[left_pad_name] = env.unwrapped.scene.rigid_objects[left_pad_name]
        if right_pad_name in env.unwrapped.scene.rigid_objects:
            sensor_pads[right_pad_name] = env.unwrapped.scene.rigid_objects[right_pad_name]
    
    print(f"创建的传感器pad数量: {len(sensor_pads)}")
    
    step_count = 0

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            step_count += 1
            
            # 更新传感器pad位置以跟随夹爪
            robot_data = robot.data
            
            # 获取左右夹爪的位置和方向
            left_finger_pos = robot_data.body_pos_w[:, robot_data.body_names.index("panda_leftfinger")]
            left_finger_quat = robot_data.body_quat_w[:, robot_data.body_names.index("panda_leftfinger")]
            right_finger_pos = robot_data.body_pos_w[:, robot_data.body_names.index("panda_rightfinger")]
            right_finger_quat = robot_data.body_quat_w[:, robot_data.body_names.index("panda_rightfinger")]
            
            # 更新左夹爪传感器pad位置 - 使用精确的正方形排列
            for i in range(4):
                pad_name = f"gripper_left_pad_{i+1}"
                if pad_name in sensor_pads:
                    pad = sensor_pads[pad_name]
                    
                    # 使用预定义的精确位置（黑色橡胶块上的正方形排列）
                    pad_pos = LEFT_FINGER_PAD_POSITIONS[i]
                    
                    # 定义局部坐标系中的偏移向量
                    local_offset = torch.tensor(pad_pos, 
                                               device=env.unwrapped.device, 
                                               dtype=torch.float32).unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                    
                    # 使用夹爪的四元数将局部偏移转换到世界坐标系
                    world_offset = quat_apply(left_finger_quat, local_offset)
                    
                    # 计算传感器pad的世界位置
                    pad_world_pos = left_finger_pos + world_offset
                    
                    # 使用新的方向计算函数获取正确的传感器pad方向
                    pad_world_quat = get_sensor_pad_orientation(left_finger_quat, is_left_finger=True)
                    
                    # 组合位置和方向为完整的状态
                    pad_root_state = torch.cat([pad_world_pos, pad_world_quat], dim=-1)
                    
                    # 更新传感器pad的状态
                    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                    pad.write_root_pose_to_sim(pad_root_state, env_ids)
                    
                    # 调试信息
                    if step_count % 60 == 0:  # 每秒打印一次
                        print(f"    左夹爪传感器pad_{i+1}: 位置 [{pad_world_pos[0,0]:.3f}, {pad_world_pos[0,1]:.3f}, {pad_world_pos[0,2]:.3f}], 方向 [{pad_world_quat[0,0]:.3f}, {pad_world_quat[0,1]:.3f}, {pad_world_quat[0,2]:.3f}, {pad_world_quat[0,3]:.3f}]")
            
            # 更新右夹爪传感器pad位置 - 使用精确的正方形排列
            for i in range(4):
                pad_name = f"gripper_right_pad_{i+1}"
                if pad_name in sensor_pads:
                    pad = sensor_pads[pad_name]
                    
                    # 使用预定义的精确位置（黑色橡胶块上的正方形排列）
                    pad_pos = RIGHT_FINGER_PAD_POSITIONS[i]
                    
                    # 定义局部坐标系中的偏移向量
                    local_offset = torch.tensor(pad_pos, 
                                               device=env.unwrapped.device, 
                                               dtype=torch.float32).unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                    
                    # 使用夹爪的四元数将局部偏移转换到世界坐标系
                    world_offset = quat_apply(right_finger_quat, local_offset)
                    
                    # 计算传感器pad的世界位置
                    pad_world_pos = right_finger_pos + world_offset
                    
                    # 使用新的方向计算函数获取正确的传感器pad方向
                    pad_world_quat = get_sensor_pad_orientation(right_finger_quat, is_left_finger=False)
                    
                    # 组合位置和方向为完整的状态
                    pad_root_state = torch.cat([pad_world_pos, pad_world_quat], dim=-1)
                    
                    # 更新传感器pad的状态
                    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                    pad.write_root_pose_to_sim(pad_root_state, env_ids)
                    
                    # 调试信息
                    if step_count % 60 == 0:  # 每秒打印一次
                        print(f"    右夹爪传感器pad_{i+1}: 位置 [{pad_world_pos[0,0]:.3f}, {pad_world_pos[0,1]:.3f}, {pad_world_pos[0,2]:.3f}], 方向 [{pad_world_quat[0,0]:.3f}, {pad_world_quat[0,1]:.3f}, {pad_world_quat[0,2]:.3f}, {pad_world_quat[0,3]:.3f}]")

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # 处理传感器数据 - 每10步处理一次以减少输出
            if step_count % 10 == 0:
                current_time = step_count * env_cfg.sim.dt * env_cfg.decimation
                print(f"\n=== 步数 {step_count}, 时间 {current_time:.2f}s ===")
                # 处理所有环境的传感器数据
                for env_id in range(env.unwrapped.num_envs):
                    sensor_manager.process_sensor_data(env_id, current_time)

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # 关闭传感器管理器
    sensor_manager.close()
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 
