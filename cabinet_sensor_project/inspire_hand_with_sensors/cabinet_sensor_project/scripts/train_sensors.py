#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
🤖 Cabinet RL Training - WITH SENSORS (New Version)

完全基于成功的open_cabinet_sm.py传感器配置方法！
传感器数据将被正确集成到observation中用于强化学习训练。

Usage:
    ./isaaclab.sh -p cabinet_rl_with_sensors_new.py --num_envs 64 --headless
"""

import argparse
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train RL agent for cabinet opening task - WITH SENSORS (New)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of training iterations")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle


def contact_sensor_data(env, asset_cfg=None) -> torch.Tensor:
    """Contact sensor data from fingertip sensors.
    
    完全基于open_cabinet_sm.py的成功方法来获取传感器数据！
    
    Returns:
        torch.Tensor: Contact sensor observations with shape (num_envs, 12)
        - [0:3]: Left finger contact forces (x, y, z) - normalized
        - [3]: Left finger contact magnitude (normalized)
        - [4]: Left finger contact binary indicator
        - [5:8]: Right finger contact forces (x, y, z) - normalized  
        - [8]: Right finger contact magnitude (normalized)
        - [9]: Right finger contact binary indicator
        - [10:11]: Reserved for future sensor data
    """
    device = env.device
    num_envs = env.num_envs
    
    # Initialize sensor observations (12 dimensions)
    sensor_obs = torch.zeros((num_envs, 12), device=device)
    
    # 🔍 调试计数器
    if not hasattr(contact_sensor_data, 'debug_counter'):
        contact_sensor_data.debug_counter = 0
    contact_sensor_data.debug_counter += 1
    
    try:
        # 🔥 使用与open_cabinet_sm.py完全相同的访问方式
        left_contact = env.unwrapped.scene["left_finger_contact"].data.net_forces_w
        right_contact = env.unwrapped.scene["right_finger_contact"].data.net_forces_w
        
        # 🔍 调试输出 - 每100步输出一次
        if contact_sensor_data.debug_counter % 100 == 1:
            print(f"\n🔍 [DEBUG] Sensor data check (step {contact_sensor_data.debug_counter}):")
            print(f"  - Left sensor shape: {left_contact.shape}")
            print(f"  - Right sensor shape: {right_contact.shape}")
            if left_contact.numel() > 0:
                left_sample = left_contact[0, -1 if left_contact.dim() == 3 else 0, :] if left_contact.dim() >= 2 else left_contact[0]
                print(f"  - Left forces (env 0): {left_sample.cpu().numpy()}")
            if right_contact.numel() > 0:
                right_sample = right_contact[0, -1 if right_contact.dim() == 3 else 0, :] if right_contact.dim() >= 2 else right_contact[0]
                print(f"  - Right forces (env 0): {right_sample.cpu().numpy()}")
        
        # Process left finger sensor data
        if left_contact.numel() > 0:
            # Handle different data formats (same as before)
            if left_contact.dim() == 3:
                # Shape: (num_envs, history_length, 3)
                left_force_current = left_contact[:, -1, :]
            elif left_contact.dim() == 2:
                # Shape: (num_envs, 3)
                left_force_current = left_contact
            else:
                left_force_current = torch.zeros((num_envs, 3), device=device)
            
            # Calculate force magnitude
            left_magnitude = torch.norm(left_force_current, dim=1, keepdim=True)
            
            # Normalize forces (avoid division by zero)
            left_forces_norm = torch.where(
                left_magnitude > 1e-6,
                left_force_current / (left_magnitude + 1e-6),
                torch.zeros_like(left_force_current)
            )
            
            # Binary contact indicator (threshold: 0.1N)
            left_contact_binary = (left_magnitude > 0.1).float()
            
            # Store in observation
            sensor_obs[:, 0:3] = left_forces_norm
            sensor_obs[:, 3:4] = torch.clamp(left_magnitude / 50.0, 0, 1)  # Normalized magnitude
            sensor_obs[:, 4:5] = left_contact_binary
        
        # Process right finger sensor data
        if right_contact.numel() > 0:
            # Handle different data formats
            if right_contact.dim() == 3:
                # Shape: (num_envs, history_length, 3)
                right_force_current = right_contact[:, -1, :]
            elif right_contact.dim() == 2:
                # Shape: (num_envs, 3)
                right_force_current = right_contact
            else:
                right_force_current = torch.zeros((num_envs, 3), device=device)
            
            # Calculate force magnitude
            right_magnitude = torch.norm(right_force_current, dim=1, keepdim=True)
            
            # Normalize forces (avoid division by zero)
            right_forces_norm = torch.where(
                right_magnitude > 1e-6,
                right_force_current / (right_magnitude + 1e-6),
                torch.zeros_like(right_force_current)
            )
            
            # Binary contact indicator (threshold: 0.1N)
            right_contact_binary = (right_magnitude > 0.1).float()
            
            # Store in observation
            sensor_obs[:, 5:8] = right_forces_norm
            sensor_obs[:, 8:9] = torch.clamp(right_magnitude / 50.0, 0, 1)  # Normalized magnitude
            sensor_obs[:, 9:10] = right_contact_binary
        
        # 🔍 调试输出 - 显示最终传感器观测
        if contact_sensor_data.debug_counter % 100 == 1:
            print(f"  - Final sensor obs shape: {sensor_obs.shape}")
            print(f"  - Sensor obs (env 0): {sensor_obs[0].cpu().numpy()}")
            non_zero_mask = torch.abs(sensor_obs[0]) > 1e-6
            if non_zero_mask.any():
                print(f"  - Non-zero elements: {torch.where(non_zero_mask)[0].cpu().numpy()}")
            else:
                print(f"  - All sensor values are zero (no contact)")
                
    except Exception as e:
        print(f"[ERROR] Contact sensor data error: {e}")
        if contact_sensor_data.debug_counter % 100 == 1:
            print(f"  - Returning zero sensor observations due to error")
        
    return sensor_obs


def patch_env_cfg_with_contact_sensors(env_cfg):
    """给env_cfg注入两个夹爪的ContactSensor配置（与open_cabinet_sm.py完全相同）"""
    print("[SENSOR CONFIG] 🚀 Using EXACT same sensor config as open_cabinet_sm.py!")
    
    env_cfg.scene.left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )
    env_cfg.scene.right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )
    
    print("[SENSOR CONFIG] ✅ Contact sensors configured:")
    print(f"  - Left finger sensor: {env_cfg.scene.left_finger_contact.prim_path}")
    print(f"  - Right finger sensor: {env_cfg.scene.right_finger_contact.prim_path}")
    print(f"  - Visualization: {'ENABLED' if env_cfg.scene.left_finger_contact.debug_vis else 'DISABLED'}")


def main():
    """Main function."""
    
    # parse configuration - 与open_cabinet_sm.py完全相同的方法
    env_cfg: CabinetEnvCfg = parse_env_cfg(
        "Isaac-Open-Drawer-Franka-IK-Abs-v0",
        device="cuda:0",
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 8,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Apply sensor patch - 与open_cabinet_sm.py完全相同
    patch_env_cfg_with_contact_sensors(env_cfg)
    
    # 🔥 关键：添加传感器数据到观测中
    env_cfg.observations.policy.contact_forces = ObsTerm(func=contact_sensor_data)
    
    # Optional: Add to critic observations as well
    if hasattr(env_cfg.observations, 'critic'):
        env_cfg.observations.critic.contact_forces = ObsTerm(func=contact_sensor_data)
    
    # Load agent configuration
    from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = CabinetPPORunnerCfg()

    # Override max_iterations if provided
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
        print(f"[INFO] Using command line max_iterations: {args_cli.max_iterations}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "cabinet_with_sensors_new")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.seed is not None:
        log_dir += f"_seed{args_cli.seed}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment - 与open_cabinet_sm.py相同的任务名
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Abs-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "episode_trigger": lambda step: step % args_cli.video_interval == 0,
            "step_trigger": None,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO] Environment observation space: {env.num_obs}")
    print(f"[INFO] Environment action space: {env.num_actions}")
    print(f"[INFO] Environment episode length: {env.max_episode_length}")
    
    # 🔍 验证观测空间是否包含传感器数据
    print(f"\n🔍 [VERIFICATION] Observation space analysis:")
    original_obs_without_sensors = 31  # 基础观测维度 (9+9+1+1+3+8 = 31)
    expected_obs_with_sensors = original_obs_without_sensors + 12  # 加上12维传感器数据
    print(f"  - Expected obs dimensions (without sensors): {original_obs_without_sensors}")
    print(f"  - Expected obs dimensions (with sensors): {expected_obs_with_sensors}")
    print(f"  - Actual obs dimensions: {env.num_obs}")
    
    if env.num_obs == expected_obs_with_sensors:
        print(f"  ✅ SUCCESS: Sensor data (12 dims) successfully added to observations!")
    elif env.num_obs == original_obs_without_sensors:
        print(f"  ❌ WARNING: Sensor data NOT added to observations!")
    else:
        print(f"  ⚠️  UNKNOWN: Unexpected observation dimensions")
    
    # 🔍 测试传感器访问
    print(f"\n🔍 [VERIFICATION] Testing sensor access:")
    try:
        test_obs = env.get_observations()
        print(f"  - Successfully got observations: {test_obs[0].shape}")
        
        # 检查传感器部分
        if env.num_obs >= expected_obs_with_sensors:
            sensor_part = test_obs[0][-12:]  # 最后12维应该是传感器数据
            print(f"  - Sensor part shape: {sensor_part.shape}")
            print(f"  - Sensor values: {sensor_part}")
            if torch.all(sensor_part == 0):
                print(f"  ⚠️  Sensor values are all zero (expected at start)")
            else:
                print(f"  ✅ Sensor values contain non-zero data")
        
    except Exception as e:
        print(f"  ❌ Error accessing observations: {e}")

    # set seed of the environment
    seed_value = args_cli.seed if args_cli.seed is not None else 42
    env.seed(seed_value)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    print("[INFO] 🚀 Starting training with contact sensors (EXACT same config as open_cabinet_sm.py)...")
    print(f"[INFO] Total training steps: {agent_cfg.max_iterations}")
    print(f"[INFO] Environment episodes per step: {env_cfg.scene.num_envs}")
    print(f"[INFO] Sensor data will be included in observations (12 dimensions)")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("[INFO] ✅ Training completed with contact sensors!")

    # save the final model
    save_path = os.path.join(log_dir, "model_{}.pt".format(runner.current_learning_iteration))
    runner.save(save_path)
    print(f"[INFO] Saved model to {save_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 