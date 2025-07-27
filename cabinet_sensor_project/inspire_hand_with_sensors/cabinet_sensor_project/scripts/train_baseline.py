#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
🤖 Cabinet RL Training - BASELINE Version (No Sensors)

基线版本的机械臂开抽屉强化学习任务，不使用接触传感器。
作为对照组，与cabinet_rl_with_sensors_new.py进行严格的控制变量对比。

Usage:
    ./isaaclab.sh -p cabinet_rl_BASELINE.py --num_envs 64 --headless
"""

import argparse
import os
import torch
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train RL agent for cabinet opening task - BASELINE (No Sensors)")
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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle


def main():
    """Main function."""
    
    # parse configuration - 与传感器版本完全相同的方法
    env_cfg: CabinetEnvCfg = parse_env_cfg(
        "Isaac-Open-Drawer-Franka-IK-Abs-v0",
        device="cuda:0",
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 8,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # 🚫 不添加传感器配置 - 这是与传感器版本的唯一区别
    # 注意：这里故意不调用patch_env_cfg_with_contact_sensors(env_cfg)
    print("[BASELINE] 🔍 No sensors configured - using standard observations only")
    
    # 🚫 不添加传感器数据到观测中 - 保持标准观测空间
    # 注意：这里故意不添加 env_cfg.observations.policy.contact_forces
    
    # Load agent configuration - 与传感器版本完全相同
    from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg
    agent_cfg: RslRlOnPolicyRunnerCfg = CabinetPPORunnerCfg()

    # Override max_iterations if provided
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
        print(f"[INFO] Using command line max_iterations: {args_cli.max_iterations}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "cabinet_baseline")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.seed is not None:
        log_dir += f"_seed{args_cli.seed}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment - 与传感器版本相同的任务名
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
    
    # 🔍 验证观测空间 - 基线版本应该是标准维度
    print(f"\n🔍 [VERIFICATION] Baseline observation space analysis:")
    expected_baseline_obs = 31  # 基础观测维度 (without sensors)
    print(f"  - Expected baseline obs dimensions: {expected_baseline_obs}")
    print(f"  - Actual obs dimensions: {env.num_obs}")
    
    if env.num_obs == expected_baseline_obs:
        print(f"  ✅ SUCCESS: Baseline using standard observations (no sensors)")
    else:
        print(f"  ⚠️  WARNING: Unexpected observation dimensions for baseline")
    
    # 🔍 测试观测访问
    print(f"\n🔍 [VERIFICATION] Testing baseline observations:")
    try:
        test_obs = env.get_observations()
        print(f"  - Successfully got observations: {test_obs[0].shape}")
        print(f"  - Baseline obs (env 0, first 10 dims): {test_obs[0][:10].cpu().numpy()}")
        
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

    print("[INFO] 🚀 Starting BASELINE training (no sensors)...")
    print(f"[INFO] Total training steps: {agent_cfg.max_iterations}")
    print(f"[INFO] Environment episodes per step: {env_cfg.scene.num_envs}")
    print(f"[INFO] Using standard observations only (NO sensor data)")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("[INFO] ✅ BASELINE training completed!")

    # save the final model
    save_path = os.path.join(log_dir, "model_{}.pt".format(runner.current_learning_iteration))
    runner.save(save_path)
    print(f"[INFO] Saved baseline model to {save_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 