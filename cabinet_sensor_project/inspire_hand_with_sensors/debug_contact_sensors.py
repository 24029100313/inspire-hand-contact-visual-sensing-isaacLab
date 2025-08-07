#!/usr/bin/env python3
"""
Debug Contact Sensors - Simplified test to diagnose sensor issues
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Debug contact sensors")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    """Simplified scene for debugging."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # Hand configuration - simplified
    hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_specific_pads.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.1, -0.045, 0.12),
            rot=(0.754649, -0.655439, 0.029940, 0.002793),
            joint_pos={
                "right_thumb_1_joint": 0.0,
                "right_thumb_2_joint": 0.0,
                "right_thumb_3_joint": 0.0,
                "right_thumb_4_joint": 0.0,
                "right_index_1_joint": 0.0,
                "right_index_2_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_middle_2_joint": 0.0,
                "right_ring_1_joint": 0.0,
                "right_ring_2_joint": 0.0,
                "right_little_1_joint": 0.0,
                "right_little_2_joint": 0.0,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit=10.0,
                velocity_limit=2.0,
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )
    
    # Cube configuration
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.07, 0.144, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Simplified contact sensor - test one sensor first
    contact_sensor_debug: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/.*",  # Monitor all hand bodies
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        # No filter - detect all contacts
    )

@configclass
class DebugEnvCfg(DirectRLEnvCfg):
    """Debug environment configuration."""
    
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene: DebugSceneCfg = DebugSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    episode_length_s = 30.0
    decimation = 4
    num_actions = 12
    num_observations = 50
    num_states = 0
    observation_space = 50
    action_space = 12

class DebugEnv(DirectRLEnv):
    """Simplified debug environment."""
    
    cfg: DebugEnvCfg
    
    def __init__(self, cfg: DebugEnvCfg, render_mode: str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        if not args_cli.headless:
            set_camera_view([0.5, 0.5, 0.5], [0.0, 0.0, 0.1])
        
        self.step_count = 0
        print("🔧 Debug Environment Initialized")
        
    def _setup_scene(self):
        """Set up debug scene."""
        self.hand = self.scene["hand"]
        self.cube = self.scene["cube"]
        self.contact_sensor = self.scene["contact_sensor_debug"]
        
        print(f"📍 Debug setup complete:")
        print(f"   Hand: {type(self.hand).__name__}")
        print(f"   Cube: {type(self.cube).__name__}")
        print(f"   Contact sensor: {type(self.contact_sensor).__name__}")
        
        # Try to get sensor information safely
        try:
            if hasattr(self.contact_sensor, 'data'):
                print(f"   Contact sensor has data attribute")
            else:
                print(f"   Contact sensor missing data attribute")
        except Exception as e:
            print(f"   Contact sensor info error: {e}")
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply simple closing motion."""
        # Simple test: gradually close index finger
        progress = min(self.step_count / 600.0, 1.0)  # 5 seconds at 120Hz
        
        target_positions = torch.zeros((self.num_envs, 12), device=self.device)
        # Only move index finger for testing
        target_positions[:, 4] = progress * 1.0  # index_1
        target_positions[:, 5] = progress * 1.5  # index_2
        
        self.hand.set_joint_position_target(target_positions)
        
    def _apply_action(self) -> None:
        pass
        
    def _get_observations(self) -> dict:
        obs = torch.zeros((self.num_envs, self.cfg.num_observations), device=self.device)
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
        
    def _get_dones(self) -> tuple:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor):
        self.hand.reset(env_ids)
        self.cube.reset(env_ids)
        if len(env_ids) > 0:
            self.step_count = 0
            
    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.step_count += 1
        
        # Debug contact data every 60 steps (0.5s)
        if self.step_count % 60 == 0:
            self._debug_contacts()
            
        return obs, reward, terminated, truncated, info
        
    def _debug_contacts(self):
        """Debug contact sensor data."""
        print(f"\n🔍 Debug Step {self.step_count} (t={self.step_count/120:.1f}s)")
        print("-" * 50)
        
        # Get contact data
        try:
            net_forces = self.contact_sensor.data.net_forces_w
            force_matrix = self.contact_sensor.data.force_matrix_w
            
            print(f"📊 Contact sensor data shape:")
            print(f"   net_forces: {net_forces.shape}")
            print(f"   force_matrix: {force_matrix.shape}")
            
            # Show any non-zero forces
            total_forces = torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)
            non_zero_forces = total_forces > 0.001  # 1mN threshold
            
            if torch.any(non_zero_forces):
                print(f"✅ Detected contacts:")
                env_idx, body_idx = torch.where(non_zero_forces)
                for i in range(len(env_idx)):
                    e, b = env_idx[i].item(), body_idx[i].item()
                    force_mag = total_forces[e, b].item()
                    force_vec = net_forces[e, b]
                    print(f"   Env {e}, Body {b}: {force_mag:.6f} N, Vec: [{force_vec[0]:.3f}, {force_vec[1]:.3f}, {force_vec[2]:.3f}]")
            else:
                print("⭕ No contacts detected")
                
            # Show max force for reference
            max_force = torch.max(total_forces).item()
            print(f"📈 Maximum force magnitude: {max_force:.6f} N")
            
        except Exception as e:
            print(f"❌ Error reading contact data: {e}")
            
        # Show hand joint positions
        try:
            joint_pos = self.hand.data.joint_pos[0]  # First environment
            print(f"🦾 Joint positions (rad):")
            joint_names = ["thumb_1", "thumb_2", "thumb_3", "thumb_4", "index_1", "index_2", 
                          "middle_1", "middle_2", "ring_1", "ring_2", "little_1", "little_2"]
            for i, (name, pos) in enumerate(zip(joint_names, joint_pos)):
                if i < 6:  # Only show first 6 joints
                    print(f"   {name}: {pos:.3f}")
        except Exception as e:
            print(f"❌ Error reading joint data: {e}")

def main():
    """Run debug test."""
    env_cfg = DebugEnvCfg()
    env = DebugEnv(cfg=env_cfg, render_mode="human" if not args_cli.headless else None)
    
    print("\n" + "="*60)
    print("🔍 CONTACT SENSOR DEBUG TEST")
    print("="*60)
    print("📋 This test will:")
    print("   1. Gradually close the index finger")
    print("   2. Monitor ALL contact forces")
    print("   3. Show detailed sensor data every 0.5s")
    print("   4. Help diagnose sensor issues")
    print("="*60 + "\n")
    
    count = 0
    while simulation_app.is_running() and count < 1200:  # 10 seconds
        with torch.inference_mode():
            actions = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
            env.step(actions)
            
        if env.unwrapped.sim.is_stopped():
            break
            
        count += 1
        
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close() 