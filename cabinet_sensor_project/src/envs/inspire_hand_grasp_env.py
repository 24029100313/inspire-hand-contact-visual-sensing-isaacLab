"""
Inspire Hand Grasping Environment with MediaPipe Control

This environment implements a grasping task where the Inspire Hand with tactile sensors
attempts to grasp a cube. The hand can be controlled through MediaPipe hand tracking.
"""

import torch
import numpy as np
import cv2
import time
import threading
from typing import Dict, Any, Tuple, Optional

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, CameraCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.actuators import ImplicitActuatorCfg

# Import our custom modules for MediaPipe control
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../inspire_hand_clone"))
from mp_read_hand import HandDetector


@configclass
class InspireHandGraspEnvCfg(DirectRLEnvCfg):
    """Configuration for the Inspire Hand grasping environment."""
    
    # Environment settings
    episode_length_s = 30.0
    decimation = 2
    num_actions = 6  # 6 DOF for the hand
    num_observations = 50
    num_states = 0
    
    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=True
    )


class InspireHandGraspEnv(DirectRLEnv):
    """Inspire Hand grasping environment with MediaPipe control support."""
    
    cfg: InspireHandGraspEnvCfg
    
    def __init__(self, cfg: InspireHandGraspEnvCfg, render_mode: str = None, **kwargs):
        """Initialize the Inspire Hand grasping environment."""
        
        # Set up the hand asset configuration
        self._configure_assets(cfg)
        
        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize MediaPipe components
        self.hand_detector = None
        self.mediapipe_enabled = False
        self.latest_hand_actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Hand control setup
        self._setup_action_scaling()
        
    def _configure_assets(self, cfg: InspireHandGraspEnvCfg):
        """Configure the hand and cube assets."""
        
        # Hand USD path
        hand_usd_path = os.path.abspath(
            "inspire_hand_with_sensors/usd/inspire_hand_processed_with_pads.usd"
        )
        
        # Hand configuration
        cfg.hand = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=hand_usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.1,
                    angular_damping=0.1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.1),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "index_1_joint": 0.0,
                    "middle_1_joint": 0.0,
                    "ring_1_joint": 0.0,
                    "little_1_joint": 0.0,
                    "thumb_1_joint": 0.0,
                    "thumb_swing_joint": 0.5,
                },
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*_1_joint", ".*_swing_joint"],
                    effort_limit=5.0,
                    velocity_limit=2.0,
                    stiffness=100.0,
                    damping=10.0,
                ),
            },
        )
        
        # Cube configuration
        cfg.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(mass=0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.3, 0.3)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.1, 0.0, 0.15),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        
        # Contact sensors
        cfg.contact_sensors = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
        )
        
    def _setup_action_scaling(self):
        """Set up action scaling from 0-1000 to radians."""
        self.action_scale = torch.tensor([
            [0.0, 1.57],   # index: 0-90 degrees
            [0.0, 1.57],   # middle: 0-90 degrees  
            [0.0, 1.57],   # ring: 0-90 degrees
            [0.0, 1.57],   # little: 0-90 degrees
            [-0.23, 1.22], # thumb: -13 to 70 degrees
            [1.57, 2.88],  # thumb_swing: 90 to 165 degrees
        ], device=self.device)
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        self.hand = self.scene.add(self.cfg.hand)
        self.cube = self.scene.add(self.cfg.cube)
        self.contact_sensors = self.scene.add(self.cfg.contact_sensors)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        if self.mediapipe_enabled:
            actions = self.latest_hand_actions.clone()
        
        scaled_actions = self._scale_actions(actions)
        self.hand.set_joint_position_target(scaled_actions)
        
    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions from 0-1000 range to joint angles."""
        actions = torch.clamp(actions, 0.0, 1000.0)
        normalized = actions / 1000.0
        
        min_vals = self.action_scale[:, 0].unsqueeze(0)
        max_vals = self.action_scale[:, 1].unsqueeze(0)
        
        return min_vals + normalized * (max_vals - min_vals)
        
    def _get_observations(self) -> dict:
        """Get environment observations."""
        hand_pos = self.hand.data.joint_pos
        hand_vel = self.hand.data.joint_vel
        cube_pos = self.cube.data.pos_w
        cube_rot = self.cube.data.quat_w
        
        obs = torch.cat([
            hand_pos.flatten(start_dim=1),
            hand_vel.flatten(start_dim=1),
            cube_pos,
            cube_rot,
        ], dim=1)
        
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards."""
        hand_palm_pos = self.hand.data.body_pos_w[:, 0, :]
        cube_pos = self.cube.data.pos_w
        distance = torch.norm(cube_pos - hand_palm_pos, dim=1)
        
        return -distance * 10.0
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cube_fell = self.cube.data.pos_w[:, 2] < 0.05
        
        return cube_fell, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments."""
        if len(env_ids) == 0:
            return
            
        self.hand.reset(env_ids)
        self.cube.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        
    def enable_mediapipe_control(self, camera_id: int = 0):
        """Enable MediaPipe hand tracking."""
        try:
            self.hand_detector = HandDetector(target_hand='right')
            self.mediapipe_enabled = True
            print("MediaPipe control enabled!")
            return True
        except Exception as e:
            print(f"MediaPipe error: {e}")
            return False


def create_inspire_hand_env(num_envs: int = 1, enable_mediapipe: bool = False):
    """Create environment."""
    cfg = InspireHandGraspEnvCfg()
    cfg.scene.num_envs = num_envs
    
    env = InspireHandGraspEnv(cfg)
    
    if enable_mediapipe:
        env.enable_mediapipe_control()
    
    return env


if __name__ == "__main__":
    print("Creating Inspire Hand environment...")
    env = create_inspire_hand_env(num_envs=1, enable_mediapipe=True)
    print("Environment created successfully!")
