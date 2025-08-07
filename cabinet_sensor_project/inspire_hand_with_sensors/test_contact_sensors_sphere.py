#!/usr/bin/env python3
"""
Test Contact Sensors for Inspire Hand with Sphere
(Based on test_contact_sensors_specific_pads.py but using sphere instead of cube)

This script creates a test environment where the Inspire Hand grasps a sphere
and monitors the contact forces on the 10 specific sensor pads.

Usage:
    ./isaaclab.sh -p test_contact_sensors_sphere.py --num_envs 1
"""

import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test contact sensors on Inspire Hand with sphere")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O operations")
parser.add_argument("--total_time", type=float, default=90.0, help="Total runtime in seconds (default: 90s)")
# Note: --headless is automatically added by AppLauncher

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else follows."""

import torch
import numpy as np
import carb

# Isaac Sim imports
from isaacsim.core.utils.viewports import set_camera_view

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene Configuration
##

@configclass
class InspireHandContactTestSceneCfg(InteractiveSceneCfg):
    """Configuration for the Inspire Hand contact test scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # Desktop table surface for placing objects
    desktop: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desktop",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),  # 120cm x 80cm x 5cm thick table
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # Table is fixed in place
                kinematic_enabled=True,  # Table won't move when touched
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=50.0,  # Heavy table
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.7, 0.6),  # Wood color
                roughness=0.3,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # Table surface at ground level
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", 
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    )
    
    # Hand configuration
    hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_specific_pads.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # Disable self-collisions for stability
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.1, -0.045, 0.12),  # Hand position at specified coordinates
            rot=(0.754649, -0.655439, 0.029940, 0.002793),  # Hand palm facing down (-82°, 2.8°, -2.01°)
            joint_pos={
                # Initial positions using Inspire Hand standard (converted to radians)
                # Ring fingers set to slight extension (50/1000 and 100/1000) to avoid drooping
                "right_thumb_1_joint": 0.0,
                "right_thumb_2_joint": 0.0,
                "right_thumb_3_joint": 0.0,
                "right_thumb_4_joint": 0.0,
                "right_index_1_joint": 0.0,
                "right_index_2_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_middle_2_joint": 0.0,
                "right_ring_1_joint": 0.072,  # 50/1000 * 1.44 = 0.072 radians (slight extension)
                "right_ring_2_joint": 0.314,  # 100/1000 * 3.14 = 0.314 radians (slight extension)
                "right_little_1_joint": 0.0,
                "right_little_2_joint": 0.0,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit=15.0,  # Increased effort limit
                velocity_limit=2.0,
                stiffness=200.0,  # Increased stiffness to resist gravity
                damping=20.0,  # Increased damping for better stability
            ),
        },
    )
    
    # Sphere configuration - positioned between thumb and index finger
    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.05,  # 5cm radius sphere (10cm diameter)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # Enable gravity so sphere sits naturally on table
                kinematic_enabled=False,  # Allow sphere to be moved by physics
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.1,  # 100g (same as original cube)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),  # Green color to distinguish from cube
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Position based on URDF analysis:
            # Thumb root: (-0.027, 0.021, 0.069)
            # Index root: (-0.039, 0.0006, 0.156)
            # Place sphere on desktop surface for realistic grasping
            # Desktop surface at z=0.025, sphere radius=0.05, so center at z=0.075
            pos=(0.068, 0.12, 0.075),  # Updated position with calculated z coordinate for 5cm sphere
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Contact sensors - use dynamic configuration like reference implementation
    # We'll add these dynamically in main() function to match the working pattern


##
# Environment Configuration
##

@configclass
class InspireHandContactTestCfg(DirectRLEnvCfg):
    """Configuration for testing contact sensors on Inspire Hand."""
    
    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,  # 120Hz for better contact detection
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # Use TGS solver for better contact stability
            min_position_iteration_count=4,
            max_position_iteration_count=16,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=4,
        ),
    )
    
    # Scene settings
    scene: InspireHandContactTestSceneCfg = InspireHandContactTestSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=True
    )
    
    # Environment settings
    episode_length_s = 30.0  # Each cycle is 30 seconds: open -> close -> hold -> open
    decimation = 4  # env_dt = sim_dt * decimation = 1/120 * 4 = 1/30s
    num_actions = 6  # Control 6 main joints
    num_observations = 50
    num_states = 0
    
    # Add required spaces
    observation_space = 50  # Should match num_observations
    action_space = 6  # Should match num_actions


##
# Environment implementation
##

class InspireHandContactTestEnv(DirectRLEnv):
    """Test environment for Inspire Hand contact sensors with sphere."""

    cfg: InspireHandContactTestCfg

    def __init__(self, cfg: InspireHandContactTestCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        
        # Define the sensor pad names we want to monitor BEFORE calling super().__init__
        # (since _setup_scene is called during super().__init__)
        self.sensor_pad_names = [
            "thumb_force_sensor_1", "thumb_force_sensor_2", "thumb_force_sensor_3", "thumb_force_sensor_4",
            "index_force_sensor_1", "index_force_sensor_2", "index_force_sensor_3",
            "middle_force_sensor_1", "middle_force_sensor_2", "middle_force_sensor_3"
        ]
        
        print(f"[INFO] Initializing InspireHandContactTestEnv with {len(self.sensor_pad_names)} sensor pads")
        print(f"[INFO] Sensor pads: {self.sensor_pad_names}")
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Joint position targets (control commands)
        self._joint_pos_target = torch.zeros(self.num_envs, self.hand.num_joints, device=self.device)
        
        # Grasp motion state machine
        self.grasp_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.state_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Grasp phases: 0=Open, 1=Closing, 2=Holding, 3=Opening
        self.phase_durations = torch.tensor([5.0, 10.0, 10.0, 5.0], device=self.device)  # seconds
        
        # Step counter for periodic logging
        self.step_count = 0
        
        print(f"[INFO] Environment initialized successfully")

    def _setup_scene(self):
        """Setup the scene with hand, sphere, and contact sensors."""
        self.hand = self.scene["hand"]
        self.sphere = self.scene["sphere"]  # Changed from cube to sphere
        self.desktop = self.scene["desktop"]

        # Setup contact sensors - following the dynamic pattern from reference implementation
        print(f"[INFO] Setting up {len(self.sensor_pad_names)} contact sensors...")
        
        sensor_count = 0
        for sensor_name in self.sensor_pad_names:
            try:
                sensor = self.scene[sensor_name]
                sensor_count += 1
                print(f"[INFO] ✓ Sensor '{sensor_name}' configured successfully")
            except KeyError:
                print(f"[WARN] ✗ Sensor '{sensor_name}' not found in scene")
        
        print(f"[INFO] Successfully configured {sensor_count}/{len(self.sensor_pad_names)} sensors")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step."""
        # Update state machine
        self._update_grasp_state()
        
        # Get target joint positions based on current state
        target_positions = self._get_joint_targets()
        self._joint_pos_target[:] = target_positions
        
        # Apply actions to the hand
        self.hand.set_joint_position_target(self._joint_pos_target)

    def _apply_action(self) -> None:
        """Apply the computed actions to the environment actors."""
        pass  # Actions are applied in _pre_physics_step

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        obs = torch.cat([
            self.hand.data.joint_pos,
            self.hand.data.joint_vel,
            self.sphere.data.root_pos_w[:, :3],  # Changed from cube to sphere
            self.sphere.data.root_quat_w,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards. For testing, return zero."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get episode termination and truncation status."""
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at given indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset actors
        super()._reset_idx(env_ids)
        
        # Reset hand to default joint positions
        default_joint_pos = torch.zeros(len(env_ids), self.hand.num_joints, device=self.device)
        
        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_vel = self.hand.data.default_joint_vel[env_ids].clone()
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Reset sphere position (changed from cube)
        self.sphere.reset(env_ids)
        
        # Reset desktop
        self.desktop.reset(env_ids)
        
        # Reset state machine
        self.grasp_state[env_ids] = 0
        self.state_timer[env_ids] = 0.0
        
        # Set initial joint targets
        self._joint_pos_target[env_ids] = default_joint_pos

    def _update_grasp_state(self):
        """Update the grasp state machine."""
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.state_timer += dt
        
        # Check for state transitions
        current_phase_duration = self.phase_durations[self.grasp_state]
        transition_mask = self.state_timer >= current_phase_duration
        
        # Update state
        self.grasp_state[transition_mask] = (self.grasp_state[transition_mask] + 1) % 4
        self.state_timer[transition_mask] = 0.0

    def _convert_to_radians(self, inspire_values: torch.Tensor) -> torch.Tensor:
        """Convert Inspire Hand values (0-1000) to radians.
        
        The Inspire Hand uses 0-1000 range where:
        - 0 = fully open
        - 1000 = fully closed
        
        We need to map this to appropriate radian values for each joint.
        Different joints have different ranges based on the URDF limits.
        """
        # Define the max angles for each joint (in radians)
        # These are approximate values based on typical finger joint ranges
        max_angles = torch.tensor([
            1.16,  # thumb_1: ~66 degrees
            0.58,  # thumb_2: ~33 degrees
            0.50,  # thumb_3: ~28 degrees
            3.14,  # thumb_4: ~180 degrees (but usually limited in practice)
            1.44,  # index_1: ~82 degrees
            3.14,  # index_2: ~180 degrees (but usually limited)
            1.44,  # middle_1: ~82 degrees
            3.14,  # middle_2: ~180 degrees
            1.44,  # ring_1: ~82 degrees
            3.14,  # ring_2: ~180 degrees
            1.44,  # little_1: ~82 degrees
            3.14,  # little_2: ~180 degrees
        ], device=self.device)
        
        # Convert 0-1000 to 0-1 normalized, then scale by max angle
        normalized = inspire_values / 1000.0
        radians = normalized * max_angles.unsqueeze(0)
        
        return radians

    def _get_joint_targets(self) -> torch.Tensor:
        """Get joint targets based on current grasp state using Inspire Hand 0-1000 range."""
        # Initialize target positions in Inspire Hand range (0-1000)
        inspire_pos = torch.zeros((self.num_envs, 12), device=self.device)
        
        for env_idx in range(self.num_envs):
            state = self.grasp_state[env_idx].item()
            
            if state == 0:  # Open
                # All joints at 0 (open position) except ring fingers
                inspire_pos[env_idx] = 0.0
                # Keep ring fingers slightly extended to avoid drooping
                inspire_pos[env_idx, 8] = 50   # ring_1: slight extension (50/1000)
                inspire_pos[env_idx, 9] = 100  # ring_2: slight extension (100/1000)
                
            elif state == 1:  # Closing
                # Gradual closing
                progress = self.state_timer[env_idx] / self.phase_durations[1]
                
                # Thumb joints - keep straight (not participating in grasp)
                inspire_pos[env_idx, 0] = 0  # thumb_1: stay straight
                inspire_pos[env_idx, 1] = 0  # thumb_2: stay straight
                inspire_pos[env_idx, 2] = 0  # thumb_3: stay straight
                inspire_pos[env_idx, 3] = 0  # thumb_4: stay straight
                
                # Index finger - main grasping finger
                inspire_pos[env_idx, 4] = 950 * progress  # index_1 (increased from 800)
                inspire_pos[env_idx, 5] = 1000 * progress  # index_2
                
                # Middle finger - support grasp
                inspire_pos[env_idx, 6] = 700 * progress  # middle_1
                inspire_pos[env_idx, 7] = 900 * progress  # middle_2
                
                # Ring finger - controlled extension to avoid drooping
                inspire_pos[env_idx, 8] = 50 + (600 - 50) * progress  # ring_1: 50->600
                inspire_pos[env_idx, 9] = 100 + (800 - 100) * progress  # ring_2: 100->800
                
                # Little finger
                inspire_pos[env_idx, 10] = 500 * progress  # little_1
                inspire_pos[env_idx, 11] = 700 * progress  # little_2
                
            elif state == 2:  # Holding
                # Maintain closed position (in 0-1000 range)
                inspire_pos[env_idx, 0] = 0   # thumb_1: stay straight
                inspire_pos[env_idx, 1] = 0   # thumb_2: stay straight
                inspire_pos[env_idx, 2] = 0   # thumb_3: stay straight
                inspire_pos[env_idx, 3] = 0   # thumb_4: stay straight
                inspire_pos[env_idx, 4] = 950   # index_1 (increased from 800)
                inspire_pos[env_idx, 5] = 1000  # index_2
                inspire_pos[env_idx, 6] = 700   # middle_1
                inspire_pos[env_idx, 7] = 900   # middle_2
                inspire_pos[env_idx, 8] = 600   # ring_1
                inspire_pos[env_idx, 9] = 800   # ring_2
                inspire_pos[env_idx, 10] = 500  # little_1
                inspire_pos[env_idx, 11] = 700  # little_2
                
            elif state == 3:  # Opening
                # Gradual opening back to start positions
                progress = self.state_timer[env_idx] / self.phase_durations[3]
                
                # Thumb joints - stay straight throughout
                inspire_pos[env_idx, 0] = 0   # thumb_1: stay straight
                inspire_pos[env_idx, 1] = 0   # thumb_2: stay straight
                inspire_pos[env_idx, 2] = 0   # thumb_3: stay straight
                inspire_pos[env_idx, 3] = 0   # thumb_4: stay straight
                inspire_pos[env_idx, 4] = 950 * (1.0 - progress)   # index_1 (increased from 800)
                inspire_pos[env_idx, 5] = 1000 * (1.0 - progress)  # index_2
                inspire_pos[env_idx, 6] = 700 * (1.0 - progress)   # middle_1
                inspire_pos[env_idx, 7] = 900 * (1.0 - progress)   # middle_2
                # Ring fingers go back to extended position, not fully open
                inspire_pos[env_idx, 8] = 600 * (1.0 - progress) + 50 * progress  # ring_1: 600->50
                inspire_pos[env_idx, 9] = 800 * (1.0 - progress) + 100 * progress  # ring_2: 800->100
                inspire_pos[env_idx, 10] = 500 * (1.0 - progress)  # little_1
                inspire_pos[env_idx, 11] = 700 * (1.0 - progress)  # little_2
        
        # Convert from Inspire Hand range (0-1000) to radians for Isaac Sim
        radians = self._convert_to_radians(inspire_pos)
        
        return radians

    def _process_contact_sensors(self):
        """Process contact sensor data and log forces."""
        total_sensors = len(self.sensor_pad_names)
        sensors_with_data = 0
        active_forces = 0
        
        force_data = {}
        
        for sensor_name in self.sensor_pad_names:
            try:
                sensor = self.scene[sensor_name]
                
                # Check if sensor has data
                if hasattr(sensor, 'data') and sensor.data is not None:
                    # Get net forces (this is the primary contact force data)
                    net_forces = sensor.data.net_forces_w
                    
                    if net_forces is not None and net_forces.shape[0] > 0:
                        sensors_with_data += 1
                        # Calculate force magnitude
                        force_magnitude = torch.norm(net_forces[0]).item()
                        force_data[sensor_name] = force_magnitude
                        
                        if force_magnitude > 0.001:  # 1mN threshold
                            active_forces += 1
                    else:
                        force_data[sensor_name] = 0.0
                else:
                    force_data[sensor_name] = 0.0
                    
            except KeyError:
                force_data[sensor_name] = 0.0
        
        return total_sensors, sensors_with_data, active_forces, force_data

    def _log_contact_forces(self):
        """Log detailed contact forces."""
        total_sensors, sensors_with_data, active_forces, force_data = self._process_contact_sensors()
        
        state_names = ["Open", "Closing", "Holding", "Opening"]
        current_state = state_names[self.grasp_state[0].item()]
        
        print(f"\n=== Contact Forces at Step {self.step_count} ===")
        print(f"Grasp State: {current_state} | Sensors: {sensors_with_data}/{total_sensors} | Active: {active_forces}")
        
        # Group forces by finger
        fingers = {
            "Thumb": ["thumb_force_sensor_1", "thumb_force_sensor_2", "thumb_force_sensor_3", "thumb_force_sensor_4"],
            "Index": ["index_force_sensor_1", "index_force_sensor_2", "index_force_sensor_3"],
            "Middle": ["middle_force_sensor_1", "middle_force_sensor_2", "middle_force_sensor_3"]
        }
        
        for finger_name, sensor_names in fingers.items():
            print(f"\n{finger_name}:")
            for sensor_name in sensor_names:
                force = force_data.get(sensor_name, 0.0)
                status = "🔴" if force > 0.1 else "⚡" if force > 0.001 else "⚫"
                print(f"  {sensor_name}: {force:.6f} N {status}")

    def _log_contact_forces_brief(self):
        """Log brief contact forces during active phases."""
        total_sensors, sensors_with_data, active_forces, force_data = self._process_contact_sensors()
        
        state_names = ["Open", "Closing", "Holding", "Opening"]
        current_state = state_names[self.grasp_state[0].item()]
        
        # Show active forces only
        active_forces_list = [(name, force) for name, force in force_data.items() if force > 0.001]
        
        if active_forces_list:
            print(f"Step {self.step_count} [{current_state}] Active Forces:")
            for sensor_name, force in active_forces_list:
                print(f"  {sensor_name}: {force:.4f} N")

    def step(self, actions):
        """Step the environment."""
        # Call parent step method
        obs, rew, terminated, truncated, extras = super().step(actions)
        
        self.step_count += 1
        
        # Log contact forces
        if self.step_count % 120 == 0:  # Every 4 seconds at 30Hz
            self._log_contact_forces()
        elif self.step_count % 30 == 0 and self.grasp_state[0].item() in [1, 2]:  # Every 1 second during Closing/Holding
            self._log_contact_forces_brief()
        
        return obs, rew, terminated, truncated, extras


def main():
    """Main function to run the contact sensor test."""
    
    # Create the environment configuration
    cfg = InspireHandContactTestCfg()
    
    # Define sensor names that match what we use in the environment
    sensor_names = [
        "thumb_force_sensor_1", "thumb_force_sensor_2", "thumb_force_sensor_3", "thumb_force_sensor_4",
        "index_force_sensor_1", "index_force_sensor_2", "index_force_sensor_3",
        "middle_force_sensor_1", "middle_force_sensor_2", "middle_force_sensor_3"
    ]
    
    # Add contact sensors dynamically (following the working pattern from reference implementation)
    print(f"[INFO] Adding {len(sensor_names)} contact sensors to scene configuration...")
    
    for sensor_name in sensor_names:
        sensor_cfg = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_name}",
            update_period=0.0,  # Update every simulation step
            history_length=1,
            debug_vis=True,  # Enable red dot visualization
            track_pose=True,  # CRITICAL: This enables sensor data collection
            force_threshold=0.001,  # 1mN threshold for detection
        )
        
        # Dynamically add sensor to scene config
        setattr(cfg.scene, sensor_name, sensor_cfg)
        print(f"[INFO] ✓ Added sensor: {sensor_name}")
    
    print(f"[INFO] All {len(sensor_names)} sensors added to configuration")
    
    # Create environment
    env = InspireHandContactTestEnv(cfg=cfg, render_mode="rgb_array")
    
    # Setup camera
    set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
    
    # Reset the environment
    env.reset()
    
    # Run simulation
    print(f"\n[INFO] Starting sphere grasping simulation...")
    print(f"[INFO] Total runtime: {args_cli.total_time} seconds")
    print("[INFO] Grasp cycle: Open (5s) -> Closing (10s) -> Holding (10s) -> Opening (5s)")
    print("[INFO] Contact forces will be logged every 4 seconds (detailed) and every 1 second (brief during grasp)")
    
    total_steps = int(args_cli.total_time / (cfg.sim.dt * cfg.decimation))
    
    for step in range(total_steps):
        # Use zero actions (let the state machine control the hand)
        actions = torch.zeros(env.num_envs, env.hand.num_joints, device=env.device)
        
        # Step environment
        env.step(actions)
        
        # Log progress every 10% of completion
        if step % (total_steps // 10) == 0:
            progress = (step / total_steps) * 100
            print(f"[INFO] Progress: {progress:.1f}% ({step}/{total_steps} steps)")
    
    print(f"\n[INFO] Simulation completed!")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 
