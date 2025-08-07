#!/usr/bin/env python3
"""
Test Contact Sensors for Inspire Hand with Specific Pads

This script creates a test environment where the Inspire Hand grasps a cube
and monitors the contact forces on the 10 specific sensor pads.

Usage:
    ./isaaclab.sh -p test_contact_sensors_specific_pads.py --num_envs 1
"""

import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test contact sensors on Inspire Hand with specific pads")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O operations")
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
            pos=(0.0, 0.0, 0.3),  # Hand position above cube
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation - hand in natural upright orientation
            joint_pos={
                # Initial positions in radians (will be converted from 0-1000 range)
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
    
    # Cube configuration - positioned between thumb and index finger
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),  # 6cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # Disable gravity to keep it fixed
                kinematic_enabled=True,  # Make it kinematic so it won't move when touched
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.1,  # 100g
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # Red color
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Position based on URDF analysis:
            # Thumb root: (-0.027, 0.021, 0.069)
            # Index root: (-0.039, 0.0006, 0.156)
            # Place cube between fingers for easy grasping
            pos=(-0.0058, 0.063, 0.43),  # Between thumb and index finger
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Contact sensors for the specific pads
    contact_sensors: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/.*",
        update_period=0.0,  # Update every step
        history_length=6,
        debug_vis=True,
        track_pose=False,  # Disable pose tracking to focus on forces
        force_threshold=0.001,  # Lower threshold for more sensitive detection
        filter_prim_paths_expr=[
            # Include only the specific sensor pads - use absolute paths
            "{ENV_REGEX_NS}/InspireHand/.*index_sensor_2_pad_045.*",
            "{ENV_REGEX_NS}/InspireHand/.*index_sensor_2_pad_046.*",
            "{ENV_REGEX_NS}/InspireHand/.*index_sensor_2_pad_052.*", 
            "{ENV_REGEX_NS}/InspireHand/.*index_sensor_2_pad_053.*",
            "{ENV_REGEX_NS}/InspireHand/.*index_sensor_3_pad_005.*",
            "{ENV_REGEX_NS}/InspireHand/.*thumb_sensor_3_pad_042.*",
            "{ENV_REGEX_NS}/InspireHand/.*thumb_sensor_3_pad_043.*",
            "{ENV_REGEX_NS}/InspireHand/.*thumb_sensor_3_pad_054.*",
            "{ENV_REGEX_NS}/InspireHand/.*thumb_sensor_3_pad_055.*",
            "{ENV_REGEX_NS}/InspireHand/.*thumb_sensor_4_pad_005.*"
        ]
    )


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
    episode_length_s = 30.0
    decimation = 4
    num_actions = 6  # Control 6 main joints
    num_observations = 50
    num_states = 0
    
    # Add required spaces
    observation_space = 50  # Should match num_observations
    action_space = 6  # Should match num_actions


class InspireHandContactTestEnv(DirectRLEnv):
    """Environment for testing contact sensors on Inspire Hand."""
    
    cfg: InspireHandContactTestCfg
    
    def __init__(self, cfg: InspireHandContactTestCfg, render_mode: str = None, **kwargs):
        """Initialize the environment."""
        
        # Initialize parent class first
        super().__init__(cfg, render_mode, **kwargs)
        
        # Set camera view after parent initialization
        if not args_cli.headless:
            set_camera_view([1.0, 1.0, 1.0], [0.0, 0.0, 0.2])
        
        # Initialize tracking variables
        self.contact_forces = {}
        self.grasp_phase = 0  # 0: open, 1: closing, 2: closed
        self.phase_timer = 0.0
        self.step_count = 0  # Add step counter for logging control
        
        # Define sensor pad names to monitor
        self.sensor_pad_names = [
            "index_sensor_2_pad_045", "index_sensor_2_pad_046", 
            "index_sensor_2_pad_052", "index_sensor_2_pad_053",
            "index_sensor_3_pad_005",
            "thumb_sensor_3_pad_042", "thumb_sensor_3_pad_043",
            "thumb_sensor_3_pad_054", "thumb_sensor_3_pad_055", 
            "thumb_sensor_4_pad_005"
        ]
        
        print("=" * 80)
        print("🔧 Inspire Hand Contact Sensor Test Environment Initialized")
        print(f"📍 Monitoring {len(self.sensor_pad_names)} specific sensor pads")
        print("🎮 Control Range: 0-1000 (Inspire Hand native range)")
        print("=" * 80)
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Access the hand, cube and sensors from the scene
        self.hand = self.scene["hand"]
        self.cube = self.scene["cube"]
        self.contact_sensors = self.scene["contact_sensors"]
        
        print(f"📍 Contact sensor info:")
        # Fix the attribute access - use data.num_bodies or just skip this for now
        try:
            if hasattr(self.contact_sensors, 'num_bodies'):
                print(f"   Total bodies tracked: {self.contact_sensors.num_bodies}")
            elif hasattr(self.contact_sensors, 'data') and hasattr(self.contact_sensors.data, 'num_bodies'):
                print(f"   Total bodies tracked: {self.contact_sensors.data.num_bodies}")
            else:
                print(f"   Contact sensors initialized successfully")
        except Exception as e:
            print(f"   Contact sensors initialized (details: {e})")
        
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # Generate grasp motion based on phase
        target_positions = self._generate_grasp_motion()
        self.hand.set_joint_position_target(target_positions)
        
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
        
    def _generate_grasp_motion(self) -> torch.Tensor:
        """Generate grasp motion for testing using Inspire Hand's 0-1000 range."""
        # Initialize target positions in Inspire Hand range (0-1000)
        inspire_pos = torch.zeros((self.num_envs, 12), device=self.device)
        
        # Phase 0: Open hand (0-5 seconds)
        if self.grasp_phase == 0:
            # All joints at 0 (open position)
            inspire_pos[:] = 0.0
            if self.phase_timer > 5.0:
                self.grasp_phase = 1
                self.phase_timer = 0.0
                print("\n🤏 Starting grasp motion...")
                
        # Phase 1: Closing hand (5-10 seconds)
        elif self.grasp_phase == 1:
            # Gradually close fingers
            close_ratio = min(self.phase_timer / 5.0, 1.0)
            
            # Joint indices mapping (based on order in URDF)
            # 0-3: thumb joints, 4-5: index joints, 6-7: middle, 8-9: ring, 10-11: little
            
            # Thumb joints - move less to allow opposition grip (0-1000 range)
            inspire_pos[:, 0] = 300 * close_ratio  # thumb_1
            inspire_pos[:, 1] = 500 * close_ratio  # thumb_2  
            inspire_pos[:, 2] = 400 * close_ratio  # thumb_3
            inspire_pos[:, 3] = 300 * close_ratio  # thumb_4
            
            # Index finger - main grasping finger
            inspire_pos[:, 4] = 800 * close_ratio  # index_1
            inspire_pos[:, 5] = 1000 * close_ratio  # index_2
            
            # Other fingers - support grasp
            inspire_pos[:, 6] = 700 * close_ratio  # middle_1
            inspire_pos[:, 7] = 900 * close_ratio  # middle_2
            inspire_pos[:, 8] = 600 * close_ratio  # ring_1
            inspire_pos[:, 9] = 800 * close_ratio  # ring_2
            inspire_pos[:, 10] = 500 * close_ratio  # little_1
            inspire_pos[:, 11] = 700 * close_ratio  # little_2
            
            if self.phase_timer > 5.0:
                self.grasp_phase = 2
                self.phase_timer = 0.0
                print("\n✊ Grasp closed, holding position...")
                
        # Phase 2: Hold grasp (10-20 seconds)
        elif self.grasp_phase == 2:
            # Maintain closed position (in 0-1000 range)
            inspire_pos[:, 0] = 300   # thumb_1
            inspire_pos[:, 1] = 500   # thumb_2
            inspire_pos[:, 2] = 400   # thumb_3
            inspire_pos[:, 3] = 300   # thumb_4
            inspire_pos[:, 4] = 800   # index_1
            inspire_pos[:, 5] = 1000  # index_2
            inspire_pos[:, 6] = 700   # middle_1
            inspire_pos[:, 7] = 900   # middle_2
            inspire_pos[:, 8] = 600   # ring_1
            inspire_pos[:, 9] = 800   # ring_2
            inspire_pos[:, 10] = 500  # little_1
            inspire_pos[:, 11] = 700  # little_2
            
            if self.phase_timer > 10.0:
                self.grasp_phase = 0
                self.phase_timer = 0.0
                print("\n🖐️ Opening hand...")
        
        # Convert from Inspire Hand range (0-1000) to radians for Isaac Sim
        target_pos_radians = self._convert_to_radians(inspire_pos)
        
        return target_pos_radians
        
    def _apply_action(self) -> None:
        """Apply actions to the simulation."""
        # Actions are handled in _pre_physics_step
        pass
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get environment observations."""
        # Get hand joint positions and velocities
        hand_joint_pos = self.hand.data.joint_pos
        hand_joint_vel = self.hand.data.joint_vel
        
        # Get cube position and orientation
        cube_pos = self.cube.data.root_pos_w - self.hand.data.root_pos_w
        cube_quat = self.cube.data.root_quat_w
        
        # Get contact sensor data
        contact_data = self._process_contact_sensors()
        
        # Combine observations
        obs = torch.cat([
            hand_joint_pos,
            hand_joint_vel,
            cube_pos,
            cube_quat,
            contact_data
        ], dim=-1)
        
        return {"policy": obs}
        
    def _process_contact_sensors(self) -> torch.Tensor:
        """Process contact sensor data and return force magnitudes."""
        # Initialize contact forces tensor
        contact_forces = torch.zeros((self.num_envs, len(self.sensor_pad_names)), device=self.device)
        
        # Get net contact forces
        net_forces = self.contact_sensors.data.net_forces_w  # Shape: (num_envs, num_bodies, 3)
        
        # Map sensor pad names to indices (this is a simplified approach)
        # In practice, you'd need to map the actual body indices to pad names
        for i, pad_name in enumerate(self.sensor_pad_names):
            if i < net_forces.shape[1]:
                # Calculate force magnitude for each pad
                force_vec = net_forces[:, i, :]
                force_mag = torch.norm(force_vec, dim=-1)
                contact_forces[:, i] = force_mag
                
                # Store for logging
                if self.num_envs == 1:
                    self.contact_forces[pad_name] = force_mag.item()
        
        return contact_forces
        
    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards (not used for testing, but required by parent class)."""
        return torch.zeros(self.num_envs, device=self.device)
        
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done flags (not used for testing)."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        # Reset hand to initial position
        self.hand.reset(env_ids)
        self.cube.reset(env_ids)
        
        # Reset phase
        if len(env_ids) > 0:
            self.grasp_phase = 0
            self.phase_timer = 0.0
            
    def step(self, action: torch.Tensor):
        """Step the environment and log contact data."""
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update phase timer
        self.phase_timer += self.physics_dt * self.cfg.decimation
        self.step_count += 1
        
        # Log contact forces every 60 steps (approximately every 0.5 seconds at 120Hz)
        if self.step_count % 60 == 0 and self.num_envs == 1:
            self._log_contact_forces()
            
        return obs, reward, terminated, truncated, info
        
    def _log_contact_forces(self):
        """Log contact forces for each sensor pad."""
        print(f"\n⏱️  Time: {self.phase_timer:.1f}s | Phase: {['Open', 'Closing', 'Holding'][self.grasp_phase]}")
        print("-" * 60)
        
        # Add joint angle debugging
        if hasattr(self.hand, 'data') and hasattr(self.hand.data, 'joint_pos'):
            joint_pos_rad = self.hand.data.joint_pos[0]  # Get first environment
            # Convert back to 0-1000 range for display
            joint_names = [
                "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                "index_1", "index_2", "middle_1", "middle_2", 
                "ring_1", "ring_2", "little_1", "little_2"
            ]
            
            print("🎮 Current Joint Angles:")
            print("   Joint Name        | Radians | 0-1000 Range")
            print("   " + "-" * 45)
            
            # Define max angles for conversion back to 0-1000
            max_angles = torch.tensor([
                1.16, 0.58, 0.50, 3.14,  # thumb joints
                1.44, 3.14, 1.44, 3.14,  # index, middle joints
                1.44, 3.14, 1.44, 3.14,  # ring, little joints
            ], device=self.device)
            
            for i, (name, pos_rad) in enumerate(zip(joint_names, joint_pos_rad)):
                if i < len(max_angles):
                    # Convert back to 0-1000 range
                    normalized = pos_rad / max_angles[i]
                    inspire_value = normalized * 1000.0
                    inspire_value = torch.clamp(inspire_value, 0, 1000)  # Ensure within range
                    print(f"   {name:12} | {pos_rad:7.3f} | {inspire_value:8.1f}")
            print()
        
        # Group by sensor type
        index_2_pads = ["index_sensor_2_pad_045", "index_sensor_2_pad_046", 
                        "index_sensor_2_pad_052", "index_sensor_2_pad_053"]
        index_3_pads = ["index_sensor_3_pad_005"]
        thumb_3_pads = ["thumb_sensor_3_pad_042", "thumb_sensor_3_pad_043",
                        "thumb_sensor_3_pad_054", "thumb_sensor_3_pad_055"]
        thumb_4_pads = ["thumb_sensor_4_pad_005"]
        
        # Print forces by group
        print("🔵 Index Sensor 2 Pads:")
        for pad in index_2_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"  # 10g threshold
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n🔵 Index Sensor 3 Pad:")
        for pad in index_3_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n🟦 Thumb Sensor 3 Pads:")
        for pad in thumb_3_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n⬜ Thumb Sensor 4 Pad:")
        for pad in thumb_4_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"
            print(f"   {pad}: {force:6.3f} N {status}")
            
        # Summary
        active_pads = sum(1 for f in self.contact_forces.values() if f > 0.098)
        print(f"\n📊 Active pads: {active_pads}/{len(self.sensor_pad_names)}")
        print("=" * 60)


def main():
    """Main function to run the contact sensor test."""
    # Create environment configuration
    env_cfg = InspireHandContactTestCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = InspireHandContactTestEnv(cfg=env_cfg, render_mode="human" if not args_cli.headless else None)
    
    # Print instructions
    print("\n" + "="*80)
    print("🤖 INSPIRE HAND CONTACT SENSOR TEST")
    print("="*80)
    print("📋 Test sequence:")
    print("   1. Hand starts in open position (0-5s)")
    print("   2. Hand closes to grasp the cube (5-10s)")
    print("   3. Hand holds the grasp (10-20s)")
    print("   4. Hand opens and cycle repeats")
    print("\n🎮 Control Info:")
    print("   - Inspire Hand native range: 0-1000")
    print("   - 0 = fully open, 1000 = fully closed")
    print("   - Converted to radians for Isaac Sim")
    print("\n✅ = Contact force > 10g (0.098N)")
    print("⭕ = No significant contact")
    print("="*80 + "\n")
    
    # Run simulation
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Step environment
            actions = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
            env.step(actions)
            
        # Exit on ESC
        if env.unwrapped.sim.is_stopped():
            break
            
        count += 1
        
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close() 
