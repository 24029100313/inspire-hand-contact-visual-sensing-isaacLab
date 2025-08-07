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
                disable_gravity=False,  # Enable gravity so cube sits naturally on table
                kinematic_enabled=False,  # Allow cube to be moved by physics
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
            # Place cube on desktop surface for realistic grasping
            # Desktop surface at z=0.025, cube is 6cm, so center at z=0.055
            pos=(0.07, 0.144, 0.0),  # New position as specified
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Contact sensors for each specific pad - configure separately for cube-only detection
    
    # Index Sensor 2 Pads (4 existing)
    contact_index_sensor_2_pad_045: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_2_pad_045",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_2_pad_046: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_2_pad_046",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_2_pad_052: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_2_pad_052",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_2_pad_053: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_2_pad_053",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    # Index Sensor 3 Pads (9 total: 001-009)
    contact_index_sensor_3_pad_001: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_001",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_002: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_002",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_003: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_003",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_004: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_004",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_005: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_005",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_006: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_006",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_007: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_007",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_008: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_008",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_index_sensor_3_pad_009: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/index_sensor_3_pad_009",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    # Thumb Sensor 3 Pads (4 existing)
    contact_thumb_sensor_3_pad_042: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_3_pad_042",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_3_pad_043: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_3_pad_043",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_3_pad_054: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_3_pad_054",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_3_pad_055: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_3_pad_055",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    # Thumb Sensor 4 Pads (9 total: 001-009)
    contact_thumb_sensor_4_pad_001: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_001",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_002: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_002",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_003: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_003",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_004: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_004",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_005: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_005",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_006: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_006",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_007: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_007",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_008: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_008",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    
    contact_thumb_sensor_4_pad_009: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/thumb_sensor_4_pad_009",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
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
    episode_length_s = 30.0  # Each cycle is 30 seconds: open -> close -> hold -> open
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
        
        # Define the sensor pad names we want to monitor (26 total)
        self.sensor_pad_names = [
            # Index sensor 2 pads (4 pads: 045, 046, 052, 053) - existing
            "index_sensor_2_pad_045", "index_sensor_2_pad_046", 
            "index_sensor_2_pad_052", "index_sensor_2_pad_053",
            
            # Index sensor 3 pads (9 pads: 001-009) - expanded from 1 to 9
            "index_sensor_3_pad_001", "index_sensor_3_pad_002", "index_sensor_3_pad_003",
            "index_sensor_3_pad_004", "index_sensor_3_pad_005", "index_sensor_3_pad_006", 
            "index_sensor_3_pad_007", "index_sensor_3_pad_008", "index_sensor_3_pad_009",
            
            # Thumb sensor 3 pads (4 pads: 042, 043, 054, 055) - existing
            "thumb_sensor_3_pad_042", "thumb_sensor_3_pad_043", 
            "thumb_sensor_3_pad_054", "thumb_sensor_3_pad_055",
            
            # Thumb sensor 4 pads (9 pads: 001-009) - expanded from 1 to 9  
            "thumb_sensor_4_pad_001", "thumb_sensor_4_pad_002", "thumb_sensor_4_pad_003",
            "thumb_sensor_4_pad_004", "thumb_sensor_4_pad_005", "thumb_sensor_4_pad_006",
            "thumb_sensor_4_pad_007", "thumb_sensor_4_pad_008", "thumb_sensor_4_pad_009"
        ]
        
        print("=" * 80)
        print("🔧 Inspire Hand Contact Sensor Test Environment Initialized")
        print(f"📍 Monitoring {len(self.sensor_pad_names)} specific sensor pads")
        print("🎮 Control Range: 0-1000 (Inspire Hand native range)")
        print("=" * 80)
        
    def _setup_scene(self):
        """Set up the simulation scene."""
        # Access the hand and cube from the scene
        self.hand = self.scene["hand"]
        self.cube = self.scene["cube"]
        self.desktop = self.scene["desktop"]  # Access the desktop table
        
        # Access all individual contact sensors (26 total)
        self.contact_sensors = {
            # Index sensor 2 pads (4 existing)
            "index_sensor_2_pad_045": self.scene["contact_index_sensor_2_pad_045"],
            "index_sensor_2_pad_046": self.scene["contact_index_sensor_2_pad_046"],
            "index_sensor_2_pad_052": self.scene["contact_index_sensor_2_pad_052"],
            "index_sensor_2_pad_053": self.scene["contact_index_sensor_2_pad_053"],
            
            # Index sensor 3 pads (9 total: 001-009)
            "index_sensor_3_pad_001": self.scene["contact_index_sensor_3_pad_001"],
            "index_sensor_3_pad_002": self.scene["contact_index_sensor_3_pad_002"],
            "index_sensor_3_pad_003": self.scene["contact_index_sensor_3_pad_003"],
            "index_sensor_3_pad_004": self.scene["contact_index_sensor_3_pad_004"],
            "index_sensor_3_pad_005": self.scene["contact_index_sensor_3_pad_005"],
            "index_sensor_3_pad_006": self.scene["contact_index_sensor_3_pad_006"],
            "index_sensor_3_pad_007": self.scene["contact_index_sensor_3_pad_007"],
            "index_sensor_3_pad_008": self.scene["contact_index_sensor_3_pad_008"],
            "index_sensor_3_pad_009": self.scene["contact_index_sensor_3_pad_009"],
            
            # Thumb sensor 3 pads (4 existing)
            "thumb_sensor_3_pad_042": self.scene["contact_thumb_sensor_3_pad_042"],
            "thumb_sensor_3_pad_043": self.scene["contact_thumb_sensor_3_pad_043"],
            "thumb_sensor_3_pad_054": self.scene["contact_thumb_sensor_3_pad_054"],
            "thumb_sensor_3_pad_055": self.scene["contact_thumb_sensor_3_pad_055"],
            
            # Thumb sensor 4 pads (9 total: 001-009)
            "thumb_sensor_4_pad_001": self.scene["contact_thumb_sensor_4_pad_001"],
            "thumb_sensor_4_pad_002": self.scene["contact_thumb_sensor_4_pad_002"],
            "thumb_sensor_4_pad_003": self.scene["contact_thumb_sensor_4_pad_003"],
            "thumb_sensor_4_pad_004": self.scene["contact_thumb_sensor_4_pad_004"],
            "thumb_sensor_4_pad_005": self.scene["contact_thumb_sensor_4_pad_005"],
            "thumb_sensor_4_pad_006": self.scene["contact_thumb_sensor_4_pad_006"],
            "thumb_sensor_4_pad_007": self.scene["contact_thumb_sensor_4_pad_007"],
            "thumb_sensor_4_pad_008": self.scene["contact_thumb_sensor_4_pad_008"],
            "thumb_sensor_4_pad_009": self.scene["contact_thumb_sensor_4_pad_009"],
        }
        
        print(f"📍 Contact sensor info:")
        print(f"   Successfully configured {len(self.contact_sensors)} individual contact sensors")
        for sensor_name in self.contact_sensors.keys():
            print(f"   - {sensor_name}")
        print(f"   Each sensor is configured to detect contact with cube only")
        
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
        """Generate grasp motion for testing using Inspire Hand's 0-1000 range - 30s cycle."""
        # Initialize target positions in Inspire Hand range (0-1000)
        inspire_pos = torch.zeros((self.num_envs, 12), device=self.device)
        
        # Phase 0: Open hand (0-8 seconds)
        if self.grasp_phase == 0:
            # All joints at 0 (open position)
            inspire_pos[:] = 0.0
            if self.phase_timer > 8.0:
                self.grasp_phase = 1
                self.phase_timer = 0.0
                print("\n🤏 Starting grasp motion...")
                
        # Phase 1: Closing hand (8-15 seconds)
        elif self.grasp_phase == 1:
            # Gradually close fingers
            close_ratio = min(self.phase_timer / 7.0, 1.0)  # 7 seconds to close
            
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
            
            if self.phase_timer > 7.0:
                self.grasp_phase = 2
                self.phase_timer = 0.0
                print("\n✊ Grasp closed, holding position...")
                
        # Phase 2: Hold grasp (15-22 seconds)
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
            
            if self.phase_timer > 7.0:  # Hold for 7 seconds
                self.grasp_phase = 3
                self.phase_timer = 0.0
                print("\n🖐️ Opening hand to complete cycle...")
                
        # Phase 3: Opening hand again (22-30 seconds)
        elif self.grasp_phase == 3:
            # Gradually open fingers back to 0
            open_ratio = min(self.phase_timer / 8.0, 1.0)  # 8 seconds to open
            
            # Start from closed positions and gradually go to 0
            inspire_pos[:, 0] = 300 * (1.0 - open_ratio)   # thumb_1
            inspire_pos[:, 1] = 500 * (1.0 - open_ratio)   # thumb_2
            inspire_pos[:, 2] = 400 * (1.0 - open_ratio)   # thumb_3
            inspire_pos[:, 3] = 300 * (1.0 - open_ratio)   # thumb_4
            inspire_pos[:, 4] = 800 * (1.0 - open_ratio)   # index_1
            inspire_pos[:, 5] = 1000 * (1.0 - open_ratio)  # index_2
            inspire_pos[:, 6] = 700 * (1.0 - open_ratio)   # middle_1
            inspire_pos[:, 7] = 900 * (1.0 - open_ratio)   # middle_2
            inspire_pos[:, 8] = 600 * (1.0 - open_ratio)   # ring_1
            inspire_pos[:, 9] = 800 * (1.0 - open_ratio)   # ring_2
            inspire_pos[:, 10] = 500 * (1.0 - open_ratio)  # little_1
            inspire_pos[:, 11] = 700 * (1.0 - open_ratio)  # little_2
            
            if self.phase_timer > 8.0:
                # Cycle complete - will be reset by environment
                print("\n🔄 Cycle complete - ready for reset...")
        
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
        """Process contact sensor data and return force magnitudes for cube contact only."""
        # Initialize contact forces tensor
        contact_forces = torch.zeros((self.num_envs, len(self.sensor_pad_names)), device=self.device)
        
        # Get force data from each individual contact sensor
        for i, pad_name in enumerate(self.sensor_pad_names):
            if pad_name in self.contact_sensors:
                # Get net contact forces from this sensor (only cube contacts due to filter)
                sensor = self.contact_sensors[pad_name]
                net_forces = sensor.data.net_forces_w  # Shape: (num_envs, num_bodies, 3)
                
                # Calculate total force magnitude for all contacts detected by this sensor
                if net_forces.shape[1] > 0:  # If there are any contacts
                    # Sum all contact forces and get magnitude
                    total_force_vec = torch.sum(net_forces, dim=1)  # Sum over all bodies
                    force_mag = torch.norm(total_force_vec, dim=-1)  # Get magnitude
                    contact_forces[:, i] = force_mag
                    
                    # Store for logging (only for single environment)
                    if self.num_envs == 1:
                        self.contact_forces[pad_name] = force_mag.item()
                else:
                    # No contacts detected
                    if self.num_envs == 1:
                        self.contact_forces[pad_name] = 0.0
        
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
        self.desktop.reset(env_ids)  # Reset desktop to initial position
        
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
        phase_names = ['Open', 'Closing', 'Holding', 'Opening']
        print(f"\n⏱️  Time: {self.phase_timer:.1f}s | Phase: {phase_names[self.grasp_phase]}")
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
        
        # Group by sensor type - Updated for all 26 sensors
        index_2_pads = ["index_sensor_2_pad_045", "index_sensor_2_pad_046", 
                        "index_sensor_2_pad_052", "index_sensor_2_pad_053"]
        
        index_3_pads = ["index_sensor_3_pad_001", "index_sensor_3_pad_002", "index_sensor_3_pad_003",
                        "index_sensor_3_pad_004", "index_sensor_3_pad_005", "index_sensor_3_pad_006", 
                        "index_sensor_3_pad_007", "index_sensor_3_pad_008", "index_sensor_3_pad_009"]
        
        thumb_3_pads = ["thumb_sensor_3_pad_042", "thumb_sensor_3_pad_043",
                        "thumb_sensor_3_pad_054", "thumb_sensor_3_pad_055"]
        
        thumb_4_pads = ["thumb_sensor_4_pad_001", "thumb_sensor_4_pad_002", "thumb_sensor_4_pad_003",
                        "thumb_sensor_4_pad_004", "thumb_sensor_4_pad_005", "thumb_sensor_4_pad_006",
                        "thumb_sensor_4_pad_007", "thumb_sensor_4_pad_008", "thumb_sensor_4_pad_009"]
        
        # Print forces by group
        print("🔵 Index Sensor 2 Pads:")
        for pad in index_2_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"  # 10g threshold
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n🔵 Index Sensor 3 Pads:")
        for pad in index_3_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n🟦 Thumb Sensor 3 Pads:")
        for pad in thumb_3_pads:
            force = self.contact_forces.get(pad, 0.0)
            status = "✅" if force > 0.098 else "⭕"
            print(f"   {pad}: {force:6.3f} N {status}")
            
        print("\n⬜ Thumb Sensor 4 Pads:")
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
    print("📋 Test sequence (30s cycles):")
    print("   1. Hand starts in open position (0-8s)")
    print("   2. Hand closes to grasp the cube (8-15s)")
    print("   3. Hand holds the grasp (15-22s)")
    print("   4. Hand opens to complete cycle (22-30s)")
    print("   5. Cycle repeats automatically")
    print("\n🎮 Control Info:")
    print("   - Inspire Hand native range: 0-1000")
    print("   - 0 = fully open, 1000 = fully closed")
    print("   - Converted to radians for Isaac Sim")
    print(f"\n⏰ Total runtime: {args_cli.total_time:.0f} seconds")
    print("✅ = Contact force > 10g (0.098N)")
    print("⭕ = No significant contact")
    print("="*80 + "\n")
    
    # Track total simulation time
    import time
    start_time = time.time()
    sim_dt = env_cfg.sim.dt * env_cfg.decimation  # Control timestep
    cycle_count = 0
    
    # Run simulation
    count = 0
    while simulation_app.is_running():
        # Check total runtime
        elapsed_time = time.time() - start_time
        if elapsed_time >= args_cli.total_time:
            print(f"\n🏁 Total runtime ({args_cli.total_time:.0f}s) completed!")
            print(f"📊 Completed {cycle_count} full cycles")
            break
            
        with torch.inference_mode():
            # Step environment
            actions = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Count completed cycles when environment resets
            if terminated[0] or truncated[0]:
                cycle_count += 1
                print(f"\n🔄 Cycle {cycle_count} completed at {elapsed_time:.1f}s")
            
        # Exit on ESC or simulation stop
        if env.unwrapped.sim.is_stopped():
            break
            
        count += 1
        
    print(f"\n📈 Final Statistics:")
    print(f"   Total time: {elapsed_time:.1f}s")
    print(f"   Cycles completed: {cycle_count}")
    print(f"   Average cycle time: {elapsed_time/max(cycle_count,1):.1f}s")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close() 
