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
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.1, -0.045, 0.21),  # Raise hand by 10 cm for clearance
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
            # Strong clamp on thumb to keep it straight
            "thumb_lock": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_.*_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=8.0,
                stiffness=2000.0,
                damping=200.0,
            ),
            # General gains for other fingers
            "other_fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_(?!thumb).*_joint"],
                effort_limit_sim=80.0,
                velocity_limit_sim=8.0,
                stiffness=1200.0,
                damping=120.0,
            ),
        },
    )
    
    # Sphere configuration - positioned between thumb and index finger
    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.045,
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
            # Original placement
            pos=(0.068, 0.12, 0.075),
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
        device="cuda:0",  # revert to GPU PhysX
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
    decimation = 1  # 120Hz control for best tracking
    
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
        self.sensor_pad_names = list(getattr(cfg, "selected_sensor_names", []))
        
        print(f"[INFO] Initializing InspireHandContactTestEnv with {len(self.sensor_pad_names)} sensor pads")
        if self.sensor_pad_names:
            print(f"[INFO] Sensor pads (first 20 shown): {self.sensor_pad_names[:20]}{' ...' if len(self.sensor_pad_names) > 20 else ''}")
        
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize last 6-channel Inspire targets (little, ring, middle, index, thumb, extra)
        self._last_inspire6 = torch.zeros(self.num_envs, 6, device=self.device)
        self._last_inspire6[:, 4] = 400.0  # thumb stays at 400 baseline

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
                sensor = self.scene[f"contact_{sensor_name}"]
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
        
        # Lazy init joint mappings and limits after PhysX views are ready
        if not hasattr(self, "name_to_index"):
            try:
                self.joint_names = list(self.hand.joint_names)
            except Exception:
                self.joint_names = [f"joint_{i}" for i in range(self.hand.num_joints)]
            self.name_to_index = {name: i for i, name in enumerate(self.joint_names)}
            limits = self.hand.data.soft_joint_pos_limits  # (N, J, 2)
            self.joint_min = limits[0, :, 0].to(self.device)
            self.joint_max = limits[0, :, 1].to(self.device)
            # One-time mapping debug will run after we write the reset state
            try:
                controlled_joint_names = [
                    "right_thumb_1_joint",
                    "right_index_1_joint",
                    "right_middle_1_joint",
                    "right_ring_1_joint",
                    "right_little_1_joint",
                ]
                self._mapping_debug_joint_names = controlled_joint_names
            except Exception as e:
                print(f"[WARN] Mapping debug failed: {e}")

        # Reset hand to default joint positions (from USD defaults + our init_state overrides)
        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_vel = self.hand.data.default_joint_vel[env_ids].clone()
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # After writing, cache the actual current joint positions as "open" defaults
        current_pos = self.hand.data.joint_pos[env_ids]
        # Use env 0 as reference for cached vectors
        self.joint_default = self.hand.data.joint_pos[0].to(self.device)
        # Print mapping debug once with updated defaults
        if hasattr(self, "_mapping_debug_joint_names"):
            print("\n[DEBUG] Joint mapping (open(q0)/min/max) and close direction (using current reset pose as open):")
            for name in self._mapping_debug_joint_names:
                j = self.name_to_index.get(name, None)
                if j is None:
                    print(f"  {name}: NOT FOUND")
                    continue
                q0 = float(self.joint_default[j].item())
                qmin = float(self.joint_min[j].item())
                qmax = float(self.joint_max[j].item())
                dmin = abs(q0 - qmin)
                dmax = abs(qmax - q0)
                close = qmax if dmax >= dmin else qmin
                print(f"  {name:22} idx={j:3d}  q0={q0:+.3f}  qmin={qmin:+.3f}  qmax={qmax:+.3f}  close={close:+.3f}")
        
        # Reset sphere position (changed from cube)
        self.sphere.reset(env_ids)
        
        # Reset desktop
        self.desktop.reset(env_ids)
        
        # Reset state machine
        self.grasp_state[env_ids] = 0
        self.state_timer[env_ids] = 0.0
        
        # Set initial joint targets to current joint positions (stay at reset pose)
        self._joint_pos_target[env_ids] = self.hand.data.joint_pos[env_ids]

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

    def _convert_to_radians(self, inspire_values: torch.Tensor, joint_names_order: list[str]) -> torch.Tensor:
        """Convert normalized open fraction (0..1) to radians using per-joint soft limits.

        Input inspire_values is normalized_open in [0,1], where 1=open, 0=closed.
        For each joint, pick the farther soft limit from q_open as q_close. Output is:
            q = q_close + normalized_open * (q_open - q_close)
        i.e., normalized_open=1 -> q_open, normalized_open=0 -> q_close.
        """
        normalized_open = torch.clamp(inspire_values, 0.0, 1.0)  # (N, K)
        q_open_list = []
        q_close_list = []
        for name in joint_names_order:
            j = self.name_to_index.get(name, None)
            if j is None:
                q_open_list.append(torch.tensor(0.0, device=self.device))
                q_close_list.append(torch.tensor(0.0, device=self.device))
                continue
            q0 = self.joint_default[j]
            qmin = self.joint_min[j]
            qmax = self.joint_max[j]
            # pick farther limit from default as "close"
            dmin = torch.abs(q0 - qmin)
            dmax = torch.abs(qmax - q0)
            q_close = torch.where(dmax >= dmin, qmax, qmin)
            q_open_list.append(q0)
            q_close_list.append(q_close)
        q_open = torch.stack(q_open_list).unsqueeze(0)    # (1, K)
        q_close = torch.stack(q_close_list).unsqueeze(0)  # (1, K)
        return q_close + normalized_open * (q_open - q_close)

    def _generate_inspire6(self) -> torch.Tensor:
        """Generate 6-channel Inspire commands per env following hardware semantics.

        Channel order: [little, ring, middle, index, thumb, extra]
        - Range 0..1000, where 0=closed, 1000=open (hardware style)
        - Thumb fixed to 400 baseline
        - Extra channel unused, set to -1 (hold)
        """
        inspire6 = torch.full((self.num_envs, 6), -1.0, device=self.device)
        for env_idx in range(self.num_envs):
            state = self.grasp_state[env_idx].item()
            t = torch.clamp(self.state_timer[env_idx] / self.phase_durations[state], 0.0, 1.0)
            if state == 0:  # Open (go to 1000)
                val = 1000.0
            elif state == 1:  # Closing (1000 -> 0)
                val = (1.0 - t.item()) * 1000.0
            elif state == 2:  # Holding (stay closed)
                val = 0.0
            else:  # Opening (0 -> 1000)
                val = t.item() * 1000.0

            # Assign to little, ring, middle, index
            inspire6[env_idx, 0] = val  # little
            inspire6[env_idx, 1] = val  # ring
            inspire6[env_idx, 2] = val  # middle
            inspire6[env_idx, 3] = val  # index
            inspire6[env_idx, 4] = 400.0  # thumb baseline
            # channel 5 remains -1 (hold)
        return inspire6

    def _map_inspire6_to_joint_targets(self, inspire6: torch.Tensor) -> torch.Tensor:
        """Map 6-channel Inspire commands to articulation joint targets.

        - Apply -1 hold and deadzone of 20 compared to last value
        - Convert 0..1000 (0=closed, 1000=open) to per-joint radians via soft limits
        - For each finger, set both proximal *_1 and distal *_2 using same normalized value
        - Thumb remains at default (locked by actuators); we do not change thumb joints here
        """
        # Apply -1 hold and deadzone
        effective = self._last_inspire6.clone()
        for i in range(6):
            new_vals = inspire6[:, i]
            hold_mask = new_vals < 0
            delta = torch.abs(new_vals - self._last_inspire6[:, i])
            small_mask = delta < 20.0
            update_mask = (~hold_mask) & (~small_mask)
            effective[:, i] = torch.where(update_mask, new_vals, effective[:, i])
        # Enforce thumb baseline
        effective[:, 4] = 400.0
        self._last_inspire6 = effective

        # Build per-finger joint name lists
        finger_to_joints = {
            0: ["right_little_1_joint", "right_little_2_joint"],
            1: ["right_ring_1_joint", "right_ring_2_joint"],
            2: ["right_middle_1_joint", "right_middle_2_joint"],
            3: ["right_index_1_joint", "right_index_2_joint"],
        }

        # Start from current for stability
        full_targets = self.hand.data.joint_pos.clone()

        # For each finger channel 0..3, compute normalized (open fraction) and set both joints
        for f_idx in range(4):
            val = torch.clamp(effective[:, f_idx], 0.0, 1000.0)
            # Convert from 0=closed,1000=open to normalized open fraction
            norm_open = val / 1000.0
            joint_names = finger_to_joints[f_idx]
            # Make a tensor (N, 2) from norm_open for two joints
            norm_pair = torch.stack([norm_open, norm_open], dim=1)
            # Map to radians using soft limits per joint
            rad_targets = self._convert_to_radians(norm_pair, joint_names)
            for j_idx, jname in enumerate(joint_names):
                j = self.name_to_index.get(jname, None)
                if j is not None:
                    full_targets[:, j] = rad_targets[:, j_idx]

        # Explicitly set thumb joints to open defaults to prevent drift
        for thumb_j in ["right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"]:
            j = self.name_to_index.get(thumb_j, None)
            if j is not None:
                full_targets[:, j] = self.joint_default[j]
        return full_targets

    def _get_joint_targets(self) -> torch.Tensor:
        """Compute joint targets from 6-channel Inspire commands strictly following controller semantics."""
        inspire6 = self._generate_inspire6()
        return self._map_inspire6_to_joint_targets(inspire6)

    def _process_contact_sensors(self):
        """Process contact sensor data and log forces."""
        total_sensors = len(self.sensor_pad_names)
        sensors_with_data = 0
        active_forces = 0
        
        force_data = {}
        
        for sensor_name in self.sensor_pad_names:
            try:
                sensor = self.scene[f"contact_{sensor_name}"]
                
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
        # Print up to 20 active sensors
        active_items = [(n, f) for n, f in force_data.items() if f > 0.001]
        active_items.sort(key=lambda x: x[1], reverse=True)
        if not active_items:
            print("No active contacts.")
        else:
            for name, fval in active_items[:20]:
                status = "🔴" if fval > 0.1 else "⚡"
                print(f"  {name}: {fval:.6f} N {status}")

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

    def _log_joint_angles_debug(self):
        """Log detailed joint angle information for debugging."""
        if not hasattr(self, 'debug_logged') or self.step_count % 60 == 0:  # Log every ~0.5s at 120Hz
            state_names = ["Open", "Closing", "Holding", "Opening"]
            current_state = state_names[self.grasp_state[0].item()]

            print(f"\n=== Joint Angle Debug at Step {self.step_count} ===")
            print(f"Grasp State: {current_state}")
            print(f"State Timer: {self.state_timer[0].item():.2f}s")

            # Recompute intended joint targets from 6-channel mapping
            intended = self._map_inspire6_to_joint_targets(self._generate_inspire6())
            actual = self.hand.data.joint_pos

            # Per-finger pairs
            pairs = [
                ("little", ["right_little_1_joint", "right_little_2_joint"]),
                ("ring", ["right_ring_1_joint", "right_ring_2_joint"]),
                ("middle", ["right_middle_1_joint", "right_middle_2_joint"]),
                ("index", ["right_index_1_joint", "right_index_2_joint"]),
            ]

            print("Finger    | Joint                 | idx  | target(rad) | actual(rad) | diff(rad)")
            print("-" * 90)
            for fname, jnames in pairs:
                for jn in jnames:
                    j = self.name_to_index.get(jn, None)
                    if j is None:
                        continue
                    t = intended[0, j].item()
                    a = actual[0, j].item()
                    d = abs(t - a)
                    status = "✓" if d < 0.1 else ("⚠️" if d < 0.3 else "❌")
                    print(f"{fname:8} | {jn:20} | {j:4d} | {t:11.3f} | {a:11.3f} | {d:8.3f} {status}")

            # Summaries per finger
            print("\nPer-finger summary (max abs diff):")
            for fname, jnames in pairs:
                diffs = []
                for jn in jnames:
                    j = self.name_to_index.get(jn, None)
                    if j is not None:
                        diffs.append(abs(intended[0, j].item() - actual[0, j].item()))
                if diffs:
                    print(f"  {fname:8}: {max(diffs):.3f} rad")

            if not hasattr(self, 'debug_logged'):
                self.debug_logged = True

    def step(self, actions):
        """Step the environment."""
        # Call parent step method
        obs, rew, terminated, truncated, extras = super().step(actions)
        
        self.step_count += 1
        
        # Log joint angles for debugging
        self._log_joint_angles_debug()
        
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
    
    # Build only the requested pad sensors: index_sensor_2_pad_*, index_sensor_3_pad_*, thumb_sensor_3_pad_*, thumb_sensor_4_pad_*
    sensor_names = [
        # Index sensor 2 pads (4 pads: 045, 046, 052, 053)
        "index_sensor_2_pad_045", "index_sensor_2_pad_046",
        "index_sensor_2_pad_052", "index_sensor_2_pad_053",

        # Index sensor 3 pads (9 pads: 001-009)
        "index_sensor_3_pad_001", "index_sensor_3_pad_002", "index_sensor_3_pad_003",
        "index_sensor_3_pad_004", "index_sensor_3_pad_005", "index_sensor_3_pad_006",
        "index_sensor_3_pad_007", "index_sensor_3_pad_008", "index_sensor_3_pad_009",

        # Thumb sensor 3 pads (4 pads: 042, 043, 054, 055)
        "thumb_sensor_3_pad_042", "thumb_sensor_3_pad_043",
        "thumb_sensor_3_pad_054", "thumb_sensor_3_pad_055",

        # Thumb sensor 4 pads (9 pads: 001-009)
        "thumb_sensor_4_pad_001", "thumb_sensor_4_pad_002", "thumb_sensor_4_pad_003",
        "thumb_sensor_4_pad_004", "thumb_sensor_4_pad_005", "thumb_sensor_4_pad_006",
        "thumb_sensor_4_pad_007", "thumb_sensor_4_pad_008", "thumb_sensor_4_pad_009",
    ]

    print(f"[INFO] Adding {len(sensor_names)} specific pad sensors to scene configuration...")
    added_count = 0
    for sensor_name in sensor_names:
        sensor_cfg = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=True,
            track_pose=True,
            force_threshold=0.001,
        )
        setattr(cfg.scene, f"contact_{sensor_name}", sensor_cfg)
        added_count += 1
    print(f"[INFO] Requested {added_count} pad sensors be added (non-existent prims will be ignored at runtime)")

    # Pass selection into cfg for the environment to use
    cfg.selected_sensor_names = sensor_names
    
    # Create environment
    env = InspireHandContactTestEnv(cfg=cfg, render_mode="rgb_array")
    
    # Setup camera
    try:
        if not args_cli.headless:
            set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
    except Exception:
        pass
    
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
