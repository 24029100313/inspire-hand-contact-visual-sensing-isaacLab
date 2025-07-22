#!/usr/bin/env python3
"""
Basic Inspire Hand Demo.

This script demonstrates basic usage of the Inspire Hand in Isaac Lab:
- Loading the hand with contact sensors
- Simple joint control
- Reading sensor data

Usage:
    ./isaaclab.sh -p basic_demo.py --num_envs 1
"""

import torch
import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Inspire Hand Basic Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# Isaac Lab imports (after launching)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.timer import Timer

# Import hand configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
from inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS


class BasicHandDemo:
    """Basic demonstration of the Inspire Hand."""
    
    def __init__(self):
        # Create scene configuration
        scene_cfg = InteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        scene_cfg.robot = INSPIRE_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/InspireHand")
        
        # Add contact sensors
        for sensor_name, sensor_cfg in CONTACT_SENSOR_CFGS.items():
            setattr(scene_cfg, f"contact_{sensor_name}", sensor_cfg.replace(
                prim_path=sensor_cfg.prim_path.replace("/Robot/", "/InspireHand/")
            ))
        
        # Create scene
        self.scene = InteractiveScene(scene_cfg)
        
        # Get robot reference
        self.robot = self.scene["robot"]
        
        # Collect contact sensors
        self.contact_sensors = {}
        for attr_name in dir(self.scene):
            if attr_name.startswith("contact_"):
                sensor_name = attr_name.replace("contact_", "")
                self.contact_sensors[sensor_name] = getattr(self.scene, attr_name)
        
        print(f"Loaded hand with {self.robot.num_joints} joints")
        print(f"Found {len(self.contact_sensors)} contact sensors")
        
        # Control variables
        self.joint_targets = torch.zeros(
            (self.scene.num_envs, self.robot.num_joints), 
            device=self.scene.device
        )
        self.step_count = 0
    
    def update_control(self):
        """Update joint control - simple open/close motion."""
        t = self.step_count * 0.01  # Time in seconds
        
        # Simple sinusoidal grasp motion
        grasp_signal = 0.5 + 0.4 * torch.sin(t * 0.5)  # Slow grasping
        
        # Get joint names
        joint_names = self.robot.joint_names
        
        for i, joint_name in enumerate(joint_names):
            if "thumb" in joint_name:
                if "1_joint" in joint_name:
                    self.joint_targets[:, i] = grasp_signal * 0.8
                else:
                    self.joint_targets[:, i] = grasp_signal * 0.6
            else:
                # Other fingers
                self.joint_targets[:, i] = grasp_signal * 0.5
        
        # Apply joint targets
        self.robot.set_joint_position_target(self.joint_targets)
    
    def print_sensor_data(self):
        """Print contact sensor data."""
        if self.step_count % 100 == 0:  # Print every ~1 second
            print(f"\n=== Step {self.step_count} ===")
            total_force = 0.0
            active_sensors = 0
            
            for sensor_name, sensor in self.contact_sensors.items():
                forces = sensor.data.net_forces_w[0]  # First environment
                force_magnitude = torch.norm(forces).item()
                
                if force_magnitude > 0.01:  # Only show active sensors
                    print(f"  {sensor_name}: {force_magnitude:.3f}N")
                    total_force += force_magnitude
                    active_sensors += 1
            
            print(f"  Total force: {total_force:.3f}N ({active_sensors} active sensors)")
    
    def run(self, num_steps=3000):
        """Run the demonstration."""
        print("Starting Basic Hand Demo...")
        print("The hand will perform slow grasping motions.")
        print("Contact forces will be displayed when detected.")
        
        # Reset scene
        self.scene.reset()
        
        # Main simulation loop
        for step in range(num_steps):
            # Update control
            self.update_control()
            
            # Step physics
            self.scene.step()
            
            # Print sensor data
            self.print_sensor_data()
            
            self.step_count += 1
        
        print("Demo completed!")


def main():
    """Main function."""
    demo = BasicHandDemo()
    demo.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
