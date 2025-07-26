#!/usr/bin/env python3
"""
Fixed Script to create a complete Isaac Lab Asset Package for Inspire Hand.

This script creates a properly structured asset package following Isaac Lab conventions,
with fixes for Isaac Sim 4.5 API compatibility.

Usage:
    /path/to/isaac-sim/python.sh create_inspire_hand_asset_package_fixed.py
"""

import os
import sys
import shutil
from pathlib import Path
import json

# Try to import Isaac Sim modules
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})
    
    import omni.kit.commands
    from isaacsim.core.utils.extensions import get_extension_path_from_name
    from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
    
    ISAAC_SIM_AVAILABLE = True
except ImportError as e:
    print(f"Isaac Sim not available: {e}")
    print("This script can still create the folder structure, but USD conversion requires Isaac Sim.")
    ISAAC_SIM_AVAILABLE = False


class InspireHandAssetPackageCreatorFixed:
    """Creates a complete Isaac Lab asset package for the Inspire Hand with fixes."""
    
    def __init__(self):
        """Initialize paths and create package structure."""
        # Source paths
        self.source_dir = Path("/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project")
        self.source_urdf_dir = self.source_dir / "urdf_right_with_force_sensor"
        self.source_urdf = self.source_urdf_dir / "urdf" / "urdf_right_with_force_sensor.urdf"
        self.source_meshes = self.source_urdf_dir / "meshes"
        self.source_textures = self.source_urdf_dir / "textures"
        
        # Package paths
        self.package_dir = self.source_dir / "inspire_hand_with_sensors"
        self.package_urdf = self.package_dir / "urdf"
        self.package_meshes = self.package_dir / "meshes"
        self.package_textures = self.package_dir / "textures"
        self.package_usd = self.package_dir / "usd"
        self.package_config = self.package_dir / "config"
        self.package_examples = self.package_dir / "examples"
        self.package_docs = self.package_dir / "docs"
        
        # Asset package structure
        self.structure = {
            "urdf": "URDF files",
            "meshes": "3D mesh files (.stl, .obj, etc.)",
            "textures": "Texture files and materials",
            "usd": "USD files for Isaac Sim",
            "config": "Isaac Lab configuration files",
            "examples": "Example usage scripts",
            "docs": "Documentation and guides"
        }
        
    def verify_source_files(self):
        """Verify that source URDF files exist."""
        if not self.source_urdf.exists():
            print(f"Error: Source URDF file not found: {self.source_urdf}")
            return False
            
        if not self.source_meshes.exists():
            print(f"Error: Meshes directory not found: {self.source_meshes}")
            return False
        
        print(f"✓ Source files verified:")
        print(f"  URDF: {self.source_urdf}")
        print(f"  Meshes: {self.source_meshes} ({len(list(self.source_meshes.glob('*.STL')))} files)")
        
        return True
    
    def create_package_structure(self):
        """Create the asset package directory structure."""
        print(f"\n🏗️  Creating asset package structure at: {self.package_dir}")
        
        # Remove existing directory if it exists
        if self.package_dir.exists():
            print(f"⚠️  Removing existing package directory")
            shutil.rmtree(self.package_dir)
        
        # Create main package directory
        self.package_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir, description in self.structure.items():
            dir_path = self.package_dir / subdir
            dir_path.mkdir(exist_ok=True)
            
            # Create README in each directory
            readme_path = dir_path / "README.md"
            readme_content = f"# {subdir.title()}\n\n{description}\n"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
        
        print(f"✓ Package structure created with {len(self.structure)} directories")
    
    def copy_source_files(self):
        """Copy source URDF, mesh and texture files to package."""
        print("\n📁 Copying source files...")
        
        # Copy URDF files
        shutil.copy2(self.source_urdf, self.package_urdf)
        print(f"  ✓ Copied URDF: {self.source_urdf.name}")
        
        # Copy other URDF-related files
        for other_file in self.source_urdf_dir.glob("*"):
            if other_file.suffix not in ['.urdf']:
                if other_file.is_file():
                    shutil.copy2(other_file, self.package_urdf)
                    print(f"  ✓ Copied: {other_file.name}")
        
        # Copy mesh files
        shutil.copytree(self.source_meshes, self.package_meshes, dirs_exist_ok=True)
        print(f"  ✓ Copied {len(list(self.source_meshes.glob('*')))} mesh files")
        
        # Copy texture files if they exist
        if self.source_textures.exists():
            shutil.copytree(self.source_textures, self.package_textures, dirs_exist_ok=True)
            print(f"  ✓ Copied {len(list(self.source_textures.glob('*')))} texture files")
    
    def preprocess_urdf_for_package(self):
        """Process URDF file to use relative paths within the package."""
        urdf_file = self.package_urdf / "urdf_right_with_force_sensor.urdf"
        processed_urdf = self.package_urdf / "inspire_hand_processed.urdf"
        
        print(f"\n🔧 Processing URDF file: {urdf_file.name}")
        
        with open(urdf_file, 'r') as f:
            urdf_content = f.read()
        
        # Replace package:// paths with relative paths
        package_name = "urdf_right_with_force_sensor"
        
        # Replace mesh paths to use relative paths within the package
        urdf_content = urdf_content.replace(
            f"package://{package_name}/meshes/",
            "../meshes/"  # Relative to urdf directory
        )
        
        # Replace texture paths if any
        urdf_content = urdf_content.replace(
            f"package://{package_name}/textures/",
            "../textures/"
        )
        
        # Update robot name
        urdf_content = urdf_content.replace(
            f'name="{package_name}"',
            f'name="inspire_hand_with_sensors"'
        )
        
        # Add metadata comments
        metadata_comment = f"""
<!-- 
Isaac Lab Asset Package: Inspire Hand with Contact Sensors
Generated from: {package_name}
Creation date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This URDF defines a complete inspire hand with integrated contact sensors:
- 1 Palm contact sensor
- 4 Thumb contact sensors (thumb_force_sensor_1 through thumb_force_sensor_4)
- 3 Index finger contact sensors
- 3 Middle finger contact sensors  
- 3 Ring finger contact sensors
- 3 Little finger contact sensors
Total: 17 contact sensors

File structure:
- urdf/: URDF robot description files
- meshes/: STL mesh files for visualization and collision
- textures/: Texture and material files
- usd/: Generated USD files for Isaac Sim
- config/: Isaac Lab configuration files
- examples/: Usage examples and demo scripts
-->

"""
        
        # Insert metadata after XML declaration
        lines = urdf_content.split('\n')
        if lines[0].startswith('<?xml'):
            lines.insert(1, metadata_comment)
            urdf_content = '\n'.join(lines)
        
        # Write processed URDF
        with open(processed_urdf, 'w') as f:
            f.write(urdf_content)
        
        print(f"  ✓ Created processed URDF: {processed_urdf.name}")
        return processed_urdf
    
    def convert_to_usd(self, urdf_file):
        """Convert URDF to USD format using simplified configuration."""
        if not ISAAC_SIM_AVAILABLE:
            print("⚠️  Isaac Sim not available - skipping USD conversion")
            print("   Run this script from Isaac Sim environment to generate USD file")
            return None
        
        print(f"\n🔄 Converting URDF to USD...")
        
        try:
            # Setup import configuration (using only supported parameters)
            print("  Setting up URDF import configuration...")
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            
            # Basic configuration parameters (confirmed working in Isaac Sim 4.5)
            import_config.merge_fixed_joints = False  # Keep sensor joints separate
            import_config.convex_decomp = False  # Use original meshes
            import_config.import_inertia_tensor = True  # Import inertial properties
            import_config.fix_base = True  # Fix the base link
            import_config.distance_scale = 1.0  # Keep original scale
            
            # Additional parameters that might be supported
            try:
                import_config.density = 1000.0  # Default density kg/m³
            except AttributeError:
                print("    Note: density parameter not available, using defaults")
            
            print(f"  Importing URDF from: {urdf_file}")
            
            # Import URDF
            status, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(urdf_file),
                import_config=import_config,
                get_articulation_root=True,
            )
            
            if not status:
                print("❌ Failed to import URDF file")
                return None
            
            print(f"  ✓ URDF imported successfully. Prim path: {prim_path}")
            
            # Save USD file
            usd_file = self.package_usd / "inspire_hand_with_sensors.usd"
            print(f"  Saving USD file to: {usd_file}")
            
            # Get stage and save
            stage = omni.usd.get_context().get_stage()
            result = omni.usd.get_context().save_as_stage(str(usd_file))
            
            if result:
                print(f"  ✅ USD file created successfully: {usd_file.name}")
                print(f"     File size: {usd_file.stat().st_size / (1024*1024):.2f} MB")
                return usd_file
            else:
                print("❌ Failed to save USD file")
                return None
                
        except Exception as e:
            print(f"❌ Error during USD conversion: {e}")
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            return None
    
    def create_isaac_lab_config(self, usd_file):
        """Create Isaac Lab configuration file with fixed variable scoping."""
        config_file = self.package_config / "inspire_hand_cfg.py"
        
        print(f"\n⚙️  Creating Isaac Lab configuration...")
        
        # Determine USD path - use relative path if USD file exists, otherwise placeholder
        if usd_file and usd_file.exists():
            usd_path_rel = f"../usd/{usd_file.name}"
            print(f"  Using USD file: {usd_file.name}")
        else:
            usd_path_rel = "../usd/inspire_hand_with_sensors.usd"
            print(f"  USD file not found, using placeholder path")
        
        config_content = f'''"""
Isaac Lab Configuration for Inspire Hand with Contact Sensors.

This configuration defines the Inspire Hand robot for use in Isaac Lab environments.
The hand includes 17 contact sensors distributed across fingers and palm.

Usage:
    from {self.package_dir.name}.config.inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS
"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg

# Get the path to the USD file (relative to this config file)
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
INSPIRE_HAND_USD_PATH = os.path.join(_CURRENT_DIR, "{usd_path_rel}")


##
# Robot Configuration
##

INSPIRE_HAND_CFG = ArticulationCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=INSPIRE_HAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collision for multi-finger hand
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        joint_pos={{
            # Initial joint positions - fingers slightly open for stable grasping
            "right_thumb_1_joint": 0.2,      # Thumb base joint
            "right_thumb_2_joint": 0.2,      # Thumb middle joint
            "right_thumb_3_joint": 0.16,     # Thumb tip (mimic joint)
            "right_index_1_joint": 0.1,      # Index base joint
            "right_index_2_joint": 0.1,      # Index tip joint
            "right_middle_1_joint": 0.1,     # Middle base joint
            "right_middle_2_joint": 0.1,     # Middle tip joint
            "right_ring_1_joint": 0.1,       # Ring base joint
            "right_ring_2_joint": 0.1,       # Ring tip joint
            "right_little_1_joint": 0.1,     # Little base joint
            "right_little_2_joint": 0.1,     # Little tip joint
        }},
        joint_vel={{".*": 0.0}},
    ),
    actuators={{
        # Thumb actuators - stronger for opposition
        "thumb": ImplicitActuatorCfg(
            joint_names_expr=["right_thumb_.*_joint"],
            effort_limit=10.0,    # Higher force limit for thumb
            velocity_limit=2.0,
            stiffness=100.0,      # Higher stiffness for precise control
            damping=10.0,
        ),
        # Index finger actuators
        "index": ImplicitActuatorCfg(
            joint_names_expr=["right_index_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Middle finger actuators
        "middle": ImplicitActuatorCfg(
            joint_names_expr=["right_middle_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Ring finger actuators
        "ring": ImplicitActuatorCfg(
            joint_names_expr=["right_ring_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Little finger actuators
        "little": ImplicitActuatorCfg(
            joint_names_expr=["right_little_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
    }},
)


##
# Contact Sensor Configurations (Fixed variable scoping - all hardcoded strings)
##

CONTACT_SENSOR_CFGS = {{}}

# Palm contact sensor
CONTACT_SENSOR_CFGS["palm_contact"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/palm_force_sensor",
    track_pose=True,
    update_period=0.0,  # Update every simulation step
    debug_vis=True,
)

# Thumb contact sensors (4 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["thumb_sensor_1"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/thumb_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_2"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/thumb_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_3"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/thumb_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_4"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/thumb_force_sensor_4",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Index finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["index_sensor_1"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/index_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["index_sensor_2"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/index_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["index_sensor_3"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/index_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Middle finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["middle_sensor_1"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/middle_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["middle_sensor_2"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/middle_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["middle_sensor_3"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/middle_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Ring finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["ring_sensor_1"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/ring_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["ring_sensor_2"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/ring_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["ring_sensor_3"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/ring_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Little finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["little_sensor_1"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/little_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["little_sensor_2"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/little_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["little_sensor_3"] = ContactSensorCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/little_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)


##
# Utility Functions
##

def get_contact_sensor_names():
    """Get list of all contact sensor names."""
    return list(CONTACT_SENSOR_CFGS.keys())


def print_hand_info():
    """Print information about the hand configuration."""
    print("Inspire Hand Configuration Summary:")
    print(f"  USD File: {{INSPIRE_HAND_USD_PATH}}")
    print(f"  Total Joints: {{len(INSPIRE_HAND_CFG.init_state.joint_pos)}}")
    print(f"  Total Contact Sensors: {{len(CONTACT_SENSOR_CFGS)}}")
    print("  Contact Sensors by Location:")
    print("    Palm: 1 sensor")
    print("    Thumb: 4 sensors")
    print("    Index: 3 sensors")
    print("    Middle: 3 sensors")
    print("    Ring: 3 sensors")
    print("    Little: 3 sensors")


if __name__ == "__main__":
    print_hand_info()
'''
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"  ✓ Isaac Lab configuration created: {config_file.name}")
        return config_file
    
    def create_example_scripts(self):
        """Create example usage scripts."""
        examples_dir = self.package_examples
        
        print(f"\n📝 Creating example scripts...")
        
        # Basic demo script
        demo_script = examples_dir / "basic_demo.py"
        demo_content = '''#!/usr/bin/env python3
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
            print(f"\\n=== Step {self.step_count} ===")
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
'''
        
        with open(demo_script, 'w') as f:
            f.write(demo_content)
        
        print(f"  ✓ Example script created: {demo_script.name}")
    
    def create_documentation(self):
        """Create documentation files."""
        docs_dir = self.package_docs
        
        print(f"\n📚 Creating documentation...")
        
        # Main README
        readme_file = self.package_dir / "README.md"
        readme_content = f'''# {self.package_dir.name.replace('_', ' ').title()}

A complete Isaac Lab asset package for the Inspire Hand with integrated contact sensors.

## Overview

This package provides a fully-configured Inspire Hand robot for Isaac Lab simulations. The hand features:
- **17 contact sensors** distributed across fingers and palm
- **11 actuated joints** for realistic finger movement
- **High-fidelity STL meshes** for accurate collision and visualization
- **Isaac Lab integration** with pre-configured sensor and actuator settings
- **Fixed API compatibility** for Isaac Sim 4.5+

## Package Structure

```
{self.package_dir.name}/
├── urdf/           # URDF robot description files
├── meshes/         # STL mesh files for visualization and collision
├── textures/       # Texture and material files  
├── usd/            # Generated USD files for Isaac Sim
├── config/         # Isaac Lab configuration files
├── examples/       # Usage examples and demo scripts
├── docs/           # Documentation and guides
└── README.md       # This file
```

## Contact Sensors

The hand includes the following contact sensors:

### Palm
- `palm_force_sensor`: Located on the palm center

### Thumb (4 sensors)
- `thumb_force_sensor_1` through `thumb_force_sensor_4`
- Distributed across thumb segments for complete coverage

### Fingers (3 sensors each)
- **Index**: `index_force_sensor_1`, `index_force_sensor_2`, `index_force_sensor_3`
- **Middle**: `middle_force_sensor_1`, `middle_force_sensor_2`, `middle_force_sensor_3`  
- **Ring**: `ring_force_sensor_1`, `ring_force_sensor_2`, `ring_force_sensor_3`
- **Little**: `little_force_sensor_1`, `little_force_sensor_2`, `little_force_sensor_3`

## Quick Start

### 1. Basic Usage

```python
from {self.package_dir.name}.config.inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS

# Use in your Isaac Lab environment configuration
env_cfg.scene.robot = INSPIRE_HAND_CFG

# Add contact sensors
for sensor_name, sensor_cfg in CONTACT_SENSOR_CFGS.items():
    setattr(env_cfg.scene, f"contact_{{sensor_name}}", sensor_cfg)
```

### 2. Run Example Demo

```bash
cd examples/
./isaaclab.sh -p basic_demo.py --num_envs 1
```

### 3. Integration with Existing Projects

Copy the entire `{self.package_dir.name}` directory to your Isaac Lab workspace and import the configuration:

```python
import sys
sys.path.append("path/to/{self.package_dir.name}")
from config.inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS
```

## File Descriptions

### Configuration Files
- `config/inspire_hand_cfg.py`: Main Isaac Lab configuration with robot and sensor definitions
- `urdf/inspire_hand_processed.urdf`: Processed URDF file optimized for Isaac Lab

### USD Files
- `usd/inspire_hand_with_sensors.usd`: Complete USD asset for Isaac Sim

### Examples
- `examples/basic_demo.py`: Basic demonstration script
- More examples can be added as needed

## Requirements

- Isaac Lab (latest version)
- Isaac Sim 4.5.0 or later
- Python 3.10+

## Notes

- The hand is configured for right-hand operation
- Contact sensors provide force feedback in world coordinates
- Joint limits and actuator settings are optimized for stable grasping
- Self-collision is disabled to prevent finger interference
- Fixed for Isaac Sim 4.5 API compatibility

## Credits

Original URDF model: `urdf_right_with_force_sensor`
Isaac Lab integration: Auto-generated asset package (Fixed Version)

Created on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"  ✓ Main README created: README.md")
    
    def create_package_info(self):
        """Create package metadata file."""
        info_file = self.package_dir / "package_info.json"
        
        print(f"\n📄 Creating package metadata...")
        
        info_data = {
            "name": self.package_dir.name,
            "version": "1.0.1",
            "description": "Inspire Hand with 17 contact sensors for Isaac Lab (Fixed Version)",
            "author": "Auto-generated (Fixed)",
            "created": __import__('datetime').datetime.now().isoformat(),
            "isaac_lab_version": ">=1.0.0",
            "isaac_sim_version": ">=4.5.0",
            "fixes": [
                "USD conversion API compatibility with Isaac Sim 4.5",
                "Fixed variable scoping in contact sensor configuration",
                "Removed unsupported URDF import parameters",
                "Added error handling and detailed logging"
            ],
            "components": {
                "robot": "Inspire Hand (Right)",
                "joints": 11,
                "contact_sensors": 17,
                "meshes": "30 STL files",
                "urdf": "Processed for Isaac Lab compatibility"
            },
            "contact_sensors": {
                "palm": 1,
                "thumb": 4,
                "index": 3,
                "middle": 3,
                "ring": 3,
                "little": 3
            },
            "files": {
                "config": "config/inspire_hand_cfg.py",
                "urdf": "urdf/inspire_hand_processed.urdf",
                "usd": "usd/inspire_hand_with_sensors.usd",
                "example": "examples/basic_demo.py"
            }
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        print(f"  ✓ Package metadata created: {info_file.name}")
    
    def create_complete_package(self):
        """Create the complete asset package with fixes."""
        print("=" * 70)
        print("🚀 Creating Complete Isaac Lab Asset Package for Inspire Hand (FIXED)")
        print("=" * 70)
        
        # Step 1: Verify source files
        if not self.verify_source_files():
            return False
        
        # Step 2: Create package structure
        self.create_package_structure()
        
        # Step 3: Copy source files
        self.copy_source_files()
        
        # Step 4: Process URDF
        processed_urdf = self.preprocess_urdf_for_package()
        
        # Step 5: Convert to USD (with fixes)
        usd_file = self.convert_to_usd(processed_urdf)
        
        # Step 6: Create Isaac Lab config (with fixes)
        config_file = self.create_isaac_lab_config(usd_file)
        
        # Step 7: Create example scripts
        self.create_example_scripts()
        
        # Step 8: Create documentation
        self.create_documentation()
        
        # Step 9: Create package metadata
        self.create_package_info()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 PACKAGE CREATION COMPLETE (FIXED VERSION)")
        print("=" * 70)
        print(f"📦 Package Location: {self.package_dir}")
        print(f"📁 Package Size: {self._get_package_size()}")
        print(f"📄 Files Created: {self._count_files()}")
        
        # Check if USD was created
        usd_path = self.package_usd / "inspire_hand_with_sensors.usd"
        if usd_path.exists():
            print(f"✅ USD file successfully created: {usd_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print(f"⚠️  USD file not created - run with Isaac Sim Python environment")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Copy package to your Isaac Lab workspace")
        print(f"2. Test with: ./isaaclab.sh -p {self.package_dir.name}/examples/basic_demo.py")
        print(f"3. Integrate into your own environments using the config files")
        print(f"4. Check package_info.json for details about fixes applied")
        
        return True
    
    def _get_package_size(self):
        """Get package size in MB."""
        total_size = 0
        for root, dirs, files in os.walk(self.package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return f"{total_size / (1024*1024):.1f} MB"
    
    def _count_files(self):
        """Count total files in package."""
        count = 0
        for root, dirs, files in os.walk(self.package_dir):
            count += len(files)
        return count


def main():
    """Main function."""
    print("🤖 Inspire Hand Asset Package Creator (Fixed Version)")
    print("    Compatible with Isaac Sim 4.5+")
    print()
    
    creator = InspireHandAssetPackageCreatorFixed()
    success = creator.create_complete_package()
    
    # Close Isaac Sim if it was started
    if ISAAC_SIM_AVAILABLE:
        simulation_app.close()
    
    return success


if __name__ == "__main__":
    main() 