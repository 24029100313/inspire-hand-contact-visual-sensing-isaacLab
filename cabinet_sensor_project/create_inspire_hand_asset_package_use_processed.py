#!/usr/bin/env python3
"""
Modified Script to create USD from existing inspire_hand_processed.urdf

This script uses the already processed URDF file to generate USD format.

Usage:
    /path/to/isaac-sim/python.sh create_inspire_hand_asset_package_use_processed.py
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


class InspireHandUSDConverter:
    """Converts existing processed URDF to USD format."""
    
    def __init__(self):
        """Initialize paths."""
        # Source paths
        self.source_dir = Path("/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project")
        self.package_dir = self.source_dir / "inspire_hand_with_sensors"
        self.package_urdf = self.package_dir / "urdf"
        self.package_meshes = self.package_dir / "meshes"
        
        # Use the existing processed URDF
        self.processed_urdf = self.package_urdf / "inspire_hand_processed.urdf"
        
        # USD output
        self.usd_dir = self.package_dir / "usd"
        self.usd_file = self.usd_dir / "inspire_hand_processed.usd"
        
        print(f"📁 Package directory: {self.package_dir}")
        print(f"📄 Source URDF: {self.processed_urdf}")
        print(f"📄 Target USD: {self.usd_file}")

    def setup_usd_directory(self):
        """Create USD output directory."""
        self.usd_dir.mkdir(exist_ok=True)
        print(f"✓ Created USD directory: {self.usd_dir}")

    def convert_to_usd(self):
        """Convert processed URDF to USD format."""
        if not ISAAC_SIM_AVAILABLE:
            print("⚠️  Isaac Sim not available - skipping USD conversion")
            print("   Run this script from Isaac Sim environment to generate USD file")
            return None

        if not self.processed_urdf.exists():
            print(f"❌ Processed URDF not found: {self.processed_urdf}")
            return None
            
        print(f"\n🔄 Converting URDF to USD...")
        print(f"   Source: {self.processed_urdf.name}")
        print(f"   Target: {self.usd_file}")
        
        try:
            # Setup import configuration
            print("  Setting up URDF import configuration...")
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            
            # Basic configuration parameters
            import_config.merge_fixed_joints = False  # Keep sensor joints separate
            import_config.convex_decomp = False  # Use original meshes
            import_config.import_inertia_tensor = True  # Import inertial properties
            import_config.fix_base = True  # Fix the base link
            import_config.distance_scale = 1.0  # Keep original scale
            
            print("  ✓ Import configuration setup complete")

            # Import URDF to USD
            print(f"  Importing URDF from: {str(self.processed_urdf)}")
            
            status, import_result = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(self.processed_urdf),
                import_config=import_config,
            )
            
            if not status:
                print("❌ Failed to import URDF")
                return None
                
            print("  ✓ URDF imported to stage")

            # Save as USD
            print(f"  Saving USD to: {str(self.usd_file)}")
            
            # Ensure the directory exists
            self.usd_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get the current stage and export it
            from omni.usd import get_context
            usd_context = get_context()
            stage = usd_context.get_stage()
            
            # Export the stage to USD
            stage.Export(str(self.usd_file))
            
            print(f"  ✓ USD file saved: {self.usd_file.name}")
            
            # Verify the file was created
            if self.usd_file.exists():
                file_size = self.usd_file.stat().st_size / 1024  # KB
                print(f"  ✓ File verification passed ({file_size:.1f} KB)")
                return self.usd_file
            else:
                print("  ❌ USD file was not created")
                return None
                
        except Exception as e:
            print(f"❌ Error during USD conversion: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_isaac_lab_config(self, usd_file):
        """Create Isaac Lab configuration file for the converted asset."""
        if not usd_file:
            return None
            
        config_dir = self.package_dir / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "inspire_hand_processed.yaml"
        
        config_content = f"""# Isaac Lab Asset Configuration for Inspire Hand (Processed)
# Generated from: {self.processed_urdf.name}
# USD File: {usd_file.name}

asset:
  class_type: "RigidObject"
  usd_path: "${{ISAAC_LAB_ASSETS}}/inspire_hand_with_sensors/usd/{usd_file.name}"
  
  # Physics properties
  physics:
    rigid_body_enabled: true
    kinematic_enabled: false
    disable_gravity: false
    retain_accelerations: false
    
  # Collision properties  
  collision:
    collision_enabled: true
    contact_offset: 0.02
    rest_offset: 0.0
    
  # Material properties
  physics_material:
    static_friction: 0.5
    dynamic_friction: 0.5
    restitution: 0.0
    
  # Visual properties
  visual_material:
    emissive_color: [0.0, 0.0, 0.0]
    emissive_intensity: 1.0

# Sensor configuration
sensors:
  contact_sensors:
    palm_force_sensor:
      body_names: ["palm_force_sensor"]
      sensor_tick: 0
      
    finger_sensors:
      thumb:
        - body_names: ["thumb_force_sensor_1"]
        - body_names: ["thumb_force_sensor_2"] 
        - body_names: ["thumb_force_sensor_3"]
        - body_names: ["thumb_force_sensor_4"]
      index:
        - body_names: ["index_force_sensor_1"]
        - body_names: ["index_force_sensor_2"]
        - body_names: ["index_force_sensor_3"]
      middle:
        - body_names: ["middle_force_sensor_1"]
        - body_names: ["middle_force_sensor_2"]
        - body_names: ["middle_force_sensor_3"]
      ring:
        - body_names: ["ring_force_sensor_1"]
        - body_names: ["ring_force_sensor_2"]
        - body_names: ["ring_force_sensor_3"]
      little:
        - body_names: ["little_force_sensor_1"]
        - body_names: ["little_force_sensor_2"]
        - body_names: ["little_force_sensor_3"]

# Joint configuration
joints:
  # Thumb joints
  right_thumb_oppose_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 1.0472]  # 0 to 60 degrees
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  right_thumb_1_joint:
    drive_type: "angular" 
    initial_position: 0.0
    position_limits: [0.0, 1.4381]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  right_thumb_2_joint:
    drive_type: "angular"
    initial_position: 0.0  
    position_limits: [0.0, 3.14]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  # Index finger joints
  right_index_1_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 1.4381]
    velocity_limits: [1.0] 
    effort_limits: [10.0]
    
  right_index_2_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 3.14]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  # Middle finger joints  
  right_middle_1_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 1.4381]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  right_middle_2_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 3.14] 
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  # Ring finger joints
  right_ring_1_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 1.4381]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  right_ring_2_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 3.14]
    velocity_limits: [1.0] 
    effort_limits: [10.0]
    
  # Little finger joints
  right_little_1_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 1.4381]
    velocity_limits: [1.0]
    effort_limits: [10.0]
    
  right_little_2_joint:
    drive_type: "angular"
    initial_position: 0.0
    position_limits: [0.0, 3.14]
    velocity_limits: [1.0]
    effort_limits: [10.0]
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        print(f"✓ Created Isaac Lab config: {config_file}")
        return config_file

    def run_conversion(self):
        """Execute the complete conversion process."""
        print("🚀 Starting Inspire Hand USD Conversion (Using Processed URDF)")
        print("=" * 60)
        
        # Check if processed URDF exists
        if not self.processed_urdf.exists():
            print(f"❌ Processed URDF not found: {self.processed_urdf}")
            print("   Please run the original script first to create the processed URDF")
            return False
            
        # Step 1: Setup USD directory
        self.setup_usd_directory()
        
        # Step 2: Convert to USD
        usd_file = self.convert_to_usd()
        
        # Step 3: Create Isaac Lab config
        config_file = self.create_isaac_lab_config(usd_file)
        
        if usd_file:
            print("\n" + "=" * 60)
            print("✅ USD Conversion completed successfully!")
            print(f"📄 USD file: {usd_file}")
            if config_file:
                print(f"⚙️  Config file: {config_file}")
            print("\nNext steps:")
            print("1. Copy the inspire_hand_with_sensors folder to your Isaac Lab assets directory")
            print("2. Use the USD file in your Isaac Lab environments")
            print("3. Access contact sensor data through the provided configuration")
            return True
        else:
            print("\n❌ USD conversion failed")
            return False


def main():
    """Main execution function."""
    try:
        converter = InspireHandUSDConverter()
        success = converter.run_conversion()
        
        if ISAAC_SIM_AVAILABLE:
            simulation_app.close()
            
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        if ISAAC_SIM_AVAILABLE:
            simulation_app.close()
            
        sys.exit(1)


if __name__ == "__main__":
    main() 