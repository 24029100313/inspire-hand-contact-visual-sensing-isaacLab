#!/usr/bin/env python3
"""
Isaac Lab Asset Package Creator for Inspire Hand with Contact Sensors
Converts processed URDF directly to USD format for Isaac Lab
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

class InspireHandUSDConverter:
    def __init__(self, source_dir="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project"):
        """Initialize the USD converter for Inspire Hand"""
        self.source_dir = Path(source_dir)
        self.asset_name = "inspire_hand_with_sensors"
        
        # Define paths
        self.asset_dir = self.source_dir / self.asset_name
        self.urdf_dir = self.asset_dir / "urdf"
        self.usd_dir = self.asset_dir / "usd"
        self.config_dir = self.asset_dir / "config"
        
        # Use the URDF with sensor pads
        self.processed_urdf = self.urdf_dir / "inspire_hand_processed_with_pads.urdf"
        self.usd_file = self.usd_dir / "inspire_hand_processed_with_pads.usd"
        
        print(f"🔧 Inspire Hand USD Converter initialized")
        print(f"📁 Source directory: {self.source_dir}")
        print(f"📁 Asset directory: {self.asset_dir}")
        print(f"📄 Source URDF: {self.processed_urdf}")
        print(f"📄 Target USD: {self.usd_file}")

    def setup_usd_directory(self):
        """Setup USD output directory"""
        self.usd_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ USD directory setup: {self.usd_dir}")

    def convert_to_usd(self):
        """Convert URDF to USD using Isaac Sim"""
        print(f"\n🔄 Converting URDF to USD...")
        print(f"📄 Input: {self.processed_urdf}")
        print(f"📄 Output: {self.usd_file}")
        
        if not self.processed_urdf.exists():
            raise FileNotFoundError(f"Source URDF not found: {self.processed_urdf}")
        
        # Setup USD directory
        self.setup_usd_directory()
        
        try:
            import omni.kit.commands
            from omni.isaac.core.utils.extensions import enable_extension
            import omni.usd
            
            # Enable necessary extensions
            enable_extension("isaacsim.asset.importer.urdf")
            
            print("🔧 Executing URDF import...")
            
            # Create URDF import configuration
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            
            # Configure import settings
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.self_collision = False
            import_config.create_physics_scene = True
            import_config.distance_scale = 1.0
            
            # Import URDF
            status, import_result = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(self.processed_urdf),
                import_config=import_config,
            )
            
            if not status:
                raise RuntimeError("Failed to import URDF")
            
            print("✅ URDF import successful")
            
            # Get the current stage and export it
            from omni.usd import get_context
            usd_context = get_context()
            stage = usd_context.get_stage()
            
            # Export the stage to USD
            stage.Export(str(self.usd_file))
            print(f"✅ USD export successful: {self.usd_file}")
            
        except ImportError as e:
            print(f"❌ Isaac Sim import error: {e}")
            print("Please ensure this script is run with Isaac Sim's Python environment")
            raise
        except Exception as e:
            print(f"❌ Conversion error: {e}")
            raise

    def create_isaac_lab_config(self):
        """Create Isaac Lab configuration YAML"""
        config_content = f'''# Isaac Lab Asset Configuration - 14x8 Sensor Array
# Generated from: {self.processed_urdf.name}
# Creation date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

inspire_hand_with_sensors:
  class_type: RigidObject
  
  # USD file path (relative to this config file)
  usd_path: "../usd/{self.usd_file.name}"
  
  # Physics properties
  physics:
    rigid_body_enabled: true
    kinematic_enabled: false
    disable_gravity: false
    
  # Contact sensor configuration
  contact_sensors:
    # Palm force sensor (original)
    palm_force_sensor:
      prim_path: "/inspire_hand_with_sensors/palm_force_sensor"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.098  # 10g in Newtons (10g * 9.8m/s²)
      
    # Palm sensor pads (112 individual sensors in 14x8 array)
    palm_sensor_pads:
      prim_path: "/inspire_hand_with_sensors/palm_sensor_pad_*"  # Wildcard for all pads
      update_period: 0.005  # 200 FPS 
      force_threshold: 0.098  # 10g trigger force
      torque_threshold: 0.1
      
  # Joint configuration
  joints:
    # Thumb joints
    right_thumb_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_thumb_2_joint:
      drive_type: "angular" 
      max_effort: 50.0
      max_velocity: 10.0
      
    right_thumb_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    # Index finger joints
    right_index_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_index_2_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_index_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    # Middle finger joints
    right_middle_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_middle_2_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_middle_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    # Ring finger joints
    right_ring_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_ring_2_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_ring_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    # Pinky joints
    right_pinky_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_pinky_2_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_pinky_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0

# Sensor pad specifications for 14x8 array
sensor_specifications:
  palm_sensor_pads:
    count: 112  # 14x8 grid
    dimensions: [0.003, 0.003, 0.0012]  # 3x3x1.2mm in meters
    resolution: "3x3mm"
    trigger_force: "10g"
    max_force: "20N" 
    sampling_rate: "200FPS"
    coverage_area: [0.042, 0.024]  # 42x24mm in meters
    layout:
      grid_size: [14, 8]
      spacing: [0.003, 0.003]  # 3x3mm spacing (tight pack)
      arrangement: "紧密排列"
'''
        
        config_file = self.config_dir / "inspire_hand_processed_with_pads.yaml"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ Isaac Lab config created: {config_file}")
        return config_file

    def run_conversion(self):
        """Run the complete conversion process"""
        print("🚀 Starting Inspire Hand USD conversion process...")
        print("=" * 60)
        
        try:
            # Step 1: Convert to USD
            print("\n📋 Step 1: Converting URDF to USD")
            self.convert_to_usd()
            
            # Step 2: Create Isaac Lab config
            print("\n📋 Step 2: Creating Isaac Lab configuration")
            config_file = self.create_isaac_lab_config()
            
            # Step 3: Summary
            print("\n🎉 Conversion completed successfully!")
            print("=" * 60)
            print(f"📄 Generated USD: {self.usd_file}")
            print(f"📄 Generated Config: {config_file}")
            print(f"📊 Total sensor pads: 112 (14×8 grid)")
            print(f"📊 Sensor specs: 3×3×1.2mm, 10g trigger, 20N max, 200FPS")
            print(f"📊 Coverage area: 42×24×1.2mm")
            
            # File sizes
            if self.usd_file.exists():
                usd_size = self.usd_file.stat().st_size / 1024 / 1024
                print(f"💾 USD file size: {usd_size:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            return False

def main():
    """Main function"""
    print("🤖 Isaac Lab Inspire Hand USD Converter (14x8 Sensor Array)")
    print("Converting URDF with 112 sensor pads to USD format")
    print("=" * 60)
    
    try:
        converter = InspireHandUSDConverter()
        success = converter.run_conversion()
        
        if success:
            print("\n✅ All operations completed successfully!")
            simulation_app.close()
            sys.exit(0)
        else:
            print("\n❌ Conversion failed!")
            simulation_app.close()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        simulation_app.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        simulation_app.close()
        sys.exit(1)

if __name__ == "__main__":
    main() 