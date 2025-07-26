#!/usr/bin/env python3
"""
Convert Inspire Hand URDF with Thumb Force Sensor 2 Pads to USD
Creates a single USD file and corresponding YAML configuration
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

class InspireHandThumb2USDConverter:
    def __init__(self, source_dir="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project"):
        """Initialize the USD converter for Inspire Hand with Thumb2 Pads"""
        self.source_dir = Path(source_dir)
        self.asset_name = "inspire_hand_with_sensors"
        
        # Define paths
        self.asset_dir = self.source_dir / self.asset_name
        self.urdf_dir = self.asset_dir / "urdf"
        
        # Use the URDF with sensor pads including thumb2
        self.processed_urdf = self.urdf_dir / "inspire_hand_processed_with_pads.urdf"
        self.usd_file = Path.cwd() / "inspire_hand_with_thumb2_pads.usd"
        self.yaml_file = Path.cwd() / "inspire_hand_with_thumb2_pads.yaml"
        
        print(f"🔧 Inspire Hand USD Converter (with Thumb2 Pads) initialized")
        print(f"📁 Source directory: {self.source_dir}")
        print(f"📄 Source URDF: {self.processed_urdf}")
        print(f"📄 Target USD: {self.usd_file}")
        print(f"📄 Target YAML: {self.yaml_file}")

    def convert_to_usd(self):
        """Convert URDF to USD using Isaac Sim"""
        print(f"\n🔄 Converting URDF to USD...")
        print(f"📄 Input: {self.processed_urdf}")
        print(f"📄 Output: {self.usd_file}")
        
        if not self.processed_urdf.exists():
            raise FileNotFoundError(f"Source URDF not found: {self.processed_urdf}")
        
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
        """Create Isaac Lab configuration YAML with Thumb2 pads"""
        config_content = f'''# Isaac Lab Asset Configuration - Inspire Hand with Thumb Force Sensor 2 Pads
# Generated from: {self.processed_urdf.name}
# Creation date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

inspire_hand_with_thumb2_pads:
  class_type: RigidObject
  
  # USD file path (same directory as this config file)
  usd_path: "{self.usd_file.name}"
  
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
      force_threshold: 0.098  # 10g in Newtons
      
    # Palm sensor pads (14x8 = 112 individual sensors)
    palm_sensor_pads:
      prim_path: "/inspire_hand_with_sensors/palm_sensor_pad_*"
      update_period: 0.005  # 200 FPS 
      force_threshold: 0.098  # 10g trigger force
      torque_threshold: 0.1
      
    # Thumb Force Sensor 1 with pads (8x12 = 96 sensors)
    thumb_force_sensor_1:
      prim_path: "/inspire_hand_with_sensors/thumb_force_sensor_1"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g in Newtons
      
    thumb_sensor_1_pads:
      prim_path: "/inspire_hand_with_sensors/thumb_sensor_1_pad_*"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g trigger force
      torque_threshold: 0.1
      
    # Thumb Force Sensor 2 with pads (2x4 = 8 sensors) - NEW
    thumb_force_sensor_2:
      prim_path: "/inspire_hand_with_sensors/thumb_force_sensor_2" 
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g in Newtons
      
    thumb_sensor_2_pads:
      prim_path: "/inspire_hand_with_sensors/thumb_sensor_2_pad_*"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g trigger force  
      torque_threshold: 0.1
      
    # Additional thumb sensors (3 and 4)
    thumb_force_sensor_3:
      prim_path: "/inspire_hand_with_sensors/thumb_force_sensor_3"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g in Newtons
      
    thumb_force_sensor_4:
      prim_path: "/inspire_hand_with_sensors/thumb_force_sensor_4"
      update_period: 0.005  # 200 FPS
      force_threshold: 0.147  # 15g in Newtons
      
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
      
    right_thumb_4_joint:
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
      
    # Little finger joints  
    right_little_1_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_little_2_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0
      
    right_little_3_joint:
      drive_type: "angular"
      max_effort: 50.0
      max_velocity: 10.0

# Comprehensive sensor pad specifications
sensor_specifications:
  palm_sensor_pads:
    count: 112  # 14x8 grid
    dimensions: [0.003, 0.003, 0.0015]  # 3x3x1.5mm in meters
    resolution: "3x3mm"
    trigger_force: "10g"
    max_force: "20N" 
    sampling_rate: "200FPS"
    coverage_area: [0.042, 0.024]  # 42x24mm in meters
    layout:
      grid_size: [14, 8]
      spacing: [0.003, 0.003]  # 3x3mm spacing
      
  thumb_sensor_1_pads:
    count: 96  # 8x12 grid
    dimensions: [0.0012, 0.0012, 0.0006]  # 1.2x1.2x0.6mm in meters
    resolution: "1.2x1.2mm"
    trigger_force: "15g"
    max_force: "20N"
    sampling_rate: "200FPS"
    coverage_area: [0.0096, 0.0144]  # 9.6x14.4mm in meters
    layout:
      grid_size: [8, 12]
      spacing: [0.0012, 0.0012]  # 1.2x1.2mm spacing
      
  thumb_sensor_2_pads:  # NEW
    count: 8  # 2x4 grid  
    dimensions: [0.0012, 0.0012, 0.0006]  # 1.2x1.2x0.6mm in meters
    resolution: "1.2x1.2mm"
    trigger_force: "15g"
    max_force: "20N"
    sampling_rate: "200FPS"
    coverage_area: [0.0024, 0.0048]  # 2.4x4.8mm in meters
    layout:
      grid_size: [2, 4]
      spacing: [0.0012, 0.0012]  # 1.2x1.2mm spacing

# Summary
total_sensors:
  force_sensors: 4  # thumb_force_sensor_1,2,3,4
  sensor_pads: 216  # 112 palm + 96 thumb1 + 8 thumb2 = 216 total
  total_contact_points: 220  # 4 + 216
'''
        
        with open(self.yaml_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"✅ Isaac Lab config created: {self.yaml_file}")
        return self.yaml_file

    def run_conversion(self):
        """Run the complete conversion process"""
        print("🚀 Starting Inspire Hand USD conversion with Thumb2 Pads...")
        print("=" * 70)
        
        try:
            # Step 1: Convert to USD
            print("\n📋 Step 1: Converting URDF to USD")
            self.convert_to_usd()
            
            # Step 2: Create Isaac Lab config
            print("\n📋 Step 2: Creating Isaac Lab configuration")
            config_file = self.create_isaac_lab_config()
            
            # Step 3: Summary
            print("\n🎉 Conversion completed successfully!")
            print("=" * 70)
            print(f"📄 Generated USD: {self.usd_file}")
            print(f"📄 Generated Config: {config_file}")
            print()
            print("📊 Sensor Summary:")
            print(f"   • Palm sensor pads: 112 (14×8 grid, 3×3mm)")
            print(f"   • Thumb1 sensor pads: 96 (8×12 grid, 1.2×1.2mm)")  
            print(f"   • Thumb2 sensor pads: 8 (2×4 grid, 1.2×1.2mm) [NEW]")
            print(f"   • Force sensors: 4 (thumb_force_sensor_1,2,3,4)")
            print(f"   • Total contact points: 220")
            print()
            print("🔧 Specifications:")
            print(f"   • Thumb2 trigger force: 15g")
            print(f"   • Thumb2 max force: 20N")
            print(f"   • Sampling rate: 200FPS")
            
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
    print("🤖 Isaac Lab Inspire Hand USD Converter")
    print("Converting URDF with Thumb Force Sensor 2 Pads (2×4 = 8 pads)")
    print("=" * 70)
    
    try:
        converter = InspireHandThumb2USDConverter()
        success = converter.run_conversion()
        
        if success:
            print("\n✅ All operations completed successfully!")
            print("📂 Files generated in current directory:")
            print("   • inspire_hand_with_thumb2_pads.usd")
            print("   • inspire_hand_with_thumb2_pads.yaml")
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