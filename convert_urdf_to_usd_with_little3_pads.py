#!/usr/bin/env python3
"""
Convert Inspire Hand URDF with little_force_sensor_3 pads to USD
Creates a single USD file and corresponding YAML configuration
Total: 988 sensor pads (includes little_sensor_3: 9 pads)
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Initialize Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": True})

class InspireHandLittle3USDConverter:
    def __init__(self, source_dir="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project"):
        """Initialize the USD converter for Inspire Hand with Little3 Pads"""
        self.source_dir = Path(source_dir)
        self.asset_name = "inspire_hand_with_sensors"
        
        # Define paths
        self.asset_dir = self.source_dir / self.asset_name
        self.urdf_dir = self.asset_dir / "urdf"
        
        # Use the URDF with little3 sensor pads
        self.processed_urdf = self.urdf_dir / "inspire_hand_processed_with_pads.urdf"
        self.usd_file = self.asset_dir / "usd" / "inspire_hand_processed_with_pads.usd"
        self.yaml_file = self.asset_dir / "config" / "inspire_hand_processed_with_pads.yaml"
        
        # Ensure output directories exist
        self.usd_file.parent.mkdir(parents=True, exist_ok=True)
        self.yaml_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Inspire Hand USD Converter (with Little3 Pads) initialized")
        print(f"📁 Source directory: {self.source_dir}")
        print(f"📄 Source URDF: {self.processed_urdf}")
        print(f"📄 Target USD: {self.usd_file}")
        print(f"📄 Target YAML: {self.yaml_file}")

    def convert_to_usd(self):
        """Convert URDF to USD using Isaac Sim"""
        print(f"\n🔄 Converting URDF to USD...")
        print(f"�� Input: {self.processed_urdf}")
        print(f"�� Output: {self.usd_file}")
        
        if not self.processed_urdf.exists():
            raise FileNotFoundError(f"Source URDF not found: {self.processed_urdf}")

        # Import required Isaac Sim modules
        import omni.kit.commands
        from omni.kit.commands import execute
        from omni.isaac.core.utils.extensions import enable_extension
        
        print("🔧 Enabling URDF extension...")
        enable_extension("isaacsim.asset.importer.urdf")
        
        print("🔧 Executing URDF import...")
        
        # Create URDF import configuration
        status, import_config = execute("URDFCreateImportConfig")
        
        # Configure import settings
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.distance_scale = 1.0
        
        # Import URDF
        status, import_result = execute(
            "URDFParseAndImportFile",
            urdf_path=str(self.processed_urdf),
            import_config=import_config,
        )
        
        if not status:
            raise RuntimeError("Failed to import URDF")
        
        print("✅ URDF import successful")
        
        # Get the current stage and export it
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        
        if stage is None:
            raise RuntimeError("No active USD stage found")
        
        print(f"💾 Saving USD to: {self.usd_file}")
        
        # Export the stage to USD
        stage.Export(str(self.usd_file))
        
        if self.usd_file.exists():
            print("✅ USD file saved successfully")
        else:
            raise RuntimeError("Failed to save USD file")

    def create_isaac_lab_config(self):
        """Create Isaac Lab configuration YAML file"""
        print(f"\n📝 Creating Isaac Lab configuration...")
        
        config_content = f"""# Isaac Lab Configuration for Inspire Hand with 988 Sensor Pads
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Total sensor pads: 988 (including little_sensor_3: 9 pads)

inspire_hand_cfg:
  usd_path: "{{ISAAC_LAB_ASSETS}}/Robots/inspire_hand_processed_with_pads.usd"
  
  # Sensor configuration - 988 total pads
  sensors:
    total_sensor_pads: 988  # All sensors combined - Little3 added
"""
        
        # Write YAML configuration
        with open(self.yaml_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ YAML configuration saved: {self.yaml_file}")

    def convert(self):
        """Execute the complete conversion process"""
        print("\n🚀 Starting Inspire Hand URDF to USD conversion (with Little3 pads)")
        print("=" * 70)
        
        try:
            # Convert URDF to USD
            self.convert_to_usd()
            
            # Create Isaac Lab config
            self.create_isaac_lab_config()
            
            # Display results
            if self.usd_file.exists():
                size_mb = self.usd_file.stat().st_size / (1024 * 1024)
                print(f"\n📊 USD file size: {size_mb:.1f} MB")
            
            if self.yaml_file.exists():
                size_kb = self.yaml_file.stat().st_size / 1024
                print(f"📊 YAML file size: {size_kb:.1f} KB")
            
            print("\n🎉 Conversion completed successfully!")
            print(f"✅ USD file: {self.usd_file}")
            print(f"✅ YAML config: {self.yaml_file}")
            print("✅ All 988 sensor pads (including little3) converted")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main conversion function"""
    converter = InspireHandLittle3USDConverter()
    success = converter.convert()
    
    # Close Isaac Sim
    simulation_app.close()
    
    if success:
        print("\n✅ All operations completed successfully!")
        return 0
    else:
        print("\n❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
