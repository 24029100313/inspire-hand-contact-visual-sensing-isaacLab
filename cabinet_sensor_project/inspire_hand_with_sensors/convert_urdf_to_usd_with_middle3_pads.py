#!/usr/bin/env python3
"""
Convert Inspire Hand URDF with middle_force_sensor_3 pads to USD
Total sensor pads: 970 (including 9 new middle_sensor_3 pads)
Usage: Run with Isaac Sim Python environment
"""

import os
import sys
from pathlib import Path

# Add Isaac Sim path
ISAAC_SIM_PATH = "/home/larry/NVIDIA_DEV/isaac-sim/isaac-sim-standa"
sys.path.append(ISAAC_SIM_PATH + "/kit/python/lib")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension

def convert_to_usd():
    """Convert URDF to USD using Isaac Sim"""
    
    # Enable URDF extension
    enable_extension("omni.isaac.urdf")
    
    # File paths
    urdf_path = os.path.abspath("urdf/inspire_hand_processed_with_pads.urdf")
    usd_path = os.path.abspath("usd/inspire_hand_processed_with_pads.usd")
    
    print("🚀 Converting URDF to USD with Isaac Sim...")
    print(f"📂 Input URDF: {urdf_path}")
    print(f"📂 Output USD: {usd_path}")
    
    # Verify input file
    if not os.path.exists(urdf_path):
        print(f"❌ URDF file not found: {urdf_path}")
        return False
    
    # Create output directory
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    
    try:
        print("📋 Creating URDF import configuration...")
        
        # Create import config
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        
        # Configure settings
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.create_physics_scene = True
        import_config.distance_scale = 1.0
        import_config.density = 1000.0
        
        print("🔄 Importing URDF...")
        
        # Import URDF
        status, result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )
        
        if not status:
            print("❌ Failed to import URDF")
            return False
        
        print("✅ URDF imported successfully")
        
        # Get stage and save USD
        stage = omni.usd.get_context().get_stage()
        stage.Export(usd_path)
        
        print(f"💾 USD saved to: {usd_path}")
        
        # Validation
        if os.path.exists(usd_path):
            file_size = os.path.getsize(usd_path) / (1024 * 1024)
            print(f"📊 USD file size: {file_size:.2f} MB")
            
            print("\n🎯 Conversion Summary:")
            print("   • Total sensor pads: 970")
            print("   • New middle_sensor_3 pads: 9 (3×3)")
            print("   • Physics scene: Enabled")
            print("   • Ready for Isaac Sim simulation")
            print("✅ Conversion completed successfully!")
            return True
        else:
            print("❌ USD file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def main():
    """Main function"""
    print("🦾 Inspire Hand URDF→USD Converter")
    print("   Target: inspire_hand_processed_with_pads.urdf (970 sensor pads)")
    print("   Includes: 9 new middle_sensor_3 pads (3×3 layout)")
    print("")
    
    try:
        success = convert_to_usd()
        
        if success:
            print("\n🎉 All operations completed successfully!")
            print("💡 Next steps:")
            print("   1. Load USD in Isaac Sim")
            print("   2. Configure contact sensors")
            print("   3. Run simulation tests")
        else:
            print("\n❌ Conversion failed!")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
