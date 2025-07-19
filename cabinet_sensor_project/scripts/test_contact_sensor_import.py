#!/usr/bin/env python3
"""
Test script to verify contact sensor experiment imports work correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing contact sensor experiment imports...")
    
    try:
        # Test basic Python imports
        import numpy as np
        import torch
        print("✅ Basic Python packages imported successfully")
        
        # Test Isaac Lab imports (this will fail if Isaac Lab is not properly set up)
        try:
            import omni.isaac.lab.sim as sim_utils
            from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor
            print("✅ Isaac Lab imports successful")
        except ImportError as e:
            print(f"❌ Isaac Lab import failed: {e}")
            print("💡 Make sure Isaac Lab is properly installed and ISAAC_SIM_PATH is set")
            return False
        
        # Test our experiment module
        try:
            from learning_experiments.contact_sensor_experiment import ContactSensorExperiment, ContactSensorExperimentCfg
            print("✅ Contact sensor experiment module imported successfully")
        except ImportError as e:
            print(f"❌ Contact sensor experiment import failed: {e}")
            return False
        
        # Test configuration creation
        try:
            cfg = ContactSensorExperimentCfg()
            print("✅ Configuration object created successfully")
        except Exception as e:
            print(f"❌ Configuration creation failed: {e}")
            return False
        
        print("\n🎉 All imports successful! Ready to run the experiment.")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("🧪 CONTACT SENSOR EXPERIMENT IMPORT TEST")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n✅ Test passed! You can now run the experiment:")
        print("   python scripts/run_contact_sensor_experiment.py")
        return 0
    else:
        print("\n❌ Test failed! Please check your environment setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
