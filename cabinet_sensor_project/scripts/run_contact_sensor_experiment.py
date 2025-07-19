#!/usr/bin/env python3
"""
Contact Sensor Experiment Runner

This script runs the contact sensor learning experiment that demonstrates:
1. Multiple contact sensors in square arrangement
2. 1kg cube falling and contacting sensors
3. Physics verification (total force = gravity)
4. Multi-sensor data collection

Usage:
    python scripts/run_contact_sensor_experiment.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning_experiments.contact_sensor_experiment import run_contact_sensor_experiment

def main():
    """Main function to run the contact sensor experiment."""
    print("=" * 60)
    print("🔬 CONTACT SENSOR LEARNING EXPERIMENT")
    print("=" * 60)
    print()
    print("📋 Experiment Overview:")
    print("• Four contact sensors arranged in a square (1.2m x 1.2m)")
    print("• 1kg cube (1m x 1m x 0.2m) falling from 2m height")
    print("• Physics verification: Σ(sensor forces) = gravity (9.81N)")
    print("• Learning multi-sensor setup for robotic applications")
    print()
    
    # Confirm start
    try:
        input("Press Enter to start experiment (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n❌ Experiment cancelled by user")
        return
    
    print("\n🚀 Starting experiment...")
    
    try:
        # Run the experiment
        run_contact_sensor_experiment()
        
        print("\n✅ Experiment completed successfully!")
        print("🎓 Learning objectives achieved:")
        print("• ✓ Multi-sensor setup demonstrated")
        print("• ✓ Contact force measurement learned")
        print("• ✓ Physics verification performed")
        print("• ✓ Ready for robotic arm sensor integration")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        print("💡 Check that IsaacLab is properly installed and configured")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
