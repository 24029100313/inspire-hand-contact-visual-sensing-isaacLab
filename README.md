# Robot Gripper Sensor Project

This repository contains robotics research projects focused on tactile sensing and manipulation using the Isaac Lab simulation environment.

## Project Structure

- **`cabinet_sensor_project/`** - Main project containing Franka Panda gripper with integrated contact sensors
- **`gripper_sensor_data/`** - Collected sensor data from experiments

## Key Components

### Contact Sensor Integration
- Modified Franka Panda URDF with 8 contact sensors (4 per finger)
- Real-time force visualization and data collection
- Tactile feedback for manipulation tasks

### Main Scripts
- `lift_cube_sm_with_sensors.py` - State machine for object lifting with sensor feedback
- `static_force_test.py` - Static force testing and calibration
- Various analysis and visualization tools

## Getting Started

1. Install Isaac Lab (follow official documentation)
2. Navigate to `cabinet_sensor_project/` for detailed usage instructions
3. Run experiments using the provided shell scripts

## Features

- Real-time contact force visualization
- Step-wise average force computation
- Comprehensive sensor data logging
- State machine-based manipulation control

## Documentation

See individual project folders for detailed documentation and usage guides.

## Dependencies

- Isaac Lab
- PyTorch
- NumPy
- Matplotlib
- OpenCV (for visualization)

---

*Developed for robotics research and tactile sensing applications* 