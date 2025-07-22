# Inspire Hand With Sensors

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
inspire_hand_with_sensors/
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
from inspire_hand_with_sensors.config.inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS

# Use in your Isaac Lab environment configuration
env_cfg.scene.robot = INSPIRE_HAND_CFG

# Add contact sensors
for sensor_name, sensor_cfg in CONTACT_SENSOR_CFGS.items():
    setattr(env_cfg.scene, f"contact_{sensor_name}", sensor_cfg)
```

### 2. Run Example Demo

```bash
cd examples/
./isaaclab.sh -p basic_demo.py --num_envs 1
```

### 3. Integration with Existing Projects

Copy the entire `inspire_hand_with_sensors` directory to your Isaac Lab workspace and import the configuration:

```python
import sys
sys.path.append("path/to/inspire_hand_with_sensors")
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

Created on: 2025-07-22 22:50:20
