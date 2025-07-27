# Inspire Hand with Force Sensors

This project contains the complete Inspire Hand model with **1061 tactile sensor pads** for Isaac Sim.

## 🚀 Quick Start
1. Load URDF: `urdf/inspire_hand_processed_with_pads.urdf`
2. Convert to USD using provided scripts: `convert_urdf_to_usd_with_1061_pads.py`
3. Import into Isaac Sim for simulation

## 📊 Current Sensor Count: **1061 pads** across **17 sensors**

### Sensor Distribution:
- **Palm**: 112 pads (14×8, 3.0×3.0×0.6mm)
- **Thumb1**: 96 pads (8×12, 1.2×1.2×0.6mm)
- **Thumb2**: 8 pads (2×4, 1.2×1.2×0.6mm)
- **Thumb3**: 96 pads (8×12, 1.2×1.2×0.6mm)
- **Thumb4**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Index1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Index2**: 96 pads (8×12, 1.2×1.2×0.6mm) ✅ **UPDATED**
- **Index3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Middle1**: 80 pads (10×8, 1.2×1.2×0.6mm)
- **Middle2**: 96 pads (8×12, 1.2×1.2×0.6mm) ✅ **UPDATED**
- **Middle3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Ring1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Ring2**: 96 pads (8×12, 1.2×1.2×0.6mm) ✅ **UPDATED**
- **Ring3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Little1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Little2**: 96 pads (8×12, 1.2×1.2×0.6mm) ✅ **UPDATED**
- **Little3**: 9 pads (3×3, 1.2×1.2×0.6mm)

## 📁 Project Structure
```
inspire_hand_with_sensors/
├── urdf/                           # URDF files
│   ├── inspire_hand_processed_with_pads.urdf      # Main URDF with all sensors
│   └── inspire_hand_processed_with_pads.urdf.backup  # Backup of previous version
├── usd/                            # USD files for Isaac Sim
│   └── inspire_hand_processed_with_pads.usd       # Complete USD model (21MB)
├── config/                         # Isaac Lab configurations
│   └── inspire_hand_processed_with_pads.yaml      # Sensor configurations
├── meshes/                         # STL mesh files
├── textures/                       # Texture files
├── convert_urdf_to_usd_with_1061_pads.py          # USD conversion script
└── README_SENSOR_PAD_PROCESS.md    # Detailed process documentation
```

## 🔧 Usage

### Convert URDF to USD
```bash
# Using Isaac Sim Python environment
/path/to/isaac-sim/python.sh convert_urdf_to_usd_with_1061_pads.py
```

### Load in Isaac Lab
```python
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

# Load the inspire hand asset
inspire_hand_cfg = RigidObject.Config(
    usd_path="path/to/inspire_hand_processed_with_pads.usd",
    # ... other configurations
)
```

## 📋 Recent Updates
- **Latest v2.0**: Enhanced 4 finger sensors (Index2, Middle2, Ring2, Little2) from 8×10 to 8×12 layout
- Increased total sensor count from 997 to **1061 pads** (+64 pads)
- Updated sensor layout for better coverage and tactile resolution
- Generated new USD file (21.0 MB) with optimized sensor arrangements
- Updated Isaac Lab configuration with all 1061 sensor pads
- Maintained uniform pad dimensions: 1.2×1.2×0.6mm

## 🔧 Layout Changes Summary
| Sensor | Previous Layout | New Layout | Pad Count Change |
|--------|----------------|------------|------------------|
| Index2 | 8×10 → 12×8    | 8×12       | 80 → 96 (+16)   |
| Middle2| 8×10 → 12×8    | 8×12       | 80 → 96 (+16)   |
| Ring2  | 8×10 → 12×8    | 8×12       | 80 → 96 (+16)   |
| Little2| 8×10 → 12×8    | 8×12       | 80 → 96 (+16)   |

## 🔗 Dependencies
- Isaac Sim 4.5+
- Python 3.10+
- Isaac Lab (optional)
- NVIDIA RTX GPU (recommended)

## 📚 Documentation
See `README_SENSOR_PAD_PROCESS.md` for detailed sensor addition process and methodology.

---
**Total Contact Points**: 1061 tactile sensor pads  
**Force Sensors**: 17 sensors  
**Uniform Thickness**: 0.6mm  
**Force Threshold**: 15g (0.147N)  
**Last Updated**: July 27, 2025 - Layout optimization v2.0
