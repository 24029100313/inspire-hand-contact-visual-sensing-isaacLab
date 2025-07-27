# Inspire Hand with Force Sensors

This project contains the complete Inspire Hand model with **997 tactile sensor pads** for Isaac Sim.

## 🚀 Quick Start
1. Load URDF: `urdf/inspire_hand_processed_with_pads.urdf`
2. Convert to USD using provided scripts: `convert_urdf_to_usd_with_thumb4_pads.py`
3. Import into Isaac Sim for simulation

## 📊 Current Sensor Count: **997 pads** across **17 sensors**

### Sensor Distribution:
- **Palm**: 112 pads (14×8, 3.0×3.0×0.6mm)
- **Thumb1**: 96 pads (8×12, 1.2×1.2×0.6mm)
- **Thumb2**: 8 pads (2×4, 1.2×1.2×0.6mm)
- **Thumb3**: 96 pads (8×12, 1.2×1.2×0.6mm)
- **Thumb4**: 9 pads (3×3, 1.2×1.2×0.6mm) ✅ **NEW**
- **Index1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Index2**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Index3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Middle1**: 80 pads (10×8, 1.2×1.2×0.6mm)
- **Middle2**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Middle3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Ring1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Ring2**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Ring3**: 9 pads (3×3, 1.2×1.2×0.6mm)
- **Little1**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Little2**: 80 pads (8×10, 1.2×1.2×0.6mm)
- **Little3**: 9 pads (3×3, 1.2×1.2×0.6mm)

## 📁 Project Structure
```
inspire_hand_with_sensors/
├── urdf/                           # URDF files
│   └── inspire_hand_processed_with_pads.urdf  # Main URDF with all sensors
├── usd/                            # USD files for Isaac Sim
│   └── inspire_hand_processed_with_pads.usd   # Complete USD model (21MB)
├── config/                         # Isaac Lab configurations
│   └── inspire_hand_processed_with_pads.yaml  # Sensor configurations
├── meshes/                         # STL mesh files
├── textures/                       # Texture files
├── convert_urdf_to_usd_with_thumb4_pads.py    # USD conversion script
└── README_SENSOR_PAD_PROCESS.md    # Detailed process documentation
```

## 🔧 Usage

### Convert URDF to USD
```bash
# Using Isaac Sim Python environment
/path/to/isaac-sim/python.sh convert_urdf_to_usd_with_thumb4_pads.py
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
- **Latest**: Added Thumb4 force sensor with 9 pads (3×3 layout)
- Updated total sensor count to 997 pads
- Generated new USD file with all sensors
- Updated Isaac Lab configuration

## 🔗 Dependencies
- Isaac Sim 4.5+
- Python 3.10+
- Isaac Lab (optional)
- NVIDIA RTX GPU (recommended)

## 📚 Documentation
See `README_SENSOR_PAD_PROCESS.md` for detailed sensor addition process and methodology.

---
**Total Contact Points**: 997 tactile sensor pads  
**Force Sensors**: 17 sensors  
**Uniform Thickness**: 0.6mm  
**Force Threshold**: 15g (0.147N)
