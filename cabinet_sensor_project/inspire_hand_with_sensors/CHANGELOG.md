# Changelog

All notable changes to the Inspire Hand with Force Sensors project will be documented in this file.

## [2.0.0] - 2025-07-27

### 🚀 Major Updates
- **Enhanced Sensor Layout**: Optimized 4 finger sensors for better tactile coverage
- **Increased Total Pads**: From 997 to **1061 pads** (+64 pads)
- **Layout Transformation**: Changed 4 sensors from 12×8 to 8×12 configuration

### ✅ Changed
| Sensor | Previous Layout | New Layout | Pad Count |
|--------|----------------|------------|-----------|
| Index2 | 12×8 (80 pads) | 8×12 (96 pads) | +16 |
| Middle2| 12×8 (80 pads) | 8×12 (96 pads) | +16 |
| Ring2  | 12×8 (80 pads) | 8×12 (96 pads) | +16 |
| Little2| 12×8 (80 pads) | 8×12 (96 pads) | +16 |

### 🔧 Technical Details
- **Pad Dimensions**: Maintained 1.2×1.2×0.6mm uniform size
- **Spacing**: 1.32mm between pad centers
- **Layout**: 8 columns × 12 rows for better finger length coverage
- **Colors**: Preserved original sensor color schemes
- **XML Validation**: Passed without errors

### 📁 Files Updated
- `urdf/inspire_hand_processed_with_pads.urdf` - Main URDF with updated sensors
- `usd/inspire_hand_processed_with_pads.usd` - New USD file (21.0 MB)
- `config/inspire_hand_processed_with_pads.yaml` - Updated Isaac Lab config
- `convert_urdf_to_usd_with_1061_pads.py` - Updated conversion script
- `README.md` - Updated documentation

### 🛠️ Process
1. Generated new pad configurations using Python automation
2. Validated XML syntax with xmllint
3. Successfully converted URDF to USD using Isaac Sim
4. Created backup of previous version
5. Updated all documentation

---

## [1.0.0] - 2025-07-23

### 🎉 Initial Release
- **Total Pads**: 997 tactile sensor pads across 17 sensors
- **Complete URDF**: Full inspire hand model with force sensors
- **USD Conversion**: Isaac Sim compatible USD files
- **Isaac Lab Support**: Configuration files for Isaac Lab
- **Documentation**: Complete setup and usage instructions

### 📊 Initial Sensor Distribution
- Palm: 112 pads (14×8)
- Thumb1-4: 209 pads total
- Index1-3: 169 pads total  
- Middle1-3: 169 pads total
- Ring1-3: 169 pads total
- Little1-3: 169 pads total

### 🔧 Features
- Uniform pad thickness: 0.6mm
- Force threshold: 15g (0.147N)
- Complete mesh and texture support
- Automatic USD conversion scripts 