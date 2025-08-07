#!/usr/bin/env python3
"""
Script to expand inspire_hand_processed_with_specific_pads.urdf by adding
all missing sensor pads from inspire_hand_processed_with_pads.urdf

This will add:
- index_sensor_3_pad: 001-004, 006-009 (missing 8 pads, keeping 005)
- thumb_sensor_4_pad: 001-004, 006-009 (missing 8 pads, keeping 005)

Total: from 10 pads to 26 pads
"""

import re
import os

def extract_sensor_pad_definition(source_file, pad_name):
    """Extract complete sensor pad definition (link + joint) from source URDF."""
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Find the link definition
    link_pattern = rf'(<link name="{pad_name}">.*?</link>)'
    link_match = re.search(link_pattern, content, re.DOTALL)
    
    # Find the joint definition
    joint_pattern = rf'(<joint name="{pad_name}_joint".*?</joint>)'
    joint_match = re.search(joint_pattern, content, re.DOTALL)
    
    if link_match and joint_match:
        return f"\n{link_match.group(1)}\n{joint_match.group(1)}\n"
    else:
        print(f"Warning: Could not find definition for {pad_name}")
        return ""

def main():
    # File paths
    source_urdf = "/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/urdf/inspire_hand_processed_with_pads.urdf"
    target_urdf = "/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/urdf/inspire_hand_processed_with_specific_pads.urdf"
    
    # Define the sensor pads to add
    index_sensor_3_pads_to_add = [
        "index_sensor_3_pad_001", "index_sensor_3_pad_002", "index_sensor_3_pad_003", "index_sensor_3_pad_004",
        "index_sensor_3_pad_006", "index_sensor_3_pad_007", "index_sensor_3_pad_008", "index_sensor_3_pad_009"
    ]  # We already have 005
    
    thumb_sensor_4_pads_to_add = [
        "thumb_sensor_4_pad_001", "thumb_sensor_4_pad_002", "thumb_sensor_4_pad_003", "thumb_sensor_4_pad_004",
        "thumb_sensor_4_pad_006", "thumb_sensor_4_pad_007", "thumb_sensor_4_pad_008", "thumb_sensor_4_pad_009"
    ]  # We already have 005
    
    # Read current target URDF
    with open(target_urdf, 'r') as f:
        target_content = f.read()
    
    # Collect all sensor pad definitions to add
    new_sensor_definitions = []
    
    print("Extracting index_sensor_3_pad definitions...")
    for pad_name in index_sensor_3_pads_to_add:
        definition = extract_sensor_pad_definition(source_urdf, pad_name)
        if definition:
            new_sensor_definitions.append(definition)
            print(f"  ✓ {pad_name}")
    
    print("Extracting thumb_sensor_4_pad definitions...")
    for pad_name in thumb_sensor_4_pads_to_add:
        definition = extract_sensor_pad_definition(source_urdf, pad_name)
        if definition:
            new_sensor_definitions.append(definition)
            print(f"  ✓ {pad_name}")
    
    # Insert new definitions before </robot>
    all_new_definitions = "".join(new_sensor_definitions)
    updated_content = target_content.replace("</robot>", f"{all_new_definitions}\n</robot>")
    
    # Create backup
    backup_file = target_urdf.replace(".urdf", "_backup.urdf")
    with open(backup_file, 'w') as f:
        f.write(target_content)
    print(f"\n📁 Backup created: {backup_file}")
    
    # Write updated URDF
    with open(target_urdf, 'w') as f:
        f.write(updated_content)
    
    print(f"\n✅ Successfully expanded URDF!")
    print(f"📈 Added {len(new_sensor_definitions)} sensor pads")
    print(f"📊 Total sensor pads: 26 (was 10)")
    print(f"   - index_sensor_2_pad: 4 (unchanged)")
    print(f"   - index_sensor_3_pad: 9 (was 1, added 8)")
    print(f"   - thumb_sensor_3_pad: 4 (unchanged)")
    print(f"   - thumb_sensor_4_pad: 9 (was 1, added 8)")
    
    print(f"\n📄 Updated file: {target_urdf}")

if __name__ == "__main__":
    main() 