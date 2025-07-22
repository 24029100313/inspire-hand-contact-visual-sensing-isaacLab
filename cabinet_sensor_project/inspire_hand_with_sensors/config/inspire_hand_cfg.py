"""
Isaac Lab Configuration for Inspire Hand with Contact Sensors.

This configuration defines the Inspire Hand robot for use in Isaac Lab environments.
The hand includes 17 contact sensors distributed across fingers and palm.

Usage:
    from inspire_hand_with_sensors.config.inspire_hand_cfg import INSPIRE_HAND_CFG, CONTACT_SENSOR_CFGS
"""

import os
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg

# Get the path to the USD file (relative to this config file)
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
INSPIRE_HAND_USD_PATH = os.path.join(_CURRENT_DIR, "../usd/inspire_hand_with_sensors.usd")


##
# Robot Configuration
##

INSPIRE_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=INSPIRE_HAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Disable self-collision for multi-finger hand
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        joint_pos={
            # Initial joint positions - fingers slightly open for stable grasping
            "right_thumb_1_joint": 0.2,      # Thumb base joint
            "right_thumb_2_joint": 0.2,      # Thumb middle joint
            "right_thumb_3_joint": 0.16,     # Thumb tip (mimic joint)
            "right_index_1_joint": 0.1,      # Index base joint
            "right_index_2_joint": 0.1,      # Index tip joint
            "right_middle_1_joint": 0.1,     # Middle base joint
            "right_middle_2_joint": 0.1,     # Middle tip joint
            "right_ring_1_joint": 0.1,       # Ring base joint
            "right_ring_2_joint": 0.1,       # Ring tip joint
            "right_little_1_joint": 0.1,     # Little base joint
            "right_little_2_joint": 0.1,     # Little tip joint
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Thumb actuators - stronger for opposition
        "thumb": ImplicitActuatorCfg(
            joint_names_expr=["right_thumb_.*_joint"],
            effort_limit=10.0,    # Higher force limit for thumb
            velocity_limit=2.0,
            stiffness=100.0,      # Higher stiffness for precise control
            damping=10.0,
        ),
        # Index finger actuators
        "index": ImplicitActuatorCfg(
            joint_names_expr=["right_index_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Middle finger actuators
        "middle": ImplicitActuatorCfg(
            joint_names_expr=["right_middle_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Ring finger actuators
        "ring": ImplicitActuatorCfg(
            joint_names_expr=["right_ring_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # Little finger actuators
        "little": ImplicitActuatorCfg(
            joint_names_expr=["right_little_.*_joint"],
            effort_limit=5.0,
            velocity_limit=2.0,
            stiffness=50.0,
            damping=5.0,
        ),
    },
)


##
# Contact Sensor Configurations (Fixed variable scoping - all hardcoded strings)
##

CONTACT_SENSOR_CFGS = {}

# Palm contact sensor
CONTACT_SENSOR_CFGS["palm_contact"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/palm_force_sensor",
    track_pose=True,
    update_period=0.0,  # Update every simulation step
    debug_vis=True,
)

# Thumb contact sensors (4 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["thumb_sensor_1"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/thumb_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_2"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/thumb_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_3"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/thumb_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["thumb_sensor_4"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/thumb_force_sensor_4",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Index finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["index_sensor_1"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/index_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["index_sensor_2"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/index_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["index_sensor_3"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/index_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Middle finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["middle_sensor_1"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/middle_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["middle_sensor_2"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/middle_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["middle_sensor_3"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/middle_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Ring finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["ring_sensor_1"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/ring_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["ring_sensor_2"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/ring_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["ring_sensor_3"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/ring_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

# Little finger contact sensors (3 sensors) - All paths hardcoded as strings
CONTACT_SENSOR_CFGS["little_sensor_1"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/little_force_sensor_1",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["little_sensor_2"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/little_force_sensor_2",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)

CONTACT_SENSOR_CFGS["little_sensor_3"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/little_force_sensor_3",
    track_pose=True,
    update_period=0.0,
    debug_vis=True,
)


##
# Utility Functions
##

def get_contact_sensor_names():
    """Get list of all contact sensor names."""
    return list(CONTACT_SENSOR_CFGS.keys())


def print_hand_info():
    """Print information about the hand configuration."""
    print("Inspire Hand Configuration Summary:")
    print(f"  USD File: {INSPIRE_HAND_USD_PATH}")
    print(f"  Total Joints: {len(INSPIRE_HAND_CFG.init_state.joint_pos)}")
    print(f"  Total Contact Sensors: {len(CONTACT_SENSOR_CFGS)}")
    print("  Contact Sensors by Location:")
    print("    Palm: 1 sensor")
    print("    Thumb: 4 sensors")
    print("    Index: 3 sensors")
    print("    Middle: 3 sensors")
    print("    Ring: 3 sensors")
    print("    Little: 3 sensors")


if __name__ == "__main__":
    print_hand_info()
