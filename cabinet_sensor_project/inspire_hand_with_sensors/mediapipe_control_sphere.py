#!/usr/bin/env python3
"""
MediaPipe-based control for Inspire Hand in Isaac Lab (sphere grasp scene)
- Uses inspire_hand_clone/mp_read_hand.HandDetector to read hand landmarks
- Maps to 6-channel 0..1000 commands (pinky, ring, middle, index, thumb, extra)
- Converts to articulation joint targets using USD soft limits and default pose
- Falls back to a synthetic open/close generator if MediaPipe is unavailable

Run:
    ./isaaclab.sh -p mediapipe_control_sphere.py --num_envs 1 --total_time 120
"""

import argparse
import threading
import time
from typing import List

import torch

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="MediaPipe control for Inspire Hand")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--total_time", type=float, default=120.0)
parser.add_argument("--target_hand", type=str, default="left", choices=["left", "right", "both"])
parser.add_argument("--no_mediapipe", action="store_true", help="Disable MediaPipe and use synthetic input")
parser.add_argument("--enforce_thumb_400", action="store_true", help="Force thumb channel to 400 baseline")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app = AppLauncher(args_cli).app

# Isaac imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )

    desktop: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desktop",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.7, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_pads.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.1, -0.045, 0.21),
            rot=(0.754649, -0.655439, 0.029940, 0.002793),
            joint_pos={
                "right_thumb_1_joint": 0.0,
                "right_thumb_2_joint": 0.0,
                "right_thumb_3_joint": 0.0,
                "right_thumb_4_joint": 0.0,
                "right_index_1_joint": 0.0,
                "right_index_2_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_middle_2_joint": 0.0,
                "right_ring_1_joint": 0.072,
                "right_ring_2_joint": 0.314,
                "right_little_1_joint": 0.0,
                "right_little_2_joint": 0.0,
            },
        ),
        actuators={
            "thumb_lock": ImplicitActuatorCfg(
                joint_names_expr=["right_thumb_.*_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=8.0,
                stiffness=2000.0,
                damping=200.0,
            ),
            "other_fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_(?!thumb).*_joint"],
                effort_limit_sim=80.0,
                velocity_limit_sim=8.0,
                stiffness=1200.0,
                damping=120.0,
            ),
        },
    )

    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.045,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=False, max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2), roughness=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.068, 0.12, 0.075), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class EnvCfg(DirectRLEnvCfg):
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=16,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=4,
        ),
    )
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=True)
    episode_length_s = 120.0
    decimation = 1
    observation_space = 50
    action_space = 6


class InspireHandMediapipeEnv(DirectRLEnv):
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg, render_mode: str | None = None, **kwargs):
        # Store sensor pad names for setup - get from cfg if available
        self.sensor_pad_names = list(getattr(cfg, "selected_sensor_names", []))
        
        print(f"[INFO] Initializing InspireHandMediapipeEnv with {len(self.sensor_pad_names)} sensor pads")
        if self.sensor_pad_names:
            print(f"[INFO] Contact sensors will be configured for all {len(self.sensor_pad_names)} pads")
        
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_pos_target = torch.zeros(self.num_envs, self.hand.num_joints, device=self.device)
        self._last_inspire6 = torch.zeros(self.num_envs, 6, device=self.device)
        # default thumb baseline (hardware semantics)
        if args_cli.enforce_thumb_400:
            self._last_inspire6[:, 4] = 400.0
        # No per-finger/joint inversion or explicit close-direction overrides (use auto selection)
        # start mediapipe thread
        self._mp_thread = None
        self._mp_running = False
        self._mp_lock = threading.Lock()
        self._latest_inspire6 = torch.full((self.num_envs, 6), -1.0, device=self.device)
        self._start_input_thread()

    def _start_input_thread(self):
        if args_cli.no_mediapipe:
            self._mp_running = True
            self._mp_thread = threading.Thread(target=self._synthetic_loop, daemon=True)
            self._mp_thread.start()
            return
        try:
            import sys
            import cv2  # noqa: F401
            sys.path.insert(0, "/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_clone")
            from mp_read_hand import HandDetector  # type: ignore
        except Exception:
            self._mp_running = True
            self._mp_thread = threading.Thread(target=self._synthetic_loop, daemon=True)
            self._mp_thread.start()
            return

        def mediapipe_loop():
            try:
                import cv2
                detector = HandDetector(target_hand=args_cli.target_hand)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                last_t = time.time()
                while self._mp_running:
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.05)
                        continue
                    landmarks, _ = detector.detect(frame)
                    if landmarks:
                        vals = detector.convert_fingure_to_inspire(landmarks[0])
                        if vals:
                            inspire6 = [
                                vals.get("pinky_finger", -1),
                                vals.get("ring_finger", -1),
                                vals.get("middle_finger", -1),
                                vals.get("index_finger", -1),
                                vals.get("thumb", 400 if args_cli.enforce_thumb_400 else -1),
                                vals.get("wrist", -1),
                            ]
                            with self._mp_lock:
                                self._latest_inspire6[:] = torch.tensor(inspire6, device=self.device).unsqueeze(0)
                    # limit rate ~30Hz
                    dt = time.time() - last_t
                    if dt < 1.0 / 30.0:
                        time.sleep(1.0 / 30.0 - dt)
                    last_t = time.time()
            except Exception:
                # fallback to synthetic if camera/mediapipe fails at runtime
                self._synthetic_loop()

        self._mp_running = True
        self._mp_thread = threading.Thread(target=mediapipe_loop, daemon=True)
        self._mp_thread.start()

    def _synthetic_loop(self):
        t = 0.0
        while self._mp_running:
            # open-close triangle wave 0..1000 over 10s
            phase = (t % 10.0) / 10.0
            if phase < 0.5:
                val = int(1000 * (phase / 0.5))
            else:
                val = int(1000 * ((1.0 - phase) / 0.5))
            inspire6 = [val, val, val, val, 400 if args_cli.enforce_thumb_400 else 400, -1]
            with self._mp_lock:
                self._latest_inspire6[:] = torch.tensor(inspire6, device=self.device).unsqueeze(0)
            time.sleep(1.0 / 30.0)
            t += 1.0 / 30.0

    def close(self):
        self._mp_running = False
        super().close()

    # ---- Env overrides ----
    def _setup_scene(self):
        self.hand = self.scene["hand"]
        self.sphere = self.scene["sphere"]
        self.desktop = self.scene["desktop"]
        
        # Setup contact sensors if they exist
        if hasattr(self, 'sensor_pad_names') and self.sensor_pad_names:
            print(f"[INFO] Setting up {len(self.sensor_pad_names)} contact sensors...")
            
            sensor_count = 0
            missing_sensors = []
            
            for sensor_name in self.sensor_pad_names:
                try:
                    sensor = self.scene[f"contact_{sensor_name}"]
                    sensor_count += 1
                    if sensor_count <= 10:  # Only log first 10 to avoid spam
                        print(f"[INFO] ✓ Sensor '{sensor_name}' configured successfully")
                except KeyError:
                    missing_sensors.append(sensor_name)
                    if len(missing_sensors) <= 10:  # Only log first 10 missing
                        print(f"[WARN] ✗ Sensor '{sensor_name}' not found in scene")
            
            print(f"[INFO] Successfully configured {sensor_count}/{len(self.sensor_pad_names)} sensors")
            if missing_sensors:
                print(f"[WARN] {len(missing_sensors)} sensors were not found (this is normal if USD pad names differ)")
                if len(missing_sensors) <= 10:
                    print(f"[WARN] Missing sensors: {missing_sensors}")
                else:
                    print(f"[WARN] Missing sensors (first 10): {missing_sensors[:10]} ...")
        else:
            print("[INFO] No contact sensors configured")

    def _process_contact_sensors(self):
        """Process contact sensor data and return summary statistics."""
        if not hasattr(self, 'sensor_pad_names') or not self.sensor_pad_names:
            return 0, 0, 0, {}
            
        total_sensors = len(self.sensor_pad_names)
        sensors_with_data = 0
        active_forces = 0
        force_data = {}
        
        for sensor_name in self.sensor_pad_names:
            try:
                sensor = self.scene[f"contact_{sensor_name}"]
                
                # Check if sensor has data
                if hasattr(sensor, 'data') and sensor.data is not None:
                    # Get net forces (this is the primary contact force data)
                    net_forces = sensor.data.net_forces_w
                    
                    if net_forces is not None and net_forces.shape[0] > 0:
                        sensors_with_data += 1
                        # Calculate force magnitude
                        force_magnitude = torch.norm(net_forces[0]).item()
                        force_data[sensor_name] = force_magnitude
                        
                        if force_magnitude > 0.001:  # 1mN threshold
                            active_forces += 1
                    else:
                        force_data[sensor_name] = 0.0
                else:
                    force_data[sensor_name] = 0.0
                    
            except KeyError:
                force_data[sensor_name] = 0.0
        
        return total_sensors, sensors_with_data, active_forces, force_data

    def _log_contact_forces(self, step_count):
        """Log contact forces summary."""
        total_sensors, sensors_with_data, active_forces, force_data = self._process_contact_sensors()
        
        if total_sensors > 0:
            print(f"\n=== Contact Forces at Step {step_count} ===")
            print(f"Sensors: {sensors_with_data}/{total_sensors} with data | Active contacts: {active_forces}")
            
            # Show top 10 active contacts
            active_items = [(name, force) for name, force in force_data.items() if force > 0.001]
            active_items.sort(key=lambda x: x[1], reverse=True)
            
            if active_items:
                print("Top active contacts:")
                for i, (name, force) in enumerate(active_items[:10]):
                    status = "🔴" if force > 0.1 else "⚡"
                    print(f"  {i+1:2d}. {name}: {force:.6f} N {status}")
            else:
                print("No active contacts detected.")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # read latest inspire6
        with self._mp_lock:
            inspire6 = self._latest_inspire6.clone()
        targets = self._map_inspire6_to_joint_targets(inspire6)
        self._joint_pos_target[:] = targets
        self.hand.set_joint_position_target(self._joint_pos_target)

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self.hand.data.joint_pos,
            self.hand.data.joint_vel,
            self.sphere.data.root_pos_w[:, :3],
            self.sphere.data.root_quat_w,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        # cache mapping
        if not hasattr(self, "name_to_index"):
            try:
                self.joint_names = list(self.hand.joint_names)
            except Exception:
                self.joint_names = [f"joint_{i}" for i in range(self.hand.num_joints)]
            self.name_to_index = {n: i for i, n in enumerate(self.joint_names)}
            limits = self.hand.data.soft_joint_pos_limits
            self.joint_min = limits[0, :, 0].to(self.device)
            self.joint_max = limits[0, :, 1].to(self.device)
        # write default state and cache default as open
        jp = self.hand.data.default_joint_pos[env_ids].clone()
        jv = self.hand.data.default_joint_vel[env_ids].clone()
        self.hand.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.joint_default = self.hand.data.joint_pos[0].to(self.device)
        self._joint_pos_target[env_ids] = self.hand.data.joint_pos[env_ids]

    # ---- Mapping helpers ----
    def _convert_open_fraction_to_radians(self, open_frac: torch.Tensor, joint_names: List[str]) -> torch.Tensor:
        open_frac = torch.clamp(open_frac, 0.0, 1.0)
        q_open_list = []
        q_close_list = []
        for name in joint_names:
            j = self.name_to_index.get(name, None)
            if j is None:
                q_open_list.append(torch.tensor(0.0, device=self.device))
                q_close_list.append(torch.tensor(0.0, device=self.device))
                continue
            q0 = self.joint_default[j]
            qmin = self.joint_min[j]
            qmax = self.joint_max[j]
            # Auto: choose the farther soft limit as "close" direction
            dmin = torch.abs(q0 - qmin)
            dmax = torch.abs(qmax - q0)
            q_close = torch.where(dmax >= dmin, qmax, qmin)
            q_open_list.append(q0)
            q_close_list.append(q_close)
        q_open = torch.stack(q_open_list).unsqueeze(0)
        q_close = torch.stack(q_close_list).unsqueeze(0)
        return q_close + open_frac * (q_open - q_close)

    def _map_inspire6_to_joint_targets(self, inspire6: torch.Tensor) -> torch.Tensor:
        # hold & deadzone, enforce thumb if requested
        effective = self._last_inspire6.clone()
        for i in range(6):
            new_vals = inspire6[:, i]
            hold_mask = new_vals < 0
            delta = torch.abs(new_vals - self._last_inspire6[:, i])
            small_mask = delta < 20.0
            update_mask = (~hold_mask) & (~small_mask)
            effective[:, i] = torch.where(update_mask, new_vals, effective[:, i])
        if args_cli.enforce_thumb_400:
            effective[:, 4] = 400.0
        self._last_inspire6 = effective

        # channels: 0=little,1=ring,2=middle,3=index,4=thumb,5=extra(ignored)
        finger_to_joints = {
            0: ["right_little_1_joint", "right_little_2_joint"],
            1: ["right_ring_1_joint", "right_ring_2_joint"],
            2: ["right_middle_1_joint", "right_middle_2_joint"],
            3: ["right_index_1_joint", "right_index_2_joint"],
        }
        full_targets = self.hand.data.joint_pos.clone()
        for f_idx in range(4):
            val = torch.clamp(effective[:, f_idx], 0.0, 1000.0)
            open_frac = val / 1000.0  # 0=closed, 1=open
            jnames = finger_to_joints[f_idx]
            # Use same open fraction for both joints; closing direction auto-selected in conversion
            pair = torch.stack([open_frac, open_frac], dim=1)
            rad = self._convert_open_fraction_to_radians(pair, jnames)
            for j_idx, jn in enumerate(jnames):
                j = self.name_to_index.get(jn, None)
                if j is not None:
                    full_targets[:, j] = rad[:, j_idx]
        # thumb: keep at open default or map if not enforcing
        if not args_cli.enforce_thumb_400:
            # map thumb 0..1000
            thumb_val = torch.clamp(effective[:, 4], 0.0, 1000.0)
            open_frac = thumb_val / 1000.0
            thumb_names = ["right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"]
            # use same open_frac for all thumb joints
            mat = torch.stack([open_frac, open_frac, open_frac, open_frac], dim=1)
            rad = self._convert_open_fraction_to_radians(mat, thumb_names)
            for j_idx, jn in enumerate(thumb_names):
                j = self.name_to_index.get(jn, None)
                if j is not None:
                    full_targets[:, j] = rad[:, j_idx]
        else:
            for jn in ["right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"]:
                j = self.name_to_index.get(jn, None)
                if j is not None:
                    full_targets[:, j] = self.joint_default[j]
        return full_targets


def main():
    cfg = EnvCfg()

    # Generate all 1061 pad sensor names based on the distribution provided
    sensor_names = []
    
    # Palm: 112 pads (14×8)
    for i in range(1, 113):  # 001-112
        sensor_names.append(f"palm_pad_{i:03d}")
    
    # Thumb1: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"thumb1_pad_{i:03d}")
    
    # Thumb2: 8 pads (2×4)
    for i in range(1, 9):  # 001-008
        sensor_names.append(f"thumb2_pad_{i:03d}")
    
    # Thumb3: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"thumb3_pad_{i:03d}")
    
    # Thumb4: 9 pads (3×3)
    for i in range(1, 10):  # 001-009
        sensor_names.append(f"thumb4_pad_{i:03d}")
    
    # Index1: 80 pads (8×10)
    for i in range(1, 81):  # 001-080
        sensor_names.append(f"index1_pad_{i:03d}")
    
    # Index2: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"index2_pad_{i:03d}")
    
    # Index3: 9 pads (3×3)
    for i in range(1, 10):  # 001-009
        sensor_names.append(f"index3_pad_{i:03d}")
    
    # Middle1: 80 pads (10×8)
    for i in range(1, 81):  # 001-080
        sensor_names.append(f"middle1_pad_{i:03d}")
    
    # Middle2: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"middle2_pad_{i:03d}")
    
    # Middle3: 9 pads (3×3)
    for i in range(1, 10):  # 001-009
        sensor_names.append(f"middle3_pad_{i:03d}")
    
    # Ring1: 80 pads (8×10)
    for i in range(1, 81):  # 001-080
        sensor_names.append(f"ring1_pad_{i:03d}")
    
    # Ring2: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"ring2_pad_{i:03d}")
    
    # Ring3: 9 pads (3×3)
    for i in range(1, 10):  # 001-009
        sensor_names.append(f"ring3_pad_{i:03d}")
    
    # Little1: 80 pads (8×10)
    for i in range(1, 81):  # 001-080
        sensor_names.append(f"little1_pad_{i:03d}")
    
    # Little2: 96 pads (8×12)
    for i in range(1, 97):  # 001-096
        sensor_names.append(f"little2_pad_{i:03d}")
    
    # Little3: 9 pads (3×3)
    for i in range(1, 10):  # 001-009
        sensor_names.append(f"little3_pad_{i:03d}")

    print(f"[INFO] Adding {len(sensor_names)} contact sensors for all pads...")
    
    # Add contact sensor configuration for all pads
    added_count = 0
    for sensor_name in sensor_names:
        sensor_cfg = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,  # Set to False to avoid visual clutter with 1061 sensors
            track_pose=True,
            force_threshold=0.001,
        )
        setattr(cfg.scene, f"contact_{sensor_name}", sensor_cfg)
        added_count += 1
    
    print(f"[INFO] Successfully added {added_count} contact sensors to scene configuration")
    print(f"[INFO] Total sensors configured: {len(sensor_names)} (expected: 1061)")

    # Pass sensor names to cfg so environment can access them
    cfg.selected_sensor_names = sensor_names

    env = InspireHandMediapipeEnv(cfg=cfg, render_mode="rgb_array")
    try:
        env.reset()
        total_steps = int(args_cli.total_time / (cfg.sim.dt * cfg.decimation))
        for step in range(total_steps):
            actions = torch.zeros(env.num_envs, env.hand.num_joints, device=env.device)
            env.step(actions)
            if step % 600 == 0:
                print(f"[INFO] Progress: {100.0*step/total_steps:.1f}% ({step}/{total_steps})")
                # Log contact forces every 5 seconds
                if hasattr(env, '_log_contact_forces'):
                    env._log_contact_forces(step)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close() 