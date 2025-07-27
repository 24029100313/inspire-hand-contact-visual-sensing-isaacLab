#!/usr/bin/env python3
"""
Simplified Contact Sensor Experiment for Isaac Lab

This is a simplified version that demonstrates the core concepts
of contact sensor usage in Isaac Lab.
"""

import torch
import numpy as np
from typing import Dict, List

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass


@configclass
class ContactSensorExpCfg(DirectRLEnvCfg):
    """Configuration for the contact sensor experiment."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,  # 60 Hz simulation
        render_interval=1,
        disable_contact_processing=False,
    )

    # Environment settings
    episode_length_s = 5.0  # 5 seconds episode
    decimation = 1
    num_actions = 0
    num_observations = 12  # 4 sensors × 3 values each (just force magnitude)
    num_states = 0

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
    )

    # 1kg Cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),  # 2m above ground
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.2),  # 1m x 1m x 0.2m
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),  # Red
            ),
        ),
    )

    # Contact sensors in square arrangement
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ContactSensor1",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        force_threshold=0.1,
        track_pose=False,
        track_air_time=False,
    )

    contact_sensor_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ContactSensor2",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        force_threshold=0.1,
        track_pose=False,
        track_air_time=False,
    )

    contact_sensor_3: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ContactSensor3",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        force_threshold=0.1,
        track_pose=False,
        track_air_time=False,
    )

    contact_sensor_4: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ContactSensor4",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
        force_threshold=0.1,
        track_pose=False,
        track_air_time=False,
    )


class ContactSensorExperiment(DirectRLEnv):
    """Simplified contact sensor experiment."""

    cfg: ContactSensorExpCfg

    def __init__(self, cfg: ContactSensorExpCfg, render_mode: str = "rgb_array", **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Sensor positions (square arrangement)
        self.sensor_positions = [
            [-0.6, -0.6, 0.0],  # Sensor 1
            [0.6, -0.6, 0.0],   # Sensor 2
            [0.6, 0.6, 0.0],    # Sensor 3
            [-0.6, 0.6, 0.0],   # Sensor 4
        ]
        
        self.expected_gravity = 9.81  # 1kg × 9.81m/s²
        self.step_count = 0
        
        print("🔬 Contact Sensor Experiment initialized")
        print(f"🎯 Expected gravity force: {self.expected_gravity:.2f} N")

    def _setup_scene(self):
        # Ground plane
        self.cfg.terrain.func(
            prim_path="/World/ground",
            cfg=self.cfg.terrain,
        )
        
        # Falling cube
        self.cfg.cube.func(
            prim_path=self.cfg.cube.prim_path,
            cfg=self.cfg.cube,
        )
        
        # Create sensors
        self._create_sensors()
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _create_sensors(self):
        """Create contact sensors."""
        sensors = [
            self.cfg.contact_sensor_1,
            self.cfg.contact_sensor_2,
            self.cfg.contact_sensor_3,
            self.cfg.contact_sensor_4,
        ]
        
        for i, (sensor_cfg, pos) in enumerate(zip(sensors, self.sensor_positions)):
            # Create sensor geometry
            sensor_cfg.spawn = sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.8, 0.1),  # Green
                ),
            )
            
            sensor_cfg.init_state = RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=(1.0, 0.0, 0.0, 0.0),
            )
            
            # Create sensor
            sensor_cfg.func(
                prim_path=sensor_cfg.prim_path,
                cfg=sensor_cfg,
            )

    def _get_observations(self) -> torch.Tensor:
        """Get observations from sensors."""
        observations = []
        
        # Get data from all sensors
        for i in range(1, 5):
            sensor_name = f"contact_sensor_{i}"
            if sensor_name in self.scene.sensors:
                sensor = self.scene.sensors[sensor_name]
                # Get force magnitude
                force_data = sensor.data.net_forces_w[:, 0, :]  # [num_envs, 3]
                force_magnitude = torch.norm(force_data, dim=1, keepdim=True)  # [num_envs, 1]
                
                # Get Z-component (vertical force)
                force_z = force_data[:, 2:3]  # [num_envs, 1]
                
                # Get contact state
                contact_state = (force_magnitude > 0.1).float()  # [num_envs, 1]
                
                sensor_obs = torch.cat([force_magnitude, force_z, contact_state], dim=1)
                observations.append(sensor_obs)
            else:
                # If sensor not found, add zeros
                observations.append(torch.zeros(self.num_envs, 3, device=self.device))
        
        return torch.cat(observations, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset cube position
        self.scene.rigid_objects["cube"].write_root_state_to_sim(
            self.scene.rigid_objects["cube"].data.default_root_state[env_ids]
        )
        
        self.step_count = 0

    def step(self, action: torch.Tensor):
        """Step the environment."""
        obs, reward, done, info = super().step(action)
        
        self.step_count += 1
        
        # Analyze forces every 60 steps (1 second)
        if self.step_count % 60 == 0:
            self._analyze_forces()
        
        return obs, reward, done, info

    def _analyze_forces(self):
        """Analyze contact forces."""
        total_force_z = 0.0
        sensor_forces = []
        
        for i in range(1, 5):
            sensor_name = f"contact_sensor_{i}"
            if sensor_name in self.scene.sensors:
                sensor = self.scene.sensors[sensor_name]
                force_z = sensor.data.net_forces_w[0, 0, 2].item()
                sensor_forces.append(force_z)
                total_force_z += force_z
        
        # Calculate error
        error = abs(total_force_z - self.expected_gravity)
        error_percent = (error / self.expected_gravity) * 100 if self.expected_gravity > 0 else 0
        
        print(f"\n📊 Force Analysis (Step {self.step_count}):")
        print(f"🔍 Sensor forces (Z): {[f'{f:.3f}' for f in sensor_forces]} N")
        print(f"⚡ Total force: {total_force_z:.3f} N")
        print(f"🎯 Expected: {self.expected_gravity:.3f} N")
        print(f"❌ Error: {error:.3f} N ({error_percent:.1f}%)")
        
        if error_percent < 5.0:
            print("✅ Physics verification: PASSED")
        else:
            print("❌ Physics verification: FAILED")


def main():
    """Run the contact sensor experiment."""
    print("🚀 Starting Contact Sensor Experiment...")
    
    # Create configuration
    cfg = ContactSensorExpCfg()
    
    # Create environment
    env = ContactSensorExperiment(cfg, render_mode="human")
    
    # Reset environment
    env.reset()
    
    # Run experiment
    print("🎯 Running experiment...")
    
    for step in range(300):  # 5 seconds at 60 Hz
        # No actions needed
        action = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Print progress
        if step % 60 == 0:
            progress = (step / 300) * 100
            print(f"⏱️  Progress: {progress:.0f}% ({step}/300 steps)")
    
    print("\n🏁 Experiment Complete!")
    env.close()


if __name__ == "__main__":
    main()
