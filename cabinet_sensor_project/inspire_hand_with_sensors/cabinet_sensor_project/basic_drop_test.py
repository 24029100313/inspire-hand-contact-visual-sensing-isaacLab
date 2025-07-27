#!/usr/bin/env python3

"""
Basic cube drop test - following Isaac Lab official pattern exactly.
Simple environment with ground plane and falling cube.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Basic cube drop test following official pattern")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene - exactly like official example."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create single origin (keeping it simple)
    origins = [[0.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Falling Cube - following exact official pattern but using cube instead of cone
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),  # 50cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),  # Red cube
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cube_object = RigidObject(cfg=cube_cfg)

    # return the scene information
    scene_entities = {"cube": cube_object}
    return scene_entities, origins


def run_simulator(sim: SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop - following official pattern exactly."""
    # Extract scene entities
    cube_object = entities["cube"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print("🎯 Basic Drop Test Started")
    print("📦 Red cube should fall from 2m height and settle on ground")
    print("="*60)
    
    # Simulate physics
    while simulation_app.is_running():
        # reset every 200 steps (shorter cycle for testing)
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = cube_object.data.default_root_state.clone()
            
            # Set cube position: 2m height above origin
            root_state[:, :3] += origins
            root_state[:, 2] += 2.0  # 2m height
            
            # write root state to simulation
            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            cube_object.reset()
            print("🔄 Reset: Cube positioned at 2m height")
            
        # apply sim data
        cube_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube_object.update(sim_dt)
        
        # Print status every 20 steps (about 0.1 seconds)
        if count % 20 == 0:
            cube_pos = cube_object.data.root_pos_w[0]
            cube_vel = cube_object.data.root_lin_vel_w[0]
            height = cube_pos[2].item()
            speed = torch.norm(cube_vel).item()
            print(f"Step {count:3d}: h={height:.2f}m, v={speed:.2f}m/s, t={sim_time:.1f}s")


def main():
    """Main function - exactly like official example."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera - close view for better observation
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Basic drop test setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 