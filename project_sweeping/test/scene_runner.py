from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

import omni.usd as usd

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.prims import create_prim
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, Usd, UsdLux

from KTH.project_sweeping.asset.ur5e_2f85 import UR5e

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
    return parser.parse_args()

# Load configuration file
def load_usd_paths(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

# Initialize world and objects
def initialize_world(usd_paths) -> dict:
    scene_entities = {}
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    assets_root_path = "omniverse://localhost/Library/Shelf"

    prim_paths = {
        "shelf": "/World/Shelf",
        "target": "/World/Object/target",
        "mount": "/World/Mount",
        "robot": "/World/Robot"
    }

    stage_utils.add_reference_to_stage(usd_path=assets_root_path + usd_paths["test_bench"], prim_path=prim_paths["shelf"])
    stage_utils.add_reference_to_stage(usd_path=assets_root_path + usd_paths["target"], prim_path=prim_paths["target"])
    stage_utils.add_reference_to_stage(usd_path=assets_root_path + usd_paths["mount"], prim_path=prim_paths["mount"])
    stage_utils.add_reference_to_stage(usd_path=assets_root_path + usd_paths["robot"], prim_path=prim_paths["robot"])

    objects = {
        "shelf": world.scene.add(RigidPrim(prim_path=prim_paths["shelf"], name="shelf", position=[-0.65, 0.0, 0.0])),
        "target": world.scene.add(RigidPrim(prim_path=prim_paths["target"], name="target", mass=0.3)),
        "mount": world.scene.add(RigidPrim(prim_path=prim_paths["mount"], name="mount", position=[0.0, 0.0, 0.79505])),
        "robot": world.scene.add(Robot(prim_path=prim_paths["robot"], name="robot", position=[-0.1, 0.0, 0.79505]))
    }
    
    create_prim("/DistantLight", "DistantLight", attributes={"inputs:intensity": 500})
    
    camera = Camera(
        prim_path="/World/sensors/cemera",
        position=np.array([0.5, 0.2, 1.2]),
        frequency=30,
        resolution=(640,480),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 180]), degrees=True),
    )
    camera.initialize()
    print(camera.get_focal_length())    
    
    scene_entities["world"] = world
    scene_entities["objects"] = objects
    scene_entities["sensors"] = camera
    
    return scene_entities

# Reset world state
def reset_world(world, objects):
    world.reset()
    objects["shelf"].set_world_pose(position=[-0.65, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])
    objects["target"].set_world_pose(position=[-0.7, 0.0, 0.98], orientation=[1.0, 0.0, 0.0, 0.0])
    objects["mount"].set_world_pose(position=[0.0, 0.0, 0.79505], orientation=[1.0, 0.0, 0.0, 0.0])
    objects["robot"].set_world_pose(position=[-0.1, 0.0, 0.79505], orientation=[0.0, 0.0, 0.0, 1.0])
    # objects["robot"].set_joint_positions(np.array([0.0, -2.0, 2.0, 0.0, 1.57, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# Run simulation loop
def run_simulation(scene_entities:dict, iterations=5, steps_per_iteration=500, test_mode=False):
    while simulation_app.is_running():
        print("Resetting...")
        world = scene_entities["world"]
        objects = scene_entities["objects"]
        camera = scene_entities["sensors"]
        reset_world(world, objects)

        for step in range(steps_per_iteration):
            world.step(render=True)
            if step == 100:
                shelf_position = objects["shelf"].get_current_dynamic_state().position
                target_position = objects["target"].get_current_dynamic_state().position
                print(f"Shelf's position at step {step}: {shelf_position}")
                print(f"Target's position at step {step}: {target_position}")
                imgplot = plt.imshow(camera.get_rgba()[:, :, :3])
                plt.show()

        if test_mode:
            break

# Main execution
def main():
    args = parse_arguments()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "config", "usd.yaml")

    usd_paths = load_usd_paths(yaml_path)
    scene_entities = initialize_world(usd_paths)

    run_simulation(scene_entities, test_mode=args.test)

    simulation_app.close()

if __name__ == "__main__":
    main()
