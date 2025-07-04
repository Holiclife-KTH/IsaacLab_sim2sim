from isaacsim import SimulationApp

import argparse
import os
import sys
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import cv2
import time
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks, Results
import onnx
import onnxruntime
import carb
from enum import Enum




parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--task_target", type=str, default="cup_1", help="Set target object of task")

config = parser.parse_args() 

# Nucleus root path
assets_root_path = "omniverse://localhost/Library/Shelf"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
TARGET = config.task_target

prim_paths = {
        "env": "/World/env",
        "object": "/World/Object",
        "robot": "/World/Robot",
        "front_camera": "/World/FrontCamera"
    }

SEGMENTATION_PATH = "/home/irol/Downloads/isaacsim/IsaacLab_sim2sim/project_sweeping/model/best_hg.pt"
SWEEPING_POLICY_PATH = "/home/irol/Downloads/isaacsim/IsaacLab_sim2sim/project_sweeping/model/sweeping/2025-05-11_00-35-08/exported/policy.onnx"

simulation_app = SimulationApp(CONFIG)

import omni.usd as usd
import omni.appwindow

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.prims import create_prim
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, Usd, UsdLux
from omni.isaac.core.utils.prims import define_prim

from IsaacLab_sim2sim.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world, load_yaml_config, load_and_reshape_pose
from IsaacLab_sim2sim.project_sweeping.src.shelf_task import ShelfTask
from IsaacLab_sim2sim.controller.rmpflow import UR5eRMPFlowController

from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.core.utils import extensions,  prims, stage, viewports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles

from omni.isaac.core.utils.types import ArticulationAction

class Sim2Sim(object):
    def __init__(self, local_dir, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self._task = ShelfTask(name="shelf_searching",
                               local_dir=local_dir,
                               target_name="mug_1",
                               ur5e_prim_path="/World/Robot",
                               ur5e_usd_path="omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd",
                               ur5e_robot_name="ur5e")
        
        self._world.add_task(self._task)

        #Front Camera
        self._front_camera: Camera = Camera(prim_path=prim_paths["front_camera"],
                                            position=np.array([0.48, 0.0, 1.27]),
                                            frequency=30,
                                            resolution=(640, 480),
                                            orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 180.0]), degrees=True),)
        

        self._front_camera.initialize()
        self._front_camera.set_focal_length(24.0)
        self._front_camera.set_focus_distance(400.0)
        self._front_camera.set_horizontal_aperture(20.955)
        self._front_camera.set_clipping_range(0.1, 1.0e5)
        self._front_camera.add_distance_to_image_plane_to_frame()

        create_prim("/DistantLight", "DistantLight", attributes={"inputs:intensity": 500})
        create_prim("/DomeLight1", "DomeLight", translation=(1.2, 0.0, 1.4), attributes={"inputs:intensity": 2700})

        self._world.reset()

        self._robot = self._world.scene.get_object("ur5e_2f85")

        self._controller = UR5eRMPFlowController(name="shelf_searching_controller", robot_articulation=self._robot)
        self._articulation_controller = self._robot.get_articulation_controller()

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

    def run_simulation(self, steps_per_iteration: int = 500) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)

            if self._world.is_playing():
                observations = self._world.get_observations()
                actions = self._controller.forward(target_end_effector_position=observations["target"]["position"] + (0.0, -0.06, 0.09),
                                                   target_end_effector_orientation=np.array([-0.7071, 0.0, 0.7071, 0.0]))
                
                # print(actions)
                self._articulation_controller.apply_action(actions)
        return

        

def main():
    simulation_app.update()
    viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0.0, 0.0, 0.5]))

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    world = Sim2Sim(local_dir=current_dir, physics_dt=1.0/60.0, render_dt=1.0/60.0)
    simulation_app.update()

    world.run_simulation()
    simulation_app.close()



if __name__=="__main__":
    main()