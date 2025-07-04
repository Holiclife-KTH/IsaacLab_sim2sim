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

from IsaacLab_sim2sim.project_sweeping.asset.ur5e_2f85 import UR5e_2f85
from IsaacLab_sim2sim.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world, load_yaml_config, load_and_reshape_pose
from IsaacLab_sim2sim.project_sweeping.src.robot_controller import ArmController
from IsaacLab_sim2sim.project_sweeping.src.sweeping_policy import ShelfSearchingPolicy

from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.core.utils import extensions,  prims, stage, viewports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles

from omni.isaac.core.utils.types import ArticulationAction


class State(Enum):
    FIRST_STATE = 0
    RESET_STATE = 1
    OBSERVATION_MOTION_STATE = 2
    SEGMENTATION_STATE = 3
    DEFAULT_POSE_STATE = 4
    POLICY_STATE = 5

class Sim2Sim(object):
    def __init__(self, local_dir, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Spawn testbench
        testbench_path = "omniverse://localhost/Library/Shelf/testbench.usd"
        stage_utils.add_reference_to_stage(usd_path=testbench_path, prim_path=prim_paths["env"])

        sweeping_policy_path = os.path.join(local_dir, "model/sweeping/2025-05-20_17-16-16/exported", "policy.pt")
        sweeping_policy_params_path = os.path.join(local_dir, "model/sweeping/2025-05-20_17-16-16/params", "env.yaml")

        grasping_policy_path = os.path.join(local_dir, "model/grasping/2025-05-18_18-12-33/exported", "policy.pt")
        grasping_policy_params_path = os.path.join(local_dir, "model/grasping/2025-05-18_18-12-33/params", "env.yaml")
        usd_path = "omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd"
        yaml_path = os.path.join(local_dir, "config", "data_cfg.yaml")

        #Spawn objects
        object_cfgs = load_yaml_config(yaml_path=yaml_path)

        self.object_path_dict = object_cfgs["objects"]
        self.object_pose_dict = object_cfgs["pose"]
        self.object_id_dict = object_cfgs["id"]
        self.object_id_dict_rev = {str(v): k for k, v in self.object_id_dict.items()}
        self.object_config = np.zeros((2, 3))
        self.object_config.fill(-1)

        self._x_bins = [100, 225, 350, 475]       # 구간은 (100,225], (225,350], (350,475]
        self._depth_bins = [0.95, 1.1, 1.25]     # 구간은 (0.95,1.05], (1.05,1.15]

        self.pose_array = load_and_reshape_pose(self.object_pose_dict)
        self.object_dict = {}
        self.seg_dict = {}
        self._moved_object = []


        for key in self.object_path_dict:
            stage_utils.add_reference_to_stage(usd_path=self.object_path_dict[key], prim_path=os.path.join("/World/Object", f"{key}"))
            self.object_dict[key] = RigidPrim(prim_path=os.path.join("/World/Object", f"{key}"),
                                                                   name=f"{key}",
                                                                   scale=(1.0, 1.0, 1.0),
                                                                   mass=0.5,
                                                                   position=self.object_pose_dict[key][:3],
                                                                   orientation=self.object_pose_dict[key][3:])
            self._world.scene.add(self.object_dict[key])

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

        # Joint_configuration
        self._joint_config_dict = {"pre_segmentation_pose": np.array([-np.pi/2.0, -2.2, 2.2, 0.0, 1.57, 1.57]),
                                   "segmentation_pose" : np.array([-np.pi/2.0, -np.pi / 18.0 , -np.pi * (7.0 / 18.0), -np.pi, -np.pi/ 2.0, 1.57]),
                                   "home_pose": np.array([0.0, -2.2, 2.2, 0.0, 1.57, 0.758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}

        self._agent = ShelfSearchingPolicy(
            prim_path="/World/Robot",
            name = "UR5e",
            usd_path=usd_path,
            sweeping_policy_path=sweeping_policy_path,
            sweeping_policy_params_path= sweeping_policy_params_path,
            grasping_policy_path=grasping_policy_path,
            grasping_policy_params_path=grasping_policy_params_path,
            position=[0.0, 0.0, 0.79505],
            orientation=[0.0, 0.0, 0.0, 1.0],
            object_dict=self.object_dict)
        
        self._world.scene.add(self._agent.robot) 

        # Load YOLO model weight
        self._model = YOLO(SEGMENTATION_PATH, verbose=False)
        self._model.eval()

        self._count = 0
        self._sub_target = None

        self._policy = ["grasping", "sweeping"]

        self._state = State.FIRST_STATE
        self._methods = {State.FIRST_STATE: self.first_func, 
                         State.RESET_STATE: self.reset_func,
                         State.OBSERVATION_MOTION_STATE: self.obs_motion_func,
                         State.SEGMENTATION_STATE: self.segment_func,
                         State.DEFAULT_POSE_STATE: self.default_pose_func,
                         State.POLICY_STATE: self.policy_func }

    def reset_func(self):
        carb.log_info("NEW EPISODE")
        self._agent._policy_counter = 0
        self._agent._action = np.zeros(7)
        self._agent._previous_action = np.zeros(7)
        self._agent.robot.set_default_state(position=[0.0, 0.0, 0.79505], orientation=[0.0, 0.0, 0.0, 1.0])
        self._agent.robot.set_joints_default_state(positions=self._joint_config_dict["home_pose"])
        self._moved_object = []
        self.object_config = np.zeros((2, 3))
        self.object_config.fill(-1)
        
        self._state = State.FIRST_STATE
        
    def obs_motion_func(self):
        current_joint_state = self._agent.robot.get_joint_positions()

        if np.abs(current_joint_state[0] + np.pi/2.0) > 0.1:
            current_joint_state = self._agent.robot.get_joint_positions()
            self._agent.robot._arm.apply_action(ArticulationAction(joint_positions=self._joint_config_dict["pre_segmentation_pose"]))
        else:
            self._agent.robot._arm.apply_action(ArticulationAction(joint_positions=self._joint_config_dict["segmentation_pose"]))
            self._count += 1
            if self._count > 100:
                self._count = 0
                self._state = State.SEGMENTATION_STATE

    def default_pose_func(self):
        current_joint_state = self._agent.robot.get_joint_positions()
        if np.abs(current_joint_state[1] + 2.2) > 0.1:
            current_joint_state = self._agent.robot.get_joint_positions()
            # self._agent.robot._arm.apply_action(ArticulationAction(joint_velocities=delta/0.005))
            self._agent.robot._arm.apply_action(ArticulationAction(joint_positions=self._joint_config_dict["pre_segmentation_pose"]))

        else:
            self._agent.robot._arm.apply_action(ArticulationAction(joint_positions=self._joint_config_dict["home_pose"][:6]))
            self._count += 1
            if self._count > 100:
                self._count = 0
                self._state = State.POLICY_STATE



    def segment_func(self):
        front_image = self._front_camera.get_rgb()
        front_depth_img = self._front_camera.get_depth()
        front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        self.seg_dict = self.get_segmentation_data(front_image)

        if TARGET in self.seg_dict.keys():
            self._sub_target = TARGET
            self._state = State.DEFAULT_POSE_STATE

        else:
            for key in self.seg_dict.keys():
                points = self.seg_dict[key]
                x = int((points[0] + points[2])/2)
                y = int((points[1] + points[3])/2)


                depth = front_depth_img[y, x]

                col = np.digitize(x, self._x_bins) - 1  # 결과는 0, 1, 2
                row = np.digitize(depth, self._depth_bins) - 1  # 결과는 0, 1

                if 0 <= col < 3 and 0 <= row < 2:
                    self.object_config[row, col] = self.object_id_dict[key]

            # 1. 우측 → 좌측 방향으로 열 순서 뒤집기
            reversed_cols = self.object_config[:, ::-1]

            # 2. 행 순서 유지한 채 1차원으로 평탄화 (row-major order)
            flattened = reversed_cols.flatten().astype(np.int8)

            # 3. -1 제외
            filtered = flattened[flattened != -1]

            # 결과
            values = filtered.tolist()

            print(values)
            # 3. 움직인 물체 제거
            filtered_values = [v for v in values if v not in self._moved_object]

            if len(filtered_values) > 0: 
                target_id = filtered_values[0]
                self._sub_target = self.object_id_dict_rev[str(target_id)]
                print(self._sub_target)
                self._moved_object.append(target_id)
                self._state = State.DEFAULT_POSE_STATE
            else:
                carb.log_error("target object is not selected!!")

    def first_func(self):
        self._agent.initialize()
        for key in self.object_path_dict:
            self.object_dict[key].initialize()
        self.reset_func()
        self._world.reset(True)

        self._state = State.OBSERVATION_MOTION_STATE
        # self.seg_step = True


    def policy_func(self):
        
        if self._sub_target == TARGET:
            self._agent.forward(mode=self._policy[0], target=TARGET)
            done = self._agent._check_done(mode=self._policy[0], target=TARGET)

            if done:
                self._state = State.RESET_STATE

        elif self._sub_target != TARGET:
            self._agent.forward(mode=self._policy[1], target=self._sub_target)
            done = self._agent._check_done(mode=self._policy[1], target=self._sub_target)
            
            if done:
                self._state = State.OBSERVATION_MOTION_STATE
            
        
    def setup(self) -> None:
        self._world.add_physics_callback(callback_name="sweeping", callback_fn=self.on_physics_step)

    def get_segmentation_data(self, img) -> dict:
        seg_dict = {}
        seg_result: Results = self._model(img)[0]
        boxes: Boxes = seg_result.boxes
        classes: dict = seg_result.names
        masks: Masks = seg_result.masks

        np_boxes = boxes.xyxy.cpu().numpy()
        np_conf = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()


        for idx in range(len(boxes)):
            id = int(np_cls[idx])
            cls = classes[id]
            conf = np_conf[idx]

            x1, y1, x2, y2 = map(int, np_boxes[idx])

            # >>> STEP 1. 신뢰도 확인
            if conf < 0.7 or abs(x2 - x1) < 30:
                continue
            
            seg_dict[cls] = (x1, y1, x2, y2)

        return seg_dict


    def get_seg_observation(self) -> np.array:
        front_image = self._front_camera.get_rgb()
        front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        self.seg_dict = self.get_segmentation_data(front_image)



    def on_physics_step(self, step_size) -> None:
        self._methods[self._state]()



    def run_simulation(self, steps_per_iteration: int = 500) -> None:
        while simulation_app.is_running():
            if not self._world.physics_callback_exists(callback_name="sweeping"):
                self.setup()
            self._world.step(render=True)
            if self._agent._policy_counter == steps_per_iteration:
                self._state = State.RESET_STATE

            if self._state == State.RESET_STATE:
                self._world.remove_physics_callback(callback_name="sweeping")
                self.reset_func()
                self._world.reset()
            
        return



def main():
    simulation_app.update()
    viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0.0, 0.0, 0.5]))

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    world = Sim2Sim(local_dir=current_dir, physics_dt=1.0/60.0, render_dt=1.0/60.0)
    simulation_app.update()

    world.reset_func()
    world._world.reset()
    simulation_app.update()

    world.setup()
    simulation_app.update()

    world.run_simulation()
    simulation_app.close()


if __name__=="__main__":
    main()