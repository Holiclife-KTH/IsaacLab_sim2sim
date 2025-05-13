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



parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

# Nucleus root path
assets_root_path = "omniverse://localhost/Library/Shelf"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

prim_paths = {
        "env": "/World/env",
        "object": "/World/Object",
        "robot": "/World/Robot",
        "front_camera": "/World/FrontCamera"
    }

SEGMENTATION_PATH = "/home/irol/.local/share/ov/pkg/isaac-sim-4.2.0/KTH/project_sweeping/model/best_hg.pt"

simulation_app = SimulationApp(CONFIG)

import omni.usd as usd

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.prims import create_prim
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, Usd, UsdLux

from KTH.project_sweeping.asset.ur5e_2f85 import UR5e_2f85
from KTH.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world, load_yaml_config, load_and_reshape_pose
from KTH.project_sweeping.src.robot_controller import ArmController

from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.core.utils import extensions,  prims, stage, viewports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles

from omni.isaac.core.utils.types import ArticulationAction




class Sim2Sim:
    def __init__(self, yaml_path):
        self.scene_entities = {}
        self.world = World(stage_units_in_meters = 1.0)
        
        object_cfgs = load_yaml_config(yaml_path=yaml_path)

        self.object_path_dict = object_cfgs["objects"]
        self.object_pose_dict = object_cfgs["pose"]
        self.object_id_dict = object_cfgs["id"]

        self.pose_array = load_and_reshape_pose(self.object_pose_dict)
        self.object_dict = {}

        self.seg_dict = {}

        # Joint_configuration
        self._joint_config_dict = {"pre_segmentation_pose": np.array([-np.pi/2.0, -2.2, 2.2, 0.0, 1.57, 1.57]),
                                   "segmentation_pose" : np.array([-np.pi/2.0, -np.pi / 18.0 , -np.pi * (7.0 / 18.0), -np.pi, -np.pi/ 2.0, 1.57]),
                                   "home_pose": np.array([0.0, -2.2, 2.2, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        
        self.robot_controller = ArmController(default_state=self._joint_config_dict["home_pose"],control_gain=0.5)


        # Load YOLO model weight
        self._model = YOLO(SEGMENTATION_PATH, verbose=False)
        self._model.eval()

        self.initialize_world()
        
    
    def initialize_world(self):
        stage_utils.add_reference_to_stage(usd_path="omniverse://localhost/Library/Shelf/testbench.usd", prim_path=prim_paths["env"])

        self.robot: UR5e_2f85 = self.world.scene.add(UR5e_2f85(prim_path=prim_paths["robot"],
                            usd_path="omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd",
                            position=[0.0, 0.0, 0.79505],
                            orientation=[0.0, 0.0, 0.0, 1.0],
                            arm_dof_names=["shoulder_pan_joint",
                                           "shoulder_lift_joint",
                                           "elbow_joint",
                                           "wrist_1_joint",
                                           "wrist_2_joint",
                                           "wrist_3_joint"],
                            gripper_dof_names=["finger_joint", "right_outer_knuckle_joint"],))

        for key in self.object_path_dict:
            stage_utils.add_reference_to_stage(usd_path=self.object_path_dict[key], prim_path=os.path.join(prim_paths["object"], f"{key}"))
            self.object_dict[key] = self.world.scene.add(RigidPrim(prim_path=os.path.join(prim_paths["object"], f"{key}"),
                                                                   name=f"{key}",
                                                                   mass=0.5,
                                                                   position=self.object_pose_dict[key][:3],
                                                                   orientation=self.object_pose_dict[key][3:]))

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
        

        create_prim("/DistantLight", "DistantLight", attributes={"inputs:intensity": 500})
        create_prim("/DomeLight1", "DomeLight", translation=(1.2, 0.0, 1.4), attributes={"inputs:intensity": 2700})


    def reset_world(self):
        self.robot.set_default_state(position=[0.0, 0.0, 0.79505], orientation=[0.0, 0.0, 0.0, 1.0])
        self.robot.set_joints_default_state(positions=self._joint_config_dict["home_pose"])
        for key in self.object_dict:
            self.object_dict[key].set_default_state(position=self.object_pose_dict[key][:3],
                                                    orientation=self.object_pose_dict[key][3:])

        self.world.reset()


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


        #     # >>> STEP 4. 바운딩 박스 그리기
        #     label = f"{cls}, {conf:.2f}"
        #     cv2.rectangle(
        #         img,
        #         (int(x1), int(y1)),
        #         (int(x2), int(y2)),
        #         color=(255, 0, 0),
        #         thickness=2,
        #     )
        #     cv2.putText(
        #         img=img,
        #         text=label,
        #         org=(int(x1), int(y1 - 10)),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=0.5,
        #         color=(0, 0, 0),
        #         thickness=2,
        #     )
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgplot = plt.imshow(img)
        # plt.show()

        return seg_dict



    def get_observation(self) -> np.array:
        front_image = self._front_camera.get_rgb()
        front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)
        self.seg_dict = self.get_segmentation_data(front_image)

        print(self.seg_dict)


    def move_init_observate_pose(self) -> None:
        current_joint_state = self.robot.get_joint_positions()
        delta = self._joint_config_dict["pre_segmentation_pose"] - current_joint_state[:6]
        while np.linalg.norm(delta, ord=1) > 0.2:
            current_joint_state = self.robot.get_joint_positions()
            delta = self._joint_config_dict["pre_segmentation_pose"] - current_joint_state[:6]
            self.robot._arm.apply_action(ArticulationAction(joint_positions=(current_joint_state[:6] + delta / 30.0)))
            self.world.step(render=True)
        
        current_joint_state = self.robot.get_joint_positions()
        delta = self._joint_config_dict["segmentation_pose"] - current_joint_state[:6]
        while np.linalg.norm(delta, ord=1) > 0.2:
            current_joint_state = self.robot.get_joint_positions()
            delta = self._joint_config_dict["segmentation_pose"] - current_joint_state[:6]
            self.robot._arm.apply_action(ArticulationAction(joint_positions=(current_joint_state[:6] + delta / 30.0)))
            self.world.step(render=True)

        
            



    def run_simulation(self, steps_per_iteration: int = 500):
        """
        
        """
        while simulation_app.is_running():
            self.reset_world()
            self.move_init_observate_pose()


            for step in range(steps_per_iteration):
                if (step != 0) & ((step % 100 )== 0):
                    
                    self.get_observation()

                self.world.step(render=True)
                



def main():
    simulation_app.update()

    viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0.0, 0.0, 0.5]))

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "config", "data_cfg.yaml")

    world = Sim2Sim(yaml_path=yaml_path)
    simulation_app.update()


    world.run_simulation()
    simulation_app.close()


if __name__=="__main__":
    main()