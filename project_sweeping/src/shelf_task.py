# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional
import argparse
import os
import sys
import numpy as np
import yaml
import time
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks, Results
import onnx
import onnxruntime
import carb
from enum import Enum
import numpy as np

from IsaacLab_sim2sim.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world, load_yaml_config, load_and_reshape_pose
from IsaacLab_sim2sim.project_sweeping.asset.ur5e_2f85 import UR5e_2f85

from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid, define_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import RigidPrim



class ShelfTask(ABC, BaseTask):
    """[summary]

    Args:
        name (str): [description]
        target_prim_path (Optional[str], optional): [description]. Defaults to None.
        target_name (Optional[str], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        local_dir: str = None,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        ur5e_prim_path: Optional[str] = None,
        ur5e_usd_path : Optional[str] = None,
        ur5e_robot_name: Optional[str] = None,    
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_name = target_name
        self._target = None
        self._target_prim_path = target_prim_path
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._target_visual_material = None
        self._obstacle_cubes = OrderedDict()

        self._ur5e_prim_path = ur5e_prim_path
        self._ur5e_usd_path = ur5e_usd_path
        self._ur5e_robot_name = ur5e_robot_name
        if self._target_position is None:
            self._target_position = np.array([0, 0.1, 0.7]) / get_stage_units()

        yaml_path = os.path.join(local_dir, "..","config", "data_cfg.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"{yaml_path} does not exist")
        
        #Spawn objects
        object_cfgs = load_yaml_config(yaml_path=yaml_path)

        self.object_path_dict = object_cfgs["objects"]
        self.object_pose_dict = object_cfgs["pose"]
        self.object_id_dict = object_cfgs["id"]
        self.object_id_dict_rev = {str(v): k for k, v in self.object_id_dict.items()}
        self.object_dict = {}
        
        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        # scene.add_default_ground_plane()

        testbench_path = "omniverse://localhost/Library/Shelf/testbench.usd"
        stage_utils.add_reference_to_stage(usd_path=testbench_path, prim_path="/World/env")
        
        for key in self.object_path_dict:
            stage_utils.add_reference_to_stage(usd_path=self.object_path_dict[key], prim_path=os.path.join("/World/Object", f"{key}"))
            self.object_dict[key] = RigidPrim(prim_path=os.path.join("/World/Object", f"{key}"),
                                                                   name=f"{key}",
                                                                   scale=(1.0, 1.0, 1.0),
                                                                   mass=0.5,
                                                                   position=self.object_pose_dict[key][:3],
                                                                   orientation=self.object_pose_dict[key][3:])
            scene.add(self.object_dict[key])
            self._task_objects[key] = self.object_dict[key]

        self._robot = self.set_robot()
        self._robot.set_joints_default_state(np.array([0.0, -2.2, 2.2, 0.0, 1.57, 0.758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return

    def set_robot(self):
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        prim = get_prim_at_path(self._ur5e_prim_path)

        if not prim.IsValid():
            prim = define_prim(self._ur5e_prim_path, "Xform")
            if self._ur5e_usd_path:
                stage_utils.add_reference_to_stage(self._ur5e_usd_path, self._ur5e_prim_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        return UR5e_2f85(prim_path=self._ur5e_prim_path, 
                         position=[0.0, 0.0, 0.79505],
                         orientation=[0.0, 0.0, 0.0, 1.0],
                         arm_dof_names=["shoulder_pan_joint",
                                        "shoulder_lift_joint",
                                        "elbow_joint",
                                        "wrist_1_joint",
                                        "wrist_2_joint",
                                        "wrist_3_joint"],
                         gripper_dof_names=["finger_joint", "right_outer_knuckle_joint"],)

    # def set_target(self) -> None:
    #     self._target = 

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["target_prim_path"] = {"value": self._target.prim_path, "modifiable": True}
        params_representation["target_name"] = {"value": self._target.name, "modifiable": True}
        position, orientation = self._target.get_local_pose()
        params_representation["target_position"] = {"value": position, "modifiable": True}
        params_representation["target_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        target_position, target_orientation = self._task_objects[self._target_name].get_local_pose()
        return {
            "robot": {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            },
            "target": {"position": np.array(target_position), "orientation": np.array(target_orientation)},
        }

    def calculate_metrics(self) -> dict:
        """[summary]"""
        raise NotImplementedError

    def is_done(self) -> bool:
        """[summary]"""
        raise NotImplementedError

    def target_reached(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        target_position, _ = self._target.get_world_pose()
        if np.mean(np.abs(np.array(end_effector_position) - np.array(target_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        if self._target_visual_material is not None:
            if hasattr(self._target_visual_material, "set_color"):
                if self.target_reached():
                    self._target_visual_material.set_color(color=np.array([0, 1.0, 0]))
                else:
                    self._target_visual_material.set_color(color=np.array([1.0, 0, 0]))

        return

    def post_reset(self) -> None:
        """[summary]"""
        return

    def add_obstacle(self, position: np.ndarray = None):
        """[summary]

        Args:
            position (np.ndarray, optional): [description]. Defaults to np.array([0.1, 0.1, 1.0]).
        """
        # TODO: move to task frame if there is one
        cube_prim_path = find_unique_string_name(
            initial_name="/World/ObstacleCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        if position is None:
            position = np.array([0.1, 0.1, 1.0]) / get_stage_units()
        cube = self.scene.add(
            DynamicCuboid(
                name=cube_name,
                position=position + self._offset,
                prim_path=cube_prim_path,
                size=0.1 / get_stage_units(),
                color=np.array([0, 0, 1.0]),
            )
        )
        self._obstacle_cubes[cube.name] = cube
        return cube

    def remove_obstacle(self, name: Optional[str] = None) -> None:
        """[summary]

        Args:
            name (Optional[str], optional): [description]. Defaults to None.
        """
        if name is not None:
            self.scene.remove_object(name)
            del self._obstacle_cubes[name]
        else:
            obstacle_to_delete = list(self._obstacle_cubes.keys())[-1]
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_cubes[obstacle_to_delete]
        return

    def get_obstacle_to_delete(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        obstacle_to_delete = list(self._obstacle_cubes.keys())[-1]
        return self.scene.get_object(obstacle_to_delete)

    def obstacles_exist(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if len(self._obstacle_cubes) > 0:
            return True
        else:
            return False

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacle_cubes.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_cubes[obstacle_to_delete]
        return
