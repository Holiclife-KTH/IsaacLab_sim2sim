# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.prims import is_prim_path_valid, define_prim, get_prim_at_path
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import RigidPrim
from IsaacLab_sim2sim.project_sweeping.asset.ur5e_2f85 import UR5e_2f85


class FollowTarget(tasks.FollowTarget):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "franka_follow_target".
        target_prim_path (Optional[str], optional): [description]. Defaults to None.
        target_name (Optional[str], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        franka_prim_path (Optional[str], optional): [description]. Defaults to None.
        franka_robot_name (Optional[str], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        ur5e_prim_path: Optional[str] = None,
        ur5e_usd_path : Optional[str] = None,
        ur5e_robot_name: Optional[str] = None,  
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        self._ur5e_prim_path = ur5e_prim_path
        self._ur5e_usd_path = ur5e_usd_path
        self._ur5e_robot_name = ur5e_robot_name
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
                         position=[0.0, 0.0, 0.0],
                         orientation=[0.0, 0.0, 0.0, 1.0],
                         arm_dof_names=["shoulder_pan_joint",
                                        "shoulder_lift_joint",
                                        "elbow_joint",
                                        "wrist_1_joint",
                                        "wrist_2_joint",
                                        "wrist_3_joint"],
                         gripper_dof_names=["finger_joint", "right_outer_knuckle_joint"],)
