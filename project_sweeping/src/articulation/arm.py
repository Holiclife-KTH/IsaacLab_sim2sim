# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import abstractmethod

import omni.kit.app

from typing import Optional, Sequence

from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.controllers.articulation_controller import ArticulationController

from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper

class Arm(Articulation):
    """Provides high level functions to set/ get properties and actions of a gripper.

    Args:
        end_effector_prim_path (str): prim path of the Prim that corresponds to the gripper root/ end effector.
    """

    def __init__(self, 
                 arm_prim_path: str,
                 end_effector_prim_name: str = None,
                 end_effector_prim_path: str = None,
                 name: str = "Robot_Arm",
                 position: Optional[Sequence[float]] = None,
                 translation: Optional[Sequence[float]] = None,
                 orientation: Optional[Sequence[float]] = None,
                 scale: Optional[Sequence[float]] = None,
                 visible: Optional[bool] = None,
                 gripper: Gripper = None
                 )-> None:
    
        if end_effector_prim_name is None == end_effector_prim_path is None:
            raise Exception(
                "Only one of the following args must be specified: end_effector_prim_name or end_effector_prim_path."
            )
        
        self._end_effector_prim_name = end_effector_prim_name
        self._end_effector_prim_path = end_effector_prim_path
        self._gripper = gripper
        self._end_effector = None
        Articulation.__init__(self, 
                              prim_path=arm_prim_path,
                              name=name,
                              position=position,
                              translation=translation,
                              orientation=orientation,
                              scale=scale,
                              visible=visible,
                              articulation_controller=None)
        return

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None.
        """
        Articulation.initialize(self, physics_sim_view=physics_sim_view)
        return

    @abstractmethod
    def set_default_state(self, *args, **kwargs):
        """Sets the default state of the gripper"""
        raise NotImplementedError

    @abstractmethod
    def get_default_state(self, *args, **kwargs):
        """Gets the default state of the gripper"""
        raise NotImplementedError

