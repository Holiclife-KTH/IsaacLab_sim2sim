# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Callable, List

import numpy as np
import omni.kit.app
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path


class UR5e(Articulation):
    """Provides high level functions to set/ get properties and actions of a parllel gripper
    (a gripper that has two fingers).

    Args:
        end_effector_prim_path (str): prim path of the Prim that corresponds to the gripper root/ end effector.
        joint_prim_names (List[str]): the left finger joint prim name and the right finger joint prim name respectively.
        joint_opened_positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively when opened.
        joint_closed_positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively when closed.
        action_deltas (np.ndarray, optional): deltas to apply for finger joint positions when openning or closing the gripper. Defaults to None.
    """

    def __init__(
        self,
        arm_prim_path: str,
        joint_prim_names: List[str],
    ) -> None:
        prim = get_prim_at_path(arm_prim_path)
        self._joint_prim_names = joint_prim_names
        self._joint_dof_indicies = np.array([None] * 6)
        self._get_joint_positions_func = None
        self._set_joint_positions_func = None
        self._articulation_num_dofs = None
        Articulation.__init__(
            self,
            prim_path=arm_prim_path,
            articulation_controller=None
        )
        return

    @property
    def joint_dof_indicies(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: joint dof indices in the articulation of the left finger joint and the right finger joint respectively.
        """
        return self._joint_dof_indicies

    @property
    def joint_prim_names(self) -> List[str]:
        """
        Returns:
            List[str]: the left finger joint prim name and the right finger joint prim name respectively.
        """
        return self._joint_prim_names

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            articulation_apply_action_func (Callable): apply_action function from the Articulation class.
            get_joint_positions_func (Callable): get_joint_positions function from the Articulation class.
            set_joint_positions_func (Callable): set_joint_positions function from the Articulation class.
            dof_names (List): dof names from the Articulation class.
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None

        Raises:
            Exception: _description_
        """
        Articulation.initialize(self, physics_sim_view=physics_sim_view)
        self._get_joint_positions_func = get_joint_positions_func
        self._articulation_num_dofs = len(self.dof_names)

        for index in range(len(self._joint_prim_names)):
            if self._joint_prim_names[index] == dof_names[index]:
                self._joint_dof_indicies[index] = index
        # make sure that all gripper dof names were resolved
        if self._joint_dof_indicies[:] is None:
            raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")
        self._articulation_apply_action_func = articulation_apply_action_func
        current_joint_positions = get_joint_positions_func()
        self._set_joint_positions_func = set_joint_positions_func
        return

    def set_default_state(self, joint_positions: np.ndarray) -> None:
        """Sets the default state of the gripper

        Args:
            joint_positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
        """
        self._default_state = joint_positions
        return

    def get_default_state(self) -> np.ndarray:
        """Gets the default state of the gripper

        Returns:
            np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
        """
        return self._default_state

    def post_reset(self):
        Robot.post_reset(self)
        self._set_joint_positions_func(
            positions=self._default_state, joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]]
        )
        return

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """
        Args:
            positions (np.ndarray): joint positions of the left finger joint and the right finger joint respectively.
        """
        self._set_joint_positions_func(
            positions=positions, joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]]
        )
        return

    def get_joint_positions(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: joint positions of the left finger joint and the right finger joint respectively.
        """
        return self._get_joint_positions_func(joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]])


    def apply_action(self, control_actions: ArticulationAction) -> None:
        """Applies actions to all the joints of an articulation that corresponds to the ArticulationAction of the finger joints only.

        Args:
            control_actions (ArticulationAction): ArticulationAction for the left finger joint and the right finger joint respectively.
        """
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self._articulation_num_dofs
            for i in range(len(self._joint_prim_names)):
                joint_actions.joint_positions[self._joint_dof_indicies[i]] = control_actions.joint_positions[i]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self._articulation_num_dofs
            for i in range(len(self._joint_prim_names)):
                joint_actions.joint_velocities[self._joint_dof_indicies[i]] = control_actions.joint_velocities[i]
                
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self._articulation_num_dofs
            for i in range(len(self._joint_prim_names)):
                joint_actions.joint_efforts[self._joint_dof_indicies[i]] = control_actions.joint_efforts[i]
        self._articulation_apply_action_func(control_actions=joint_actions)
        return
