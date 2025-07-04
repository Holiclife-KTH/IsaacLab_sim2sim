from typing import List, Optional, Sequence

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.robots import Robot
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.controllers.articulation_controller import ArticulationController

from IsaacLab_sim2sim.project_sweeping.src.articulation.ur5e import UR5e


class UR5e_2f85(Articulation):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur5e_2f85",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        end_effector_prim_name: Optional[str] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = None,
        gripper: ParallelGripper = None,
        arm_dof_names: Optional[Sequence[str]]=None,
        gripper_dof_names: Optional[Sequence[str]]=None,
        gripper_open_position: Optional[Sequence[float]] = None,
        gripper_closed_position: Optional[Sequence[float]] = None,
        deltas: Optional[Sequence[float]] = None,
        articulation_controller: Optional[ArticulationController] = None
    ) -> None:
        self._arm = None
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self._tcp_prim_path = prim_path + "/robotiq_base_link" + "/tcp"

        if self._end_effector_prim_name is None:
            self._end_effector_prim_path = prim_path + "/robotiq_base_link"
        else:
            self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        if gripper_dof_names is None:
            gripper_dof_names = ["finger_joint", "right_outer_knuckle_joint"]
        if gripper_open_position is None:
            gripper_open_position = np.array([0.0, 0.0]) / get_stage_units()
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.5, 0.5])
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=articulation_controller)
        if arm_dof_names is not None:
            self._arm = UR5e(
                arm_prim_path=prim_path,
                joint_prim_names=arm_dof_names
            )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.0, 0.0]) / get_stage_units()
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
        return

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            Gripper: [description]
        """
        return self._gripper
    
    @property
    def tcp(self) -> XFormPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        
        """
        return self._tcp

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._tcp = XFormPrim(prim_path=self._tcp_prim_path, name=self.name + "_tcp", translation=(0.13, 0.0, 0.0))

        self._end_effector.initialize(physics_sim_view)
        self._tcp.initialize(physics_sim_view)

        self._arm.initialize(physics_sim_view=physics_sim_view,
                             articulation_apply_action_func=self.apply_action,
                             get_joint_positions_func=self.get_joint_positions,
                             set_joint_positions_func=self.set_joint_positions,
                             dof_names=self.dof_names)

        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return
    
    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
