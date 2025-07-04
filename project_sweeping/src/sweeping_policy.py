from typing import Optional

import carb
import numpy as np
import torch
import omni.kit.commands
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.utils.types import ArticulationAction
from IsaacLab_sim2sim.controller import PolicyController
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path

from IsaacLab_sim2sim.project_sweeping.asset.ur5e_2f85 import UR5e_2f85
from IsaacLab_sim2sim.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world, load_yaml_config, load_and_reshape_pose


class ShelfSearchingPolicy(PolicyController):

    def __init__(
            self,
            prim_path: str,
            root_path: Optional[str] = None,
            name: str = "sweeping",
            usd_path: str = None,
            sweeping_policy_path: str = None,
            sweeping_policy_params_path: str = None,
            grasping_policy_path: str = None,
            grasping_policy_params_path: str = None,
            position: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
            object_dict: Optional[dict] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot
        """

        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        robot = UR5e_2f85(prim_path=prim_path,
                        position=[0.0, 0.0, 0.79505],
                        orientation=[0.0, 0.0, 0.0, 1.0],
                        arm_dof_names=["shoulder_pan_joint",
                                        "shoulder_lift_joint",
                                        "elbow_joint",
                                        "wrist_1_joint",
                                        "wrist_2_joint",
                                        "wrist_3_joint"],
                        gripper_dof_names=["finger_joint", "right_outer_knuckle_joint"],)


        super().__init__(name, prim_path, root_path, usd_path, position, orientation, robot)

        self._sweeping_policy = self.load_policy(sweeping_policy_path, sweeping_policy_params_path)
        self._grasping_policy = self.load_policy(grasping_policy_path, grasping_policy_params_path)
        self._action_scale = 0.5
        self._action = np.zeros(7)
        self._previous_action = np.zeros(7)
        self._processed_action = np.zeros(14)
        self._policy_counter = 0
        self._done = False

        if object_dict is not None:
            self._object_dict = object_dict
        else:
            carb.log_error("You should set the object dictionary.")



    def _compute_action(self, mode: str, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy

        Args:
            obs (np.ndarray): The observation

        Returns:
            np.ndarray: The action
        """

        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            if mode == "grasping":
                action = self._grasping_policy(obs).detach().view(-1).numpy()
            
            elif mode == "sweeping":
                action = self._sweeping_policy(obs).detach().view(-1).numpy()
        return action

    def _check_done(self, mode:str, target: str):
        obj = self._object_dict[target]

        ee_pose_w = self.robot.tcp.get_world_pose()[0]
        obj_pos_w = obj.get_current_dynamic_state().position
        goal_pos_w = obj.get_default_state().position + (0.0, 0.15, 0.0)

        joint_position = self.robot.get_joint_positions()[:6]

        if mode == "sweeping":
            self._done = (np.linalg.norm(goal_pos_w - obj_pos_w, ord=1) < 0.05) & (np.linalg.norm(self.default_pos[:6] - joint_position, ord=1) < 0.2)

        elif mode == "grasping":
            self._done = (np.linalg.norm(obj_pos_w - ee_pose_w, ord=1) < 0.05) & (np.linalg.norm(self.default_pos[:6] - joint_position, ord=1) < 1.0)

        return self._done


    def _compute_observation(self, mode:str, target:str,):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """

        obj = self._object_dict[target]

        # get joint states
        joint_pos_rel = self.robot.get_joint_positions()[:8] - self.default_pos[:8]
        joint_vel_rel = self.robot.get_joint_velocities()[:8] - self.default_vel[:8]

        
        actions = self._previous_action
        ee_pose_w = self.robot.tcp.get_world_pose()
        
        ee_pos_b, ee_quat_b = get_local_from_world(parent_transform=tf_matrices_from_pose(translation=self.robot.get_default_state().position, orientation=np.array([0.0, 0.0, 0.0, 1.0])),
                                                     position=ee_pose_w[0],
                                                     orientation=ee_pose_w[1])

        obj_pos_w = obj.get_current_dynamic_state().position
        obj_quat_w = obj.get_current_dynamic_state().orientation
        obj_pos_b, obj_quat_b = get_local_from_world(parent_transform=tf_matrices_from_pose(translation=self.robot.get_default_state().position, orientation=np.array([0.0, 0.0, 0.0, 1.0])),
                                                     position=obj_pos_w,
                                                     orientation=obj_quat_w)

        goal_pos_w = obj.get_default_state().position
        goal_quat_w = obj.get_default_state().orientation
        goal_pos_b, _ = get_local_from_world(parent_transform=tf_matrices_from_pose(translation=self.robot.get_default_state().position, orientation=np.array([0.0, 0.0, 0.0, 1.0])),
                                                     position=goal_pos_w + np.array([0.0, 0.15, 0.0]),
                                                     orientation=goal_quat_w)
        if mode == "grasping":
            obs = np.concatenate([
                joint_pos_rel,
                joint_vel_rel,
                actions,
                obj_pos_b,
                ee_pos_b,
                ee_quat_b,
            ], dtype=np.float32)
        
        elif mode == "sweeping":
            obs = np.concatenate([
                joint_pos_rel,
                joint_vel_rel,
                actions,
                obj_pos_b,
                ee_pos_b,
                ee_quat_b,
                goal_pos_b
            ], dtype=np.float32)

        return obs
    
    

    def forward(self, mode:str, target:str):
        """
        Compute the desired position and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world
        
        """
        
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(mode=mode, target=target)
            self._action = self._compute_action(mode=mode, obs=obs)
            self._action = np.clip(self._action, -3.14, 3.14)
            self._previous_action = self._action.copy()

        self._processed_action[:6] = self.default_pos[:6] + (self._action[:6] * self._action_scale)
        if self._action[-1] < 0:
            self._processed_action[6:] = [0.5, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, -0.5]
        else:
            self._processed_action[6:] = np.zeros(8)
         
        # print(self._processed_action)
        action = ArticulationAction(joint_positions=self._processed_action)
        # print(action.joint_positions)
        self.robot.apply_action(action)

        self._policy_counter += 1