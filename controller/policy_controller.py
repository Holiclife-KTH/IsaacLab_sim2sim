import io
from typing import Optional

import carb
import numpy as np
import omni
import omni.client
import omni.physics
import omni.physics.tensors
import torch

from IsaacLab_sim2sim.controller.base_controller import BaseController
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
import omni.isaac.core.utils.stage as stage_utils

from .config_loader import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config



class PolicyController(BaseController):
    """
    A controller that loads and executes a policy from a file.

    Args: 
        name (str): The name of the controller
        prim_path (str): The path ot the prim in the stage
        root_path (Optinal[str], None): The path to the articulation root of the robot
        usd_path (Optional[str], optional): The path to the USD file. Defaults to None.
        position (Optional [np.ndarray], optional): The initial position of the robot. Defaults to None.
        orientation (Optional [np.ndarray], optional): The initial orientation of the robot. Defaults to None.

    Attributes: 
        robot (Articulation): The robot articulation.
    """

    def __init__(
            self,
            name: str,
            prim_path: str,
            root_path: Optional[str] = None,
            usd_path: Optional[str] = None,
            position: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
            robot: Optional[Articulation] = None
    ) -> None:
        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            prim = define_prim(prim_path, "Xfrom")
            if usd_path:
                stage_utils.add_reference_to_stage(usd_path, prim_path)
                # prim.GetReferences().AddReference(usd_path)

            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if robot != None:
            self.robot = robot
        else:
            self.robot = Articulation(prim_path=prim_path, name=name, position=position, orientation=orientation)


    
    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads a policy from a file.

        Args:
            policy_file_path (str): The path to the policy file.
            policy_env_path (str): The path to the environment configuration file.
        """
        file_content = omni.client.read_file(policy_file_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        policy = torch.jit.load(file)
        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)
        return policy



    def initialize(self,
            physics_sim_view: omni.physics.tensors.SimulationView = None,
            effort_modes: str = "force",
            control_mode: str = "position",
            set_gains: bool = True,
            set_limits: bool = False,
            set_articulation_props: bool = True) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        self.robot.get_articulation_controller().switch_control_mode(control_mode)
        max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = get_robot_joint_properties(self.policy_env_params, self.robot.dof_names)
        if set_gains:
            self.robot._articulation_view.set_gains(stiffness, damping)
        if set_limits:
            self.robot._articulation_view.set_max_efforts(max_effort)
            self.robot._articulation_view.set_max_joint_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()
    def _set_articulation_props(self) -> None:
        """
        Sets the articulation root properties from the policy environment parameters
        """
        articulation_prop = get_articulation_props(self.policy_env_params)

        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collision = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
        if solver_velocity_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
        if stabilization_threshold not in [None, float("inf")]:
            self.robot.set_stabilization_threshold(stabilization_threshold)
        if isinstance(enabled_self_collision, bool):
            self.robot.set_enabled_self_collisions(enabled_self_collision)
        if sleep_threshold not in [None, float("inf")]:
            self.robot.set_sleep_threshold(sleep_threshold)

    
    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy

        Args:
            obs (np.ndarray): The observation

        Returns:
            np.ndarray: The action
        """
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented
        """
        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    def forward(self) -> NotImplementedError:
        """
        Forwards the controller. Not implemented.
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )

    def post_reset(self) -> None:
        """
        Called after the controller is reset.
        """
        self.robot.post_reset()
        
