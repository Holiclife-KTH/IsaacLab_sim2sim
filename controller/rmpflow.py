import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation

class UR5eRMPFlowController(mg.MotionPolicyController):
    """[summary]

    Args:
        name (str): [description]
        robot_articulation (Articulation): [description]
        physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
        attach_gripper (bool, optional): [description]. Defaults to False.
    """

    def __init__(
            self,
            name: str,
            robot_articulation: Articulation,
            physics_dt: float = 1.0 / 60.0,
            attach_gripper: bool = False,) -> None:
        
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR5e", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return
    
    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )