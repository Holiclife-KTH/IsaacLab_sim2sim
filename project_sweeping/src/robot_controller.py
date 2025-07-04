from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np

class ArmController(BaseController):
    def __init__(self, default_state:np.ndarray, control_gain=0.5):
        super().__init__(name="my_arm_controller")
        # An open loop controller that uses a unicycle model
        self.command = np.zeros(7, dtype=np.float32)
        self.control_gain = control_gain
        self.joint_position = np.zeros(14, dtype=np.float32)
        self.default_state = default_state
        self.joint_vel = np.zeros(6)
        self.joint_effort = np.zeros(6)
        
        
        

    def forward(self, command, current_joint):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).


        # self.joint_vel[:6] = self.joint_position[:6] - current 

        self.joint_position[:6] = command[:6] * self.control_gain
        
        if self.command[-1] <= 0:
            self.joint_position[6:] = np.array([0.5, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, -0.5])
        elif self.command[-1] > 0:
            self.joint_position[6:] = np.zeros(8)
        self.joint_position[:6] = self.joint_position[:6] + self.default_state[:6]

        diff = self.joint_position[:6] - current_joint
        diff[:3] = diff[:3] * 1.5    
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_positions= current_joint + diff / 3.0, joint_velocities=self.joint_vel, joint_efforts=self.joint_effort)