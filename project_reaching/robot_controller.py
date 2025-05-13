from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np

class ArmController(BaseController):
    def __init__(self, default_state:np.ndarray, control_gain=1.0):
        super().__init__(name="my_arm_controller")
        # An open loop controller that uses a unicycle model
        self.command = np.zeros(7, dtype=np.float32)
        self.control_gain = control_gain
        self.joint_velocity = np.zeros(6, dtype=np.float32)
        self.default_state = default_state
        
        

    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        
        self.joint_velocity = np.zeros(6, dtype=np.float32)
        self.joint_velocity[:6] = command[:6] * self.control_gain
        
        self.joint_velocity = self.joint_velocity + self.default_state
        
        # # print(self.joint_position)
            
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=self.joint_velocity)