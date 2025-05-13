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
        self._mode = ["segmentation", "policy"]
        
        

    def forward(self, command, mode):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        if mode not in self._mode:
            raise ValueError("mode is not defined.")

        if mode == "segmentation":
            self.joint_position[:6] = command[:6]
            self.joint_position[6:] = np.zeros(8)


            # self.joint_vel[:6] = self.joint_position[:6] - current 

        elif mode == "policy":
            self.joint_position[:6] = command[:6] * self.control_gain
            
            if self.command[-1] <= 0:
                self.joint_position[6:] = np.array([0.5, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, -0.5])
            elif self.command[-1] > 0:
                self.joint_position[6:] = np.zeros(8)
            self.joint_position[:6] = self.joint_position[:6] + self.default_state[:6]
        
            
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_positions=self.joint_position)