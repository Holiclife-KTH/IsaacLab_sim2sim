import sys
import os
import numpy as np
import argparse
import yaml

from isaacsim import SimulationApp

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

# Nucleus root path
assets_root_path = "omniverse://localhost/Library/Shelf"

prim_paths = {
        "env": "/World/env",
        "target": "/World/Object/target",
        "robot": "/World/Robot"
    }

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

from ur3 import UR3
from robot_controller import ArmController
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils import extensions,  prims, stage, viewports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles
import project_utils



from omni.isaac.core.utils.prims import create_prim
import omni.graph.core as og
import usdrt.Sdf

import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, Usd, UsdLux, Gf

# Labraries for Policy network
import torch
import onnx
import onnxruntime
import time
import math


# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")


class Searching_and_Retrieving:
    def __init__(self, usd_paths, model_path):
        self.scene_entities = {}
        self.action = np.zeros(shape=(1,6), dtype=np.float32)
        self.last_action = np.zeros(shape=(1,6), dtype=np.float32)
        self.world = World(stage_units_in_meters=1.0)
        self.usd_paths = usd_paths
        self.model_path = model_path
        self.robot_state_w = torch.Tensor([-0.1, 0.0, 0.79505, 0.0, 0.0, 0.0, 1.0])
        self.robot_controller = ArmController(default_state=np.array([0.0, -2.0, 2.0, 0.0, 1.57, 0.0]),control_gain=1.0)
        self.robot_controller.set_gains(kds=21)
        self.command = np.zeros(shape=(1,7), dtype=np.float32)
        
        self.initialize_world()
        
    def initialize_world(self):
        stage_utils.add_reference_to_stage(usd_path=assets_root_path + self.usd_paths["test_bench"], prim_path=prim_paths["env"])
        stage_utils.add_reference_to_stage(usd_path=assets_root_path + self.usd_paths["robot"], prim_path=prim_paths["robot"])
        

        self.objects = {"robot": self.world.scene.add(UR3(prim_path=prim_paths["robot"],
                                                        usd_path=assets_root_path + self.usd_paths["robot"],
                                                        position=[-0.1, 0.0, 0.79505],
                                                        orientation=[0.0, 0.0, 0.0, 1.0],
                                                        attach_gripper=False,
                                                        articulation_controller=self.robot_controller)),
                        "target": self.world.scene.add(VisualCuboid(
                                                        name="target_cube",
                                                        prim_path="/World/Target",
                                                        position=(0, 0, 0),
                                                        orientation=(1, 0, 0, 0),
                                                        color=np.array([1, 0, 0]),
                                                        size=1.0,
                                                        scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),)),
                            }
        
        create_prim("/DistantLight", "DistantLight", attributes={"inputs:intensity": 500})
        
        self.robot = self.objects["robot"]
        
    def action_graph(self):
        
        ROBOT_BASE_LINK = prim_paths["robot"]
        
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                        ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                        ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                        ("Context.outputs:context", "PublishJointState.inputs:context"),
                        ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                        ("Context.outputs:context", "PublishClock.inputs:context"),
                        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                        ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                        (
                            "SubscribeJointState.outputs:positionCommand",
                            "ArticulationController.inputs:positionCommand",
                        ),
                        (
                            "SubscribeJointState.outputs:velocityCommand",
                            "ArticulationController.inputs:velocityCommand",                        
                        ),
                        ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        # Setting the /UR3/base_link target prim to Articulation Controller node
                        ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_BASE_LINK)]),
                        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                        ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_BASE_LINK)]),
                        # ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(ROBOT_BASE_LINK)]),
                    ]
                }
            )
        
        except Exception as e:
            print(e)
            simulation_app.close()
            
    # Reset world state
    def reset_world(self):
        
        self.objects["robot"].set_default_state(position=[-0.1, 0.0, 0.79505], orientation=[0.0, 0.0, 0.0, 1.0])
        self.objects["robot"].set_joints_default_state(positions=np.array([0.0, -2.0, 2.0, 0.0, 1.57, 0.0]))
        time.sleep(0.5)
        self.action = np.zeros((1,6), dtype=np.float32)
        self.world.reset()

        self.generate_command(pos_x_range=(0.1, 0.35), pos_y_range=(-0.2, 0.2), pos_z_range=(0.15, 0.4), ori_x_range=(0.0, 0.0), ori_y_range=(math.pi / 2, math.pi/2), ori_z_range=(-math.pi, math.pi))

    def generate_command(self, pos_x_range, pos_y_range, pos_z_range, ori_x_range, ori_y_range, ori_z_range, num_samples=1):
        
        target = self.objects["target"]
        robot = self.objects["robot"]

        pos_x_samples = (pos_x_range[1] - pos_x_range[0]) * torch.rand(num_samples) + pos_x_range[0]
        pos_y_samples = (pos_y_range[1] - pos_y_range[0]) * torch.rand(num_samples) + pos_y_range[0]
        pos_z_samples = (pos_z_range[1] - pos_z_range[0]) * torch.rand(num_samples) + pos_z_range[0]

        ori_x_samples = (ori_x_range[1] - ori_x_range[0]) * torch.rand(num_samples) + ori_x_range[0]
        ori_y_samples = (ori_y_range[1] - ori_y_range[0]) * torch.rand(num_samples) + ori_y_range[0]
        ori_z_samples = (ori_z_range[1] - ori_z_range[0]) * torch.rand(num_samples) + ori_z_range[0]

        self.command[...,:3] = np.array([pos_x_samples, pos_y_samples, pos_z_samples]).T
        quat = project_utils.quat_from_euler_xyz(roll=ori_x_samples, pitch=ori_y_samples, yaw=ori_z_samples)
        quat = project_utils.quat_unique(quat)
        self.command[..., 3:] = np.array(quat)

        cube_pos, cube_ori = project_utils.combine_frame_transforms(self.robot_state_w[:3], self.robot_state_w[3:7], torch.tensor(self.command[...,:3]), torch.tensor(self.command[..., 3:].squeeze(0)))


        target.set_world_pose(position= cube_pos, orientation = cube_ori)
        

        
    def get_observations(self):
        
        robot = self.objects["robot"]

        self.joint_default_states = robot.get_joints_default_state()
        self.rel_joint_positions = robot.get_joint_positions()[ :6] - self.joint_default_states.positions[ :6]
        self.rel_joint_velocities = robot.get_joint_velocities()[ :6] - self.joint_default_states.velocities[ :6]
        ee_state_w = robot.end_effector.get_world_pose()
        ee_pos_w = torch.tensor(ee_state_w[0])
        ee_quat_w = torch.tensor(ee_state_w[1])
        self.ee_pos, self.ee_quat = project_utils.subtract_frame_transforms(self.robot_state_w[:3], self.robot_state_w[3:7], ee_pos_w, ee_quat_w)
        pose_command = self.command[0, :]
        self.last_action = self.action[0, :]

        # print(torch.tensor(pose_command[:3]) - self.ee_pos)

        # print(self.rel_joint_positions)
        # print(self.rel_joint_velocities)
        # print(ee_pos_w)
        # print(self.ee_quat)
        # print(pose_command)
        # print(self.last_action)
        

        
        
        
        
        self.observation_state = np.concatenate([
            self.rel_joint_positions,
            self.rel_joint_velocities,
            self.ee_pos,
            self.ee_quat,
            pose_command,
            self.last_action
        ], dtype=np.float32)

        # 2D로 변환 (batch_size=1)
        self.observation_state = np.expand_dims(self.observation_state, axis=0)

        
        
        
        return self.observation_state
        

    # Run simulation loop
    def run_simulation(self, iterations=5, steps_per_iteration=5000, test_mode=False):
     
        
        model_ort = onnxruntime.InferenceSession(self.model_path)
        
        # GPU 설정 확인
        print("Available Providers:", model_ort.get_providers())
        
        
        # input_name = model_ort.get_inputs()[0].name
        # output_name = sweeping_ort.get_outputs()[0].name
        # print("Input Name:", input_name)
        # print("Output Name:", output_name)
        
        self.reset_world()
        while simulation_app.is_running():

            self.reset_world()
            for step in range(steps_per_iteration):

                if (step % 10) == 0:
                    
                    self.world.step(render=True)

                observation = self.get_observations()

                self.action = model_ort.run(["actions"], {"obs": observation})[0]
                    
                self.action = np.clip(self.action, -3.14, 3.14)

                # print(f"action: {self.action}")

                self.robot.apply_action(self.robot_controller.forward(command=self.action[0,:]))


                    
                    

# Main execution
def main():
    
    simulation_app.update()

    # Preparing stage
    viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "config", "data_cfg.yaml")
    
    # Load configuration file
    with open(yaml_path, 'r') as file:
        usd_paths = yaml.safe_load(file)
        
    task = Searching_and_Retrieving(usd_paths=usd_paths, 
                                    model_path= current_dir + usd_paths["model_path"])
    
    task.action_graph()
    
    simulation_app.update()
    
    task.run_simulation(test_mode=args.test)

    simulation_app.close()

if __name__ == "__main__":
    main()
