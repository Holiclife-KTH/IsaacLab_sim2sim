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

from KTH.project_sweeping.asset.ur3 import UR3
from KTH.project_sweeping.src.project_utils import tf_matrices_from_pose, get_local_from_world
from KTH.project_sweeping.src.robot_controller import ArmController

from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from omni.isaac.nucleus import get_assets_root_path

from omni.isaac.core.utils import extensions,  prims, stage, viewports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.transformations import get_relative_transform, pose_from_tf_matrix
from omni.isaac.core.utils.numpy.rotations import quats_to_euler_angles



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


# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")


class Searching_and_Retrieving:
    def __init__(self, usd_paths, sweeping_model_path):
        self.scene_entities = {}
        self.action = np.zeros(shape=(1,7), dtype=np.float32)
        self.last_action = np.zeros(shape=(1,7), dtype=np.float32)
        self.world = World(stage_units_in_meters=1.0)
        self.usd_paths = usd_paths
        self.sweeping_model_path = sweeping_model_path
        self.robot_controller = ArmController(default_state=np.array([0.0, -2.0, 2.0, 0.0, 1.57, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),control_gain=0.5)
        
        self.initialize_world()
        
    def initialize_world(self):
        stage_utils.add_reference_to_stage(usd_path=assets_root_path + self.usd_paths["test_bench"], prim_path=prim_paths["env"])
        stage_utils.add_reference_to_stage(usd_path=assets_root_path + self.usd_paths["target"], prim_path=prim_paths["target"])
        stage_utils.add_reference_to_stage(usd_path=assets_root_path + self.usd_paths["robot"], prim_path=prim_paths["robot"])

        self.objects = {"robot": self.world.scene.add(UR3(prim_path=prim_paths["robot"],
                                                        usd_path=assets_root_path + self.usd_paths["robot"],
                                                        position=[-0.1, 0.0, 0.79505],
                                                        orientation=[0.0, 0.0, 0.0, 1.0],
                                                        gripper_dof_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
                                                        articulation_controller=self.robot_controller)),
                        "target": self.world.scene.add(RigidPrim(prim_path=prim_paths["target"], name="target", mass=0.3, 
                                                            position=[-0.55, 0.0, 0.98], orientation=[1.0, 0.0, 0.0, 0.0])),
                        
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
        self.objects["robot"].set_joints_default_state(positions=np.array([0.0, -2.0, 2.0, 0.0, 1.57, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        time.sleep(0.5)
        self.objects["target"].set_default_state(position=[-0.7, 0.0, 0.98], orientation=[1.0, 0.0, 0.0, 0.0])
        self.action = np.zeros((1,7), dtype=np.float32)
        self.world.reset()
        
        
    def get_observations(self):
        
        robot = self.objects["robot"]
        target = self.objects["target"]
        
        self.joint_default_states = robot.get_joints_default_state()
        self.rel_joint_positions = robot.get_joint_positions()[ :8] - self.joint_default_states.positions[ :8]
        self.rel_joint_velocities = robot.get_joint_velocities()[ :8] - self.joint_default_states.velocities[ :8]
        self.object_pos_rel_r, _ = pose_from_tf_matrix(get_relative_transform(source_prim=target.prim, target_prim=robot.prim))
        self.ee_pos, self.ee_quat = pose_from_tf_matrix(get_relative_transform(source_prim=robot.end_effector.prim, target_prim=robot.prim))
        self.goal_pos, _ = get_local_from_world(parent_transform=tf_matrices_from_pose(translation=robot.get_robot_position(), orientation=np.array([0.0, 0.0, 0.0, 1.0])),
                                        position=target.get_default_state().position + np.array([0.0, 0.15, 0.0]),
                                        orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        
        self.last_action = self.action[0, :]
        
        self.ee_quat = robot.end_effector.get_world_pose()[1]
        
        # print(robot.end_effector.get_world_pose()[0])
        print(f"ee_pos: {self.ee_pos}")
        print(f"ee_quat: {self.ee_quat}")
        
        
        self.observation_state = np.concatenate([
            self.rel_joint_positions,
            self.rel_joint_velocities,
            self.object_pos_rel_r,
            self.ee_pos,
            self.ee_quat,
            self.goal_pos,
            self.last_action
        ], dtype=np.float32)

        # 2D로 변환 (batch_size=1)
        self.observation_state = np.expand_dims(self.observation_state, axis=0)
        
        
        
        return self.observation_state
        

    def reset_condition(self) -> bool:
        target = self.objects["target"]
        
        target_quat = target.get_world_pose()[1]
        
        target_euler = quats_to_euler_angles(target_quat)
        
        slipping = (np.abs(target_euler[0]) > 0.8) or (np.abs(target_euler[1]) > 0.8)

        return slipping     
        
        

        # print(f"rel_joint_positions: {rel_joint_positions}")
        # print(f"rel_joint_velocities: {rel_joint_velocities}")
        # print(f"object position relative to robot: { object_pos_rel_r}")
        # print(f"end effector position relative to robot: { ee_pos}")
        # print(f"goal position relative to robot: {goal_pos}")
        
        # print(robot.get_robot_position())
        # print(robot.get_robot_orientation())
        
    # Run simulation loop
    def run_simulation(self, iterations=5, steps_per_iteration=500, test_mode=False):
     
        
        sweeping_ort = onnxruntime.InferenceSession(self.sweeping_model_path)
        
        # GPU 설정 확인
        print("Available Providers:", sweeping_ort.get_providers())
        
        
        # input_name = sweeping_ort.get_inputs()[0].name
        # output_name = sweeping_ort.get_outputs()[0].name
        # print("Input Name:", input_name)
        # print("Output Name:", output_name)
        
        self.reset_world()
        while simulation_app.is_running():
            for step in range(steps_per_iteration):
                self.world.step(render=True)
                # print(self.reset_condition)
                if self.reset_condition():
                    self.reset_world()
                if (step % 50) == 0:
                    
                    observation = self.get_observations()
                    print(observation)
                    
   
                    self.action = sweeping_ort.run(["actions"], {"obs": observation})[0]
                    
                    self.action = np.clip(self.action, -6.28, 6.28)
                    # print(self.action)
                    # print(self.robot.dof_names)

                    # self.robot.apply_action(self.robot_controller.forward(command=self.action[0,:]))
                
                    
                
                # if reset_condition(self):
                    
                    

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
                                    sweeping_model_path= current_dir + usd_paths["sweeping_model"])
    
    task.action_graph()
    
    simulation_app.update()
    
    task.run_simulation(test_mode=args.test)

    simulation_app.close()

if __name__ == "__main__":
    main()
