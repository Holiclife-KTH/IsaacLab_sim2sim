# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from IsaacLab_sim2sim.project_sweeping.src.follow_target import FollowTarget
from IsaacLab_sim2sim.controller.rmpflow import UR5eRMPFlowController

my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task",
                       offset=(0.0, 0.0, 0.14),
                       ur5e_prim_path="/World/Robot",
                       ur5e_usd_path="omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd",
                       ur5e_robot_name="ur5e")
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_controller = UR5eRMPFlowController(name="target_follower_controller", robot_articulation=my_franka)
articulation_controller = my_franka.get_articulation_controller()
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        print(observations[target_name]["position"],)
        print(observations[target_name]["orientation"])
        actions = my_controller.forward(
            target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
        )
        articulation_controller.apply_action(actions)

simulation_app.close()
