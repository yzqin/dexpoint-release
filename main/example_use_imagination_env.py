import os
from time import time

import numpy as np
import open3d as o3d

from hand_teleop.env.rl_env.relocate_env import LabArmAllegroRelocateRLEnv
from hand_teleop.real_world import task_setting

if __name__ == '__main__':
    def create_env_fn():
        object_category = "ycb"
        object_name = "mustard_bottle"
        if object_name == "mustard_bottle":
            robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
        elif object_name in ["tomato_soup_can", "potted_meat_can"]:
            robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
        else:
            print(object_name)
            raise NotImplementedError

        rotation_reward_weight = 0
        use_visual_obs = True
        env_params = dict(object_name=object_name, object_category=object_category, robot_name=robot_name,
                          rotation_reward_weight=rotation_reward_weight, randomness_scale=1,
                          use_visual_obs=use_visual_obs, use_gui=False, no_rgb=True)

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = LabArmAllegroRelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify modality
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

        # Specify imagination
        environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_goal_robot"])
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("Observation keys")
    print(obs.keys())

    tic = time()
    duration = 100
    for _ in range(duration):
        obs, reward, done, info = env.step(env.action_space.sample())
    print(f"Simulation of {duration} steps task takes {time() - tic}s")

    obs = env.reset()
    action = np.zeros(22)
    action[0] = 0.1
    for _ in range(100):
        obs, reward, done, info = env.step(action)

    pc = obs["relocate-point_cloud"]
    goal_pc = obs["imagination_goal"]
    goal_robot = obs["imagination_robot"]
    imagination_cloud = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.concatenate([goal_pc, goal_robot])))
    imagination_cloud.paint_uniform_color(np.array([0, 1, 0]))
    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
    obs_cloud.paint_uniform_color(np.array([1, 0, 0]))
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([imagination_cloud, coordinate, obs_cloud])

    env.scene = None
