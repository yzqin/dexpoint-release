import os
from time import time

import numpy as np
import open3d as o3d

from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.real_world import task_setting

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        if object_name == "mustard_bottle":
            robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
        elif object_name in ["tomato_soup_can", "potted_meat_can"]:
            robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
        else:
            raise NotImplementedError
        rotation_reward_weight = 0
        use_visual_obs = True
        env_params = dict(object_name=object_name, robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                          constant_object_state=False, randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False,
                          no_rgb=True)

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = RelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify modality
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("Observation keys")
    print(obs.keys())
    pc = obs["relocate-point_cloud"]
    # Note1: point cloud are represented in (OpenGL/Blender) convention, visual the world coordinate to get a sense
    # Note2: you may also need to remove the points with smaller depth
    cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud, coordinate])

    tic = time()
    duration = 5000
    for _ in range(duration):
        obs, reward, done, info = env.step(env.action_space.sample())

    print(f"Simulation of {duration} steps task takes {time() - tic}s")
    env.scene = None
