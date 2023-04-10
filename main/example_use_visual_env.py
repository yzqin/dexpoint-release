import os
from time import time

import imageio
import numpy as np
import open3d as o3d
from PIL import ImageColor

from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv

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
                          constant_object_state=False, randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False)

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        env = RelocateRLEnv(**env_params)

        # Create camera
        camera_cfg = {
            "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
                                  right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(64, 64)),
            "door_view": dict(position=np.array([-0.6, -0.3, 0.8]), look_at_dir=np.array([0.6, 0.3, -0.8]),
                              right_dir=np.array([1, -2, 0]), fov=np.deg2rad(69.4), resolution=(64, 64))
        }
        env.setup_camera_from_config(camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info},
                       "door_view": {"depth": empty_info}}
        env.setup_visual_obs_config(camera_info)
        return env


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    rgb = obs["relocate_view-rgb"]
    print("Observation keys")
    print(obs.keys())
    rgb_pic = (rgb * 255).astype(np.uint8)
    imageio.imsave("relocate-rgb.png", rgb_pic)

    # Segmentation
    link_seg = obs["relocate_view-segmentation"][..., 0]
    part_seg = obs["relocate_view-segmentation"][..., 1]
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
    imageio.imsave("relocate-link_seg.png", color_palette[link_seg].astype(np.uint8))
    imageio.imsave("relocate-part_seg.png", color_palette[part_seg].astype(np.uint8))

    tic = time()
    duration = 500
    for _ in range(duration):
        obs, reward, done, info = env.step(env.action_space.sample())
        rgb = obs["relocate_view-rgb"]

    print(f"Simulation of {duration} steps task takes {time() - tic}s")
    env.scene = None
