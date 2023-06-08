import os

import imageio
import numpy as np
from PIL import ImageColor

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0
        use_visual_obs = True
        env_params = dict(object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False, no_rgb=False)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = AllegroRelocateRLEnv(**env_params)

        # Create camera
        camera_cfg = {
            "cam1": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
                         right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(256, 256)),
            "cam2": dict(position=np.array([-0.6, -0.3, 0.8]), look_at_dir=np.array([0.6, 0.3, -0.8]),
                         right_dir=np.array([1, -2, 0]), fov=np.deg2rad(69.4), resolution=(256, 256))
        }
        environment.setup_camera_from_config(camera_cfg)

        # Specify observation modality
        empty_info = {}  # level empty dict for default observation setting
        obs_cfg = {"cam1": {"rgb": empty_info, "segmentation": empty_info},
                   "cam2": {"depth": empty_info}}
        environment.setup_visual_obs_config(obs_cfg)
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("For state task, observation is a numpy array. For visual tasks, observation is a python dict.")

    print("Observation keys")
    print(obs.keys())
    rgb = obs["cam1-rgb"]
    rgb_pic = (rgb * 255).astype(np.uint8)
    imageio.imsave("cam1-rgb.png", rgb_pic)

    # Segmentation
    link_seg = obs["cam1-segmentation"][..., 0]
    part_seg = obs["cam1-segmentation"][..., 1]
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
    imageio.imsave("cam1-link_seg.png", color_palette[link_seg].astype(np.uint8))
    imageio.imsave("cam1-part_seg.png", color_palette[part_seg].astype(np.uint8))

    # Depth normalization
    depth = obs["cam2-depth"] / 10 * 65535
    imageio.imwrite("cam2-depth.png", depth[..., 0].astype(np.uint16))

    env.scene = None
