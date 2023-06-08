import os
from time import time

import numpy as np

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world import task_setting

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        env_params = dict(object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False,
                          no_rgb=True)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = AllegroRelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify observation
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

        # Specify imagination
        environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()

    tic = time()
    rl_steps = 1000
    for _ in range(rl_steps):
        action = np.zeros(env.action_space.shape)
        action[0] = 0.002  # Moving forward ee link in x-axis
        obs, reward, done, info = env.step(action)
    elapsed_time = time() - tic

    simulation_steps = rl_steps * env.frame_skip
    print(f"Single process for point-cloud environment with {rl_steps} RL steps "
          f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
    print("Keep in mind that using multiple processes during RL training can significantly increase the speed.")
    env.scene = None
