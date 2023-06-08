import os
from time import time

import numpy as np

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = False
        env_params = dict(object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False, no_rgb=True)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = AllegroRelocateRLEnv(**env_params)

        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("For state task, observation is a numpy array. For visual tasks, observation is a python dict.")
    print(f"Shape of observation: {obs.shape}")

    tic = time()
    rl_steps = 1000
    for _ in range(rl_steps):
        obs, reward, done, info = env.step(env.action_space.sample())

    elapsed_time = time() - tic
    simulation_steps = rl_steps * env.frame_skip
    print(f"Single process for state-only environment with {rl_steps} RL steps "
          f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
    print("Keep in mind that using multiple processes during RL training can significantly increase the speed.")
    env.scene = None
