import os, sys
from dexpoint.env.rl_env.double_arm_env import DoubleAllegroRelocateRLEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from stable_baselines_dexpoint.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np


def create_env_fn():
        # object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        # object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        object_name='any_train'
        object_category="02876657"
        env_params = dict(object_name=object_name, object_category=object_category, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=False,robot_name="allegro_hand_xarm7",
                          no_rgb=True,frame_skip=10)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda:0"
        env = DoubleAllegroRelocateRLEnv(**env_params)
        from dexpoint.real_world import task_setting

        # Create camera
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify observation
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
        # Specify imagination
        env.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])
        
        return env

def create_vec_env(num_workers):
    return SubprocVecEnv([create_env_fn] * num_workers, "spawn")

if __name__ == '__main__':

    env = create_vec_env(5)
    env.reset()
    action = np.zeros(44)
    actions=[action for _ in range(5)]

    for i in range(100):
        obs,reward,done,info=env.step(actions)
        if i == 1:
            print('rew:', info)
            num_actions=env.get_attr('action_dim',0)
            print('action dim:', num_actions[0])