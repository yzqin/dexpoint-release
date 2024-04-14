import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import time

import argparse
from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world import task_setting
from stable_baselines_dexpoint.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines_dexpoint.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines_dexpoint.ppo import PPO
from stable_baselines_dexpoint.simple_callback import SimpleCallback
import torch

BASE_DIR = os.path.abspath((os.path.join(os.path.dirname(__file__), '..')))
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
SAVE_DIR=os.path.join(BASE_DIR,'assets/checkpoints/')+time_now


def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "relocate-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in task_setting.IMG_CONFIG['relocate_robot_only'].keys()],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="laptop")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=SAVE_DIR)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.n
    print(f"freeze: {args.freeze}")

    
    

    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        # object_name='any_train'
        # object_category="02876657"
        env_params = dict(robot_name="allegro_hand_xarm7",object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=True,
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


    # def create_eval_env_fn():
    #     unseen_indeces = TRAIN_CONFIG[task_name]['unseen']
    #     environment = create_env(task_name=task_name,
    #                              use_visual_obs=True,
    #                              use_gui=False,
    #                              is_eval=True,
    #                              pc_noise=True,
    #                              index=unseen_indeces,
    #                              img_type='robot',
    #                              rand_pos=rand_pos,
    #                              rand_degree=rand_degree)
    #     return environment


    #env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.
    env=create_env_fn()

    checkpoint_path='/home/wyk/Dex/dexpoint-release/assets/checkpoints/20240411-1803/model_1000.zip'
    print(f"checkpoint_path: {checkpoint_path}")
    policy = PPO.load(checkpoint_path, env, 'cuda',
                      policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                      check_obs_space=False, force_load=True)
    print("Policy loaded")
    
    
    
    
    while True:
        obs = env.reset()

        for j in range(env.horizon):
            if isinstance(obs, dict):
                for key, value in obs.items():
                    obs[key] = value[np.newaxis, :]
            else:
                obs = obs[np.newaxis, :]
            action = policy.predict(observation=obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            print(f"reward = {reward}")
            env.render()
            if done:
                break
