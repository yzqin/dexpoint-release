import os
import random
import torch.nn as nn
import numpy as np
import time

import argparse
from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world import task_setting
from stable_baselines_dexpoint.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines_dexpoint.ppo import PPO

from simple_pc import SimplePointCloud


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


def create_env_fn():
    object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
    object_name = np.random.choice(object_names)
    # object_category="YCB"
    object_name = "any_train"
    object_category = "02876657"
    env_params = dict(robot_name="allegro_hand_xarm7",
                      object_name=object_name,
                      rotation_reward_weight=0,
                      randomness_scale=1,
                      use_visual_obs=True,
                      use_gui=True,
                      frame_skip=10,
                      no_rgb=True,
                      object_category=object_category)

    environment = AllegroRelocateRLEnv(**env_params)

    environment.setup_camera_from_config(
        task_setting.CAMERA_CONFIG["relocate"])
    environment.setup_visual_obs_config(
        task_setting.OBS_CONFIG["relocate_noise"])
    environment.setup_imagination_config(
        task_setting.IMG_CONFIG["relocate_robot_only"])
    return environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze',
                        action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="allegro_hand_xarm7")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default="20240425-1907")
    parser.add_argument('--model_path', type=str, default="model_230.zip")
    parser.add_argument('--horizon', type=str, default="200")
    args = parser.parse_args()

    task_name = args.task_name
    args.seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    args.dir = os.path.join(os.path.dirname(__file__), '..')
    args.pretrain_path = os.path.join(
        args.dir, 'assets/checkpoints/', args.pretrain_path, args.model_path)

    env = create_env_fn()

    policy = PPO.load(args.pretrain_path,
                      env,
                      'cuda',
                      policy_kwargs=get_3d_policy_kwargs(
                          extractor_name=args.extractor_name),
                      check_obs_space=False,
                      force_load=True)
    print("Policy loaded")
    simple_pc = SimplePointCloud()

    while True:
        obs = env.reset()
        for j in range(env.horizon):
            if isinstance(obs, dict):
                for key, value in obs.items():
                    obs[key] = value[np.newaxis, :]
            else:
                obs = obs[np.newaxis, :]
            # 固定距离
            # print(obs['state'][0][26:])
            # obs['state'][0][26:29] = [0.40, -0.15, 0.2]
            # print(obs['state'][0][7:23])
            # time.sleep(2)

            action = policy.predict(observation=obs, deterministic=True)[0]
            # print("action",action)
            # action[6:0] = action[6:] - [-0.0,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815,-0.78539815 ,-0.78539815 , -0.0,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815]

            obs, reward, done, _ = env.step(action)
            simple_pc.render(obs, is_imitation=True)
            env.render()
            time.sleep(0.05)
            if done:
                break
    simple_pc.vis.destroy_window()
