# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
from matplotlib.pyplot import get
import numpy as np
import random, os, sys
print('path:', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dexpoint.env.rl_env.vec_env import create_vec_env
from dexpoint.env.rl_env.multi_agent_wrapper import MultiVecTaskPythonAllegro
from utils.config import set_np_formatting, set_seed, get_args, load_cfg
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    if args.algo in ["mappo", "happo", "hatrpo", "maddpg", "ippo"]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"
        if args.model_dir != "":
            cfg["is_test"] = True
        else:
            cfg["is_test"] = False

        num_workers=2
        task = create_vec_env(num_workers)
        env=MultiVecTaskPythonAllegro(task=task, num_workers=num_workers, rl_device='cuda:0')

        runner = process_MultiAgentRL(args, env=env, config=cfg_train, model_dir=args.model_dir)

        # test
        if args.play:
            runner.eval(1000)
        else:
            runner.run()

    elif args.algo in ["ppo", "ddpg", "sac", "td3", "trpo"]:
        if args.model_dir != "":
            cfg["is_test"] = True
        else:
            cfg["is_test"] = False

        num_workers=2
        task = create_vec_env(num_workers)
        env=MultiVecTaskPythonAllegro(task=task, num_workers=num_workers, rl_device='cuda:0')

        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        sarl.run(
            num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"]
        )

    else:
        print(
            "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]"
        )


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    #sim_params = parse_sim_params(args, cfg, cfg_train)
    seed=set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    cfg_train['seed']=seed
    train()
