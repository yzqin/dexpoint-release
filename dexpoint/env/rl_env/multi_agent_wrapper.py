# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from tabnanny import process_tokens
from tracemalloc import start
from gym import spaces

import torch
import numpy as np
import copy


# VecEnv Wrapper for RL training
class MultiVecTaskAllegro():
    def __init__(self, task, rl_device, num_workers, clip_observations=5.0, clip_actions=1.0):
        self.task = task
        
        self.num_environments = num_workers
        self.num_states = task.get_attr('observation_space')[0]['oracle_state'].shape[0]

        # self.num_robot_imagination=task.get_attr('observation_space')[0]['l_imagination_robot'].shape[0]
        # self.num_object_cloud=task.get_attr('observation_space')[0]['relocate-point_cloud'].shape[0]
        self.num_prop_state=23+3+7

        # self.agent_index = self.task.agent_index
        self.num_agents = 2  # used for multi-agent environments
        self.num_actions = int(task.get_attr('action_dim')[0]/self.num_agents)


        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device


        # COMPATIBILITY
        #shape_obs=self.num_robot_imagination+self.num_object_cloud+self.num_prop_state
        shape_obs=self.num_prop_state
        self.obs_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(shape_obs,)) for _ in range(self.num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_states,)) for _ in
                                        range(self.num_agents)]
        self.act_space = tuple([spaces.Box(low=np.ones(self.num_actions) * -clip_actions,
                                    high=np.ones(self.num_actions) * clip_actions) for _ in
                                range(self.num_agents)])

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents}
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

# Python CPU/GPU Class
class MultiVecTaskPythonAllegro(MultiVecTaskAllegro):

    def get_state(self):
        oracle_state=torch.from_numpy(np.stack(self.task.env_method('get_oracle_state')))
        print('oracle_state:', oracle_state.shape)
        return torch.clamp(oracle_state, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):

        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))
        actions = a_hand_actions

        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, dones,_=self.task.step(actions_tensor.cpu().numpy())

        hand_obs = []
        # l_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['l_imagination_robot'], observations['relocate-point_cloud'], observations['state'][:, :23],
        #                                 observations['state'][:, 46:49], observations['state'][:, -7:]], axis=1)),
        #                                 -self.clip_obs, self.clip_obs).to(self.rl_device)
        l_obs=torch.clamp(torch.from_numpy(np.concatenate([ observations['state'][:, :23],observations['state'][:, 46:49], observations['state'][:, -7:]], axis=1)),
                                        -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(l_obs)
        # r_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['r_imagination_robot'], observations['relocate-point_cloud'], observations['state'][:, 23:46],
        #                                 observations['state'][:, 49:52], observations['state'][:, -7:]], axis=1)),
        #                                 -self.clip_obs, self.clip_obs).to(self.rl_device)
        r_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['state'][:, 23:46],observations['state'][:, 49:52], observations['state'][:, -7:]], axis=1)),
                                        -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(r_obs)
        
        state_buf = torch.clamp(torch.from_numpy(observations['oracle_state']), -self.clip_obs, self.clip_obs).to(self.rl_device)

        rewards = torch.from_numpy(rewards).unsqueeze(-1).to(self.rl_device)
        dones = torch.from_numpy(dones).to(self.rl_device)

        sub_agent_obs = []
        agent_state = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.num_environments, self.num_actions * self.num_agents], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        observations, rewards, dones,_=self.task.step(actions.cpu().numpy())

        hand_obs = []
        # l_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['l_imagination_robot'], observations['relocate-point_cloud'], observations['state'][:, :23],
        #                                 observations['state'][:, 46:49], observations['state'][:, -7:]], axis=1)),
        #                                 -self.clip_obs, self.clip_obs).to(self.rl_device)
        l_obs=torch.clamp(torch.from_numpy(np.concatenate([ observations['state'][:, :23],observations['state'][:, 46:49], observations['state'][:, -7:]], axis=1)),
                                        -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(l_obs)
        # r_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['r_imagination_robot'], observations['relocate-point_cloud'], observations['state'][:, 23:46],
        #                                 observations['state'][:, 49:52], observations['state'][:, -7:]], axis=1)),
        #                                 -self.clip_obs, self.clip_obs).to(self.rl_device)
        r_obs=torch.clamp(torch.from_numpy(np.concatenate([observations['state'][:, 23:46],observations['state'][:, 49:52], observations['state'][:, -7:]], axis=1)),
                                        -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(r_obs)
        state_buf = torch.clamp(torch.from_numpy(observations['oracle_state']), -self.clip_obs, self.clip_obs).to(self.rl_device)
        sub_agent_obs = []
        agent_state = []

        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)

        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, state_all, None

if __name__ == '__main__':
    from vec_env import create_vec_env
    
    num_workers=5
    task = create_vec_env(num_workers)
    env=MultiVecTaskPythonAllegro(task=task, num_workers=num_workers, rl_device='cuda:0')
    env.get_state()
    
    actions=[torch.zeros(num_workers,22) for _ in range(2)]
    env.reset()
    env.step(actions)