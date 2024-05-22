from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from dexpoint.env.rl_env.base import BaseRLEnv
from dexpoint.env.sim_env.relocate_env import LabRelocateEnv
from dexpoint.real_world import lab

OBJECT_LIFT_LOWER_LIMIT = -0.03


class AllegroRelocateRLEnv(LabRelocateEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=10, robot_name="allegro_hand_xarm6_wrist_mounted_face_front",
                 rotation_reward_weight=0, object_category="YCB", object_name="tomato_soup_can",
                 randomness_scale=1, friction=1, root_frame="robot", **renderer_kwargs):
        if "allegro" not in robot_name or "free" in robot_name:
            raise ValueError(f"Robot name: {robot_name} is not valid xarm allegro robot.")

        super().__init__(use_gui, frame_skip, object_category, object_name, randomness_scale, friction,
                         **renderer_kwargs)

        # Base class
        self.setup(robot_name)
        self.rotation_reward_weight = rotation_reward_weight

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Base frame for observation
        self.root_frame = root_frame
        self.base_frame_pos = np.zeros(3)

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
        finger_contact_link_name = [
            "link_15.0_tip", "link_15.0", "link_14.0",
            "link_3.0_tip", "link_3.0", "link_2.0", "link_1.0",
            "link_7.0_tip", "link_7.0", "link_6.0", "link_5.0",
            "link_11.0_tip", "link_11.0", "link_10.0", "link_9.0"
        ]
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]
        self.finger_contact_links = [self.robot.get_links()[robot_link_names.index(name)] for name in
                                     finger_contact_link_name]
        self.finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4])
        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])
        self.finger_reward_scale = np.ones(len(self.finger_tip_links)) * 0.02
        # self.finger_reward_scale[0] = 0.1

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.object_pose_init = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.object_in_tip = np.zeros([len(finger_tip_names), 3])
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0
        self.lift = False

        self.print_is = False

        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four tip, palm

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        # check_contact_links = self.finger_contact_links + [self.palm_link]
        # TODO link
        palm_link = [link for link in self.robot.get_links() if link.get_name() == 'palm_center'][0]
        check_contact_links = self.finger_contact_links + [palm_link]
        # print(palm_link)
        contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
        self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_in_object = palm_link.get_pose().p - self.object_pose.p
        self.palm_pos_in_base = self.palm_pose.p - self.base_frame_pos
        self.object_in_tip = self.object_pose.p[None, :] - self.finger_tip_pos
        self.object_lift = self.object_pose.p[2] - self.object_height
        self.object_move = np.linalg.norm(self.object_pose.p[:2] - self.object_pose_init.p[:2])
        self.target_in_object = self.target_pose.p - self.object_pose.p
        hand_action = self.robot.get_qpos()[7:]
        hand_action[0] = hand_action[1]
        hand_action[1] = hand_action[2]
        hand_action[2] = hand_action[3]
        hand_action[3] = hand_action[5]
        hand_action[4] = hand_action[6]
        hand_action[5] = hand_action[7]
        hand_action[6] = hand_action[9]
        hand_action[7] = hand_action[10]
        hand_action[8] = hand_action[11]
        hand_action[9] = hand_action[12]
        hand_action[10] = hand_action[14]
        hand_action[11] = hand_action[15]
        self.hand_action = hand_action[:12]
        # hand_other = self.robot.get_qpos()[7:]
        # hand_other[0] =  hand_action[0]
        # hand_other[1] =  hand_action[4]
        # hand_other[2] =  hand_action[8]
        # self.hand_other = hand_other[:3]
        self.target_in_object_angle[0] = np.arccos(
            np.clip(np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        self.is_contact = np.sum(self.robot_object_contact) >= 2
        if self.print_is:
            print("self.object_move:", self.object_move)
            print("palm_in_object:", self.palm_in_object)
            print("object_in_tip:", self.object_in_tip)
            print("self.object_lift",self.object_lift)
            print("self.target_in_object",self.target_in_object)

    def get_oracle_state(self):
        object_pos = self.object_pose.p
        object_quat = self.object_pose.q
        object_pose_vec = np.concatenate([object_pos - self.base_frame_pos, object_quat])
        robot_qpos_vec = self.robot.get_qpos()
        return np.concatenate([
            robot_qpos_vec,
            self.palm_pos_in_base,  # dof + 3
            object_pose_vec,
            self.object_in_tip.flatten(),
            self.robot_object_contact,  # 7 + 12 + 5
            self.target_in_object,
            self.target_pose.q,
            self.target_in_object_angle  # 3 + 4 + 1
        ])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        return np.concatenate([
            robot_qpos_vec,
            self.palm_pos_in_base,
            self.target_pose.p - self.base_frame_pos,
            self.target_pose.q
        ])

    def get_reward(self, action):
        reward = 0.0
        reward += self.reward_palm_object_dis(action)
        reward += self.reward_hand_action(action)
        reward += self.reward_finger_object_dis(action)
        reward += self.reward_contact_and_lift(action)
        reward += self.reward_target_obj_dis(action)
        reward += self.reward_other(action)
        return reward / 10

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Gym reset function
        if seed is not None:
            self.seed(seed)

        self.reset_internal()
        # Set robot qpos
        qpos = np.zeros(self.robot.dof)
        # print(qpos)
        xarm_qpos = self.robot_info.arm_init_qpos
        qpos[:self.arm_dof] = xarm_qpos
        self.robot.set_qpos(qpos)
        
        # hand_action_res = [-0.0,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815,-0.78539815 ,-0.78539815 , -0.0,-0.78539815,-0.78539815,-0.78539815,-0.56,-0.78539815,-0.0,-0.78539815]
        # for i in range(16):
        #     qpos[i+7] += hand_action_res[i] 
        self.robot.set_drive_target(qpos)

        # Set robot pose
        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)

        if self.root_frame == "robot":
            self.base_frame_pos = self.robot.get_pose().p
        elif self.root_frame == "world":
            self.base_frame_pos = np.zeros(3)
        else:
            raise NotImplementedError
        self.update_cached_state()
        self.update_imagination(reset_goal=True)
        self.object_pose_init = self.manipulated_object.get_pose()
        return self.get_observation()

    def get_info(self):
        # 将reward信息返回
        info = {
            "reward_finger_object_dis": self.reward_finger_object_dis_val,
            "reward_hand_action": self.reward_hand_action_val,
            "reward_palm_object_dis": self.reward_palm_object_dis_val,
            "reward_contact_and_lift": self.reward_contact_and_lift_val,
            "reward_target_obj_dis": self.reward_target_obj_dis_val,
            "reward_other": self.reward_other_val
        }
        if self.print_is:
            print(info)
        return info

    def reward_hand_action(self, action):
        reward = 0.0
        hand_action = self.hand_action
        # hand_other = self.hand_other
        if np.sum(self.robot_object_contact) < 10 :
            hand_action = np.clip(hand_action, 0, 0.5)
            if self.palm_in_object_val < 0. :
                reward += 0.5 * (np.sum(hand_action) - np.sum(hand_action[6:9]))
                reward = np.clip(reward, 0, 5)
                if hand_action[9] < 1 :
                    reward -=  3 * (1-hand_action[9])
                for i in range(12):
                    if i == 6 or i==7 or i==8 :
                        continue
                    if hand_action[i] < 0.25 :
                        reward -=  2 * (0.25-hand_action[i])
                # reward = np.clip(reward, 0, 2.8)
                # if np.sum(self.robot_object_contact)>=3:
                #     reward = 7
            else :
                reward -= 1 * np.sum(hand_action)
                # reward -= 0.5 * np.sum(np.abs(self.hand_other))
                reward = np.clip(reward, -10, 0)
        # action
        self.reward_hand_action_val = reward
        return reward
    
    def reward_finger_object_dis(self, action):
        reward = 0.0
        if np.sum(self.robot_object_contact) < 10 :
            if self.palm_in_object_val < 0.10:
            # if True:
                finger_object_dist = np.linalg.norm(self.object_in_tip, axis=1, keepdims=False)
                if self.print_is:
                    print("finger_object_dist:", finger_object_dist)
                finger_object_dist = np.clip(finger_object_dist, 0.05, 0.8) #0.05 0.15
                reward = np.sum(1.0 / (0.01 + finger_object_dist) * self.finger_reward_scale)
                # reward = np.clip(reward, 0, 1.1)
                # if np.sum(self.robot_object_contact)>=3:
                #     reward = 1.5
        self.reward_finger_object_dis_val = reward
        return reward

    def reward_palm_object_dis(self, action):
        reward = 0.0
        if np.sum(self.robot_object_contact) < 10 :
        # if True:
            palm_in_object = np.linalg.norm(self.palm_in_object)
            self.palm_in_object_val = np.clip(palm_in_object, 0.07, 0.8) #0.13
            reward = np.sum(1.0 / (0.01 + self.palm_in_object_val) * 0.05)
            if self.print_is:
                print("palm_in_object:", self.palm_in_object_val)
        self.reward_palm_object_dis_val = reward
        return reward

    def reward_contact_and_lift(self, action):
        reward = 0.0
        self.lift = False
        if np.sum(self.robot_object_contact) >= 1:
            reward += 0.25 * np.clip(np.sum(self.robot_object_contact), 0, 3)
            # reward += 0.25 * self.robot_object_contact
        if self.is_contact:
            # reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 10 * lift
            if lift > 0.02:
                self.lift = True
        reward = reward * 2
        self.reward_contact_and_lift_val = reward
        return reward

    def reward_target_obj_dis(self, action):
        reward = 0.0
        if self.lift:
            reward = 1
            target_obj_dist = np.linalg.norm(self.target_in_object)
            reward += 1.0 / (0.04 + target_obj_dist)
        reward = reward * 2
        self.reward_target_obj_dis_val = reward
        return reward
    
    def reward_other(self, action):
        reward = 0.0
        reward += np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * -0.01
        reward += (self.cartesian_error ** 2) * -1e3
        self.reward_other_val = reward
        return reward

    def is_done(self):
        if  np.linalg.norm(self.target_in_object) < 0.04 or self.object_lift < OBJECT_LIFT_LOWER_LIMIT or (self.object_move >0.04 and self.object_lift < 0.02):
            return True
        else:
            return False

    @cached_property
    def horizon(self):
        return 200

if __name__ == '__main__':
    pass
