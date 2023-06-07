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
        self.finger_reward_scale = np.ones(len(self.finger_tip_links)) * 0.01
        self.finger_reward_scale[0] = 0.04

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.object_in_tip = np.zeros([len(finger_tip_names), 3])
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0

        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four tip, palm

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        check_contact_links = self.finger_contact_links + [self.palm_link]
        contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
        self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = self.palm_pose.p - self.base_frame_pos
        self.object_in_tip = self.object_pose.p[None, :] - self.finger_tip_pos
        self.object_lift = self.object_pose.p[2] - self.object_height
        self.target_in_object = self.target_pose.p - self.object_pose.p
        self.target_in_object_angle[0] = np.arccos(
            np.clip(np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))

    def get_oracle_state(self):
        object_pos = self.object_pose.p
        object_quat = self.object_pose.q
        object_pose_vec = np.concatenate([object_pos - self.base_frame_pos, object_quat])
        robot_qpos_vec = self.robot.get_qpos()
        return np.concatenate([
            robot_qpos_vec, self.palm_pos_in_base,  # dof + 3
            object_pose_vec, self.object_in_tip.flatten(), self.robot_object_contact,  # 7 + 12 + 5
            self.target_in_object, self.target_pose.q, self.target_in_object_angle  # 3 + 4 + 1
        ])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        return np.concatenate([
            robot_qpos_vec, self.palm_pos_in_base,
            self.target_pose.p - self.base_frame_pos, self.target_pose.q
        ])

    def get_reward(self, action):
        finger_object_dist = np.linalg.norm(self.object_in_tip, axis=1, keepdims=False)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        reward = np.sum(1.0 / (0.06 + finger_object_dist) * self.finger_reward_scale)
        # at least one tip and palm or two tips are contacting obj. Thumb contact is required.
        is_contact = np.sum(self.robot_object_contact) >= 2

        if is_contact:
            reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 10 * lift
            if lift > 0.02:
                reward += 1
                target_obj_dist = np.linalg.norm(self.target_in_object)
                reward += 1.0 / (0.04 + target_obj_dist)

                if target_obj_dist < 0.1:
                    theta = self.target_in_object_angle[0]
                    reward += 4.0 / (0.4 + theta) * self.rotation_reward_weight

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * -0.01
        controller_penalty = (self.cartesian_error ** 2) * -1e3
        return (reward + action_penalty + controller_penalty) / 10

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # Gym reset function
        if seed is not None:
            self.seed(seed)

        self.reset_internal()
        # Set robot qpos
        qpos = np.zeros(self.robot.dof)
        xarm_qpos = self.robot_info.arm_init_qpos
        qpos[:self.arm_dof] = xarm_qpos
        self.robot.set_qpos(qpos)
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
        return self.get_observation()

    def is_done(self):
        return self.object_lift < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 200


def main_env():
    from time import time
    env = AllegroRelocateRLEnv(use_gui=True, robot_name="allegro_hand_xarm6_wrist_mounted_face_front",
                               object_name="any_train", object_category="02876657", frame_skip=10,
                               use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()

    tic = time()
    env.reset()
    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    tic = time()
    for i in range(1000):
        action = np.random.rand(robot_dof) * 2 - 1
        action[2] = 0.1
        obs, reward, done, _ = env.step(action)
    tac = time()
    print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    env.reset()
    # env.reset_env()
    viewer.toggle_pause(True)
    pose = env.palm_link.get_pose()
    for i in range(5000):
        action = np.zeros(robot_dof)
        action[0] = 0.05
        obs, reward, done, _ = env.step(action)
        env.render()
        if i == 100:
            pose_error = pose.inv() * env.palm_link.get_pose()
            print(pose_error)

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
