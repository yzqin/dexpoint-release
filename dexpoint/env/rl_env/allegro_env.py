from abc import ABC
from typing import Optional, List

import numpy as np
import sapien.core as sapien

from dexpoint.env.rl_env.base import BaseRLEnv


class AllegroRLEnv(BaseRLEnv, ABC):
    """
    This base environment are design for RL with allegro hand, either flying allegro or allegro hand with robot arm.
    It provides basic utilities based on the link name of Allegro hand and can not be used with other robot hand.
    """

    def __init__(self, use_gui=True, frame_skip=5, use_visual_obs=False, **renderer_kwargs):
        # Do not write any meaningful in this __init__ function other than type definition,
        # Since multiple parents are presented for the child RLEnv class
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        self.root_frame: str = ""
        self.base_frame_pos: np.ndarray = np.zeros(3)
        self.robot_qpos_vec: np.ndarray = np.zeros(0)

        self.finger_tip_links: List[sapien.Link] = []
        self.finger_contact_links: List[sapien.Link] = []
        self.finger_contact_ids: np.ndarray = np.array([])
        self.finger_tip_pos = np.zeros([0, 3])
        self.finger_reward_scale = np.ones(0)

        self.palm_link_name: str = ""
        self.palm_link: Optional[sapien.Link] = None
        self.palm_pose: sapien.Pose = sapien.Pose()
        self.palm_pos_in_base: np.ndarray = np.zeros(3)

    def setup_allegro(self, robot_name="allegro_hand_free", root_frame="robot"):
        if "allegro" not in robot_name:
            raise ValueError(f"Robot name: {robot_name} is not valid allegro robot.")

        # Setup Allegro hand
        self.setup(robot_name)
        self.root_frame = root_frame
        self.robot_qpos_vec = np.zeros(self.robot.dof)

        # Build fingertip related information with order: thumb, index, middle, ring
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

        # Build palm related information
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)

    def reset_allegro(self, *, seed: Optional[int] = None, robot_init_pose: sapien.Pose):
        if seed is not None:
            self.seed(seed)
        self.reset_internal()

        # Set robot qpos and pose
        qpos = np.zeros(self.robot.dof)
        xarm_qpos = self.robot_info.arm_init_qpos
        if not self.is_robot_free:
            qpos[:self.arm_dof] = xarm_qpos
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)
        self.robot.set_pose(robot_init_pose)

        # Update cached properties
        if self.root_frame == "robot":
            self.base_frame_pos = self.robot.get_pose().p
        elif self.root_frame == "world":
            self.base_frame_pos = np.zeros(3)
        else:
            raise NotImplementedError
        self.update_cached_state()
        self.update_imagination(reset_goal=True)
