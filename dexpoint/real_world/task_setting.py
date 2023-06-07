from dexpoint.env.rl_env.pc_processing import process_relocate_pc, add_gaussian_noise, process_relocate_pc_noise
from dexpoint.real_world import lab
import numpy as np

# Camera config
CAMERA_CONFIG = {
    "relocate": {
        "relocate": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(64, 64),
                         ), },
    "viz_only": {  # only for visualization (human), not for visual observation
        "relocate_viz": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(640, 480), ),
        "door_viz": dict(position=np.array([-0.6, -0.3, 0.8]), look_at_dir=np.array([0.6, 0.3, -0.8]),
                         right_dir=np.array([1, -2, 0]), fov=np.deg2rad(69.4), resolution=(640, 480))},
}

# Observation config type
OBS_CONFIG = {
    "relocate": {
        "relocate": {"point_cloud": {"process_fn": process_relocate_pc, "num_points": 512}},
    },
    "relocate_noise": {
        "relocate": {
            "point_cloud": {"process_fn": process_relocate_pc_noise, "num_points": 512, "pose_perturb_level": 0.5,
                            "process_fn_kwargs": {"noise_level": 0.5}},
        },
    }
}

# Imagination config type
IMG_CONFIG = {
    "relocate_goal_only": {
        "goal": {"target_object": 64},
    },
    "relocate_robot_only": {
        "robot": {
            "link_15.0_tip": 8, "link_3.0_tip": 8, "link_7.0_tip": 8, "link_11.0_tip": 8,
            "link_15.0": 8, "link_3.0": 8, "link_7.0": 8, "link_11.0": 8,
            "link_14.0": 8, "link_2.0": 8, "link_6.0": 8, "link_10.0": 8,
        },
    },
    "relocate_goal_robot": {
        "goal": {"target_object": 64},
        "robot": {
            "link_15.0_tip": 8, "link_3.0_tip": 8, "link_7.0_tip": 8, "link_11.0_tip": 8,
            "link_15.0": 8, "link_3.0": 8, "link_7.0": 8, "link_11.0": 8,
            "link_14.0": 8, "link_2.0": 8, "link_6.0": 8, "link_10.0": 8,
        },
    },
}
