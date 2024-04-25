import numpy as np
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
import transforms3d.quaternions
CAM2ROBOT = Pose.from_transformation_matrix(np.array(
    [[0.60346958, 0.36270068, -0.7101216, 0.962396],
     [0.7960018, -0.22156729, 0.56328419, -0.35524235],
     [0.04696384, -0.90518294, -0.42241951, 0.31896536],
     [0., 0., 0., 1.]]
))

# CAM2ROBOT = Pose()
# cam_rotation = R.from_quat([0.315856, -0.212542, 0.0472674, 0.923486])
# CAM2ROBOT.set_p([0.962396, -0.355242, 0.318965])
# CAM2ROBOT.set_q([0.315856, -0.212542, 0.0472674, 0.923486])


CAM2WORLD = Pose()
CAM2WORLD.set_p([0.35 + 0.37, 0.0, 0.35])   # [0.35 + 0.37, 0.0, 0.35]
CAM2WORLD.set_q([0.00412999, 0.186402, 0.000783578, -0.982465])
# cam_rotation = R.from_euler('xyz', [-0.48, -21, 0.0], degrees=True)
# CAM2WORLD.set_rotation(cam_rotation.as_matrix())


# sapien2opencv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
# sapien2opencv_quat = transforms3d.quaternions.mat2quat(sapien2opencv)
# pose_cam = CAM2WORLD * Pose(q=sapien2opencv_quat)
# print(pose_cam.to_transformation_matrix())

DESK2ROBOT_Z_AXIS = 0.0   # -0.05

# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.005, 0.6]


# TODO:
l_ROBOT2BASE = Pose(p=np.array([-0.37+0.125, -0.64 + 1.025, -DESK2ROBOT_Z_AXIS]))  # Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))
r_ROBOT2BASE = Pose(p=np.array([-0.37+0.125, -0.64 + 0.125, -DESK2ROBOT_Z_AXIS]))  # Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))


# Table size
TABLE_XY_SIZE = np.array([0.74, 1.28])    # origin 0.6, 1.2
TABLE_ORIGIN = np.array([0.0, 0.0]) # origin 0, -0.15


# camera = add_camera(name="", width=1024, height=768, fovy=1.57, near=0.1, far=100)
# camera.set_local_pose(Pose([1.15312, -0.0071789, 0.5284], [0.00777893, -0.347578, 0.00288369, 0.937614]))
