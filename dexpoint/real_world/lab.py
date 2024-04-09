import numpy as np
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

CAM2ROBOT = Pose.from_transformation_matrix(np.array(
    [[0.60346958, 0.36270068, -0.7101216, 0.962396],
     [0.7960018, -0.22156729, 0.56328419, -0.35524235],
     [0.04696384, -0.90518294, -0.42241951, 0.31896536],
     [0., 0., 0., 1.]]
))

# CAM2ROBOT = Pose.from_transformation_matrix(np.array(

#     [[-0.8660254, -0.       , -0.5       ,   0.60 -(-0.37+0.125)],
#      [ 0.       ,  1.       , -0.        ,   0.0 - (-0.64 + 0.125)],
#      [ 0.5      ,  0.       , -0.8660254 ,   0.35],
#      [0., 0., 0., 1.]]
# ))


CAM2WORLD = Pose()
CAM2WORLD.set_p([0.35 + 0.37, 0.0, 0.35])

cam_rotation = R.from_euler('xyz', [0, 0, np.pi/2],)
CAM2WORLD.set_rotation(cam_rotation.as_matrix())

DESK2ROBOT_Z_AXIS = 0.0   # -0.05

# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.005, 0.6]


# TODO:
ROBOT2BASE = Pose(p=np.array([-0.37+0.125, -0.64 + 0.125, -DESK2ROBOT_Z_AXIS]))  # Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))

# Table size
TABLE_XY_SIZE = np.array([0.74, 1.28])    # origin 0.6, 1.2
TABLE_ORIGIN = np.array([0.0, 0.0]) # origin 0, -0.15

