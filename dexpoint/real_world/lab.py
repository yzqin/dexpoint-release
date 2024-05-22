import numpy as np
from sapien.core import Pose

# CAM2ROBOT = Pose.from_transformation_matrix(np.array(
#     [[0.60346958, 0.36270068, -0.7101216, 0.962396],
#      [0.7960018, -0.22156729, 0.56328419, -0.35524235],
#      [0.04696384, -0.90518294, -0.42241951, 0.31896536],
#      [0., 0., 0., 1.]]
# ))

# CAM2ROBOT = Pose.from_transformation_matrix(np.array(
# [[-0.76266186  ,0.31902235 ,-0.56264698 , 0.42],
#             [ 0.64610816 , 0.33562137 ,-0.68549438,  0.51],
#             [-0.02985168 ,-0.88633122 ,-0.46208856  ,0.56],
#             [ 0.       ,   0.      ,    0.   ,       1.        ]]
# ))
CAM2ROBOT = Pose.from_transformation_matrix(np.array([[-0.63523252  ,0.46191307, -0.61896362,  0.37498969],
 [ 0.76830547 , 0.29632912 ,-0.56735858 , 0.47422152],
 [-0.0786534,  -0.83595775, -0.54312823 , 0.50563454],
 [ 0.       ,   0.         , 0.        ,  1.        ]]))

DESK2ROBOT_Z_AXIS = 0.00

# Relocate
RELOCATE_BOUND = [0.1, 0.8, -0.4, 0.4, DESK2ROBOT_Z_AXIS + 0.001, 0.6]

# TODO:
ROBOT2BASE = Pose(p=np.array([0.0, 0., DESK2ROBOT_Z_AXIS]))

# Table size
TABLE_XY_SIZE = np.array([0.6, 1.2])
TABLE_ORIGIN = np.array([0, -0.15])

