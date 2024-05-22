from eigengrasp import EigenGrasp
import numpy as np

executor = EigenGrasp(16, 7).load_from_file("grasp_model.pkl")

joint_pos_min = [-0.47, -0.196, -0.174, -0.227,
                 -0.47, -0.196, -0.174, -0.227,
                 -0.47, -0.196, -0.174, -0.227,
                 0.263, -0.105, -0.189, -0.162]
joint_pos_max = [0.47, 1.61, 1.709, 1.618,
                 0.47, 1.61, 1.709, 1.618,
                 0.47, 1.61, 1.709, 1.618,
                 1.396, 1.163, 1.644, 1.719]
ranges = np.stack([
    np.array(joint_pos_min),
    np.array(joint_pos_max)], axis=0)
ranges = executor.reduce_original_dim(ranges)
ranges = ranges.transpose()
alpha=[[0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5,0.5,0.5]]
grasp=executor.compute_grasp(alpha)
print('grasp:',grasp)
print(ranges)


