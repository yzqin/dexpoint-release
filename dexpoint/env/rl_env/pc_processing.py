import numpy as np

from dexpoint.real_world import lab

ROBOT_TABLE_MARGIN_X = 0.06
ROBOT_TABLE_MARGIN_Y = 0.04


def process_relocate_pc(cloud: np.ndarray, camera_pose: np.ndarray, num_points: int, np_random: np.random.RandomState,
                        segmentation=None) -> np.ndarray:
    """ pc: nxm, camera_pose: 4x4 """
    if segmentation is not None:
        raise NotImplementedError

    pc = cloud[..., :3]
    pc = pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    bound = lab.RELOCATE_BOUND

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z)))[0]

    num_index = len(within_bound)
    if num_index == 0:
        return np.zeros([num_points, 3])
    if num_index < num_points:
        indices = np.concatenate([within_bound, np.ones(num_points - num_index, dtype=np.int32) * within_bound[0]])
    else:
        indices = within_bound[np_random.permutation(num_index)[:num_points]]
    cloud = np.concatenate([pc[indices, :], cloud[indices, 3:]], axis=1)

    return cloud


def process_relocate_pc_noise(cloud: np.ndarray, camera_pose: np.ndarray, num_points: int,
                              np_random: np.random.RandomState, segmentation=None, noise_level=1) -> np.ndarray:
    """ pc: nxm, camera_pose: 4x4 """
    if segmentation is not None:
        raise NotImplementedError

    pc = cloud[..., :3]
    pc = pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    bound = lab.RELOCATE_BOUND

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z)))[0]

    num_index = len(within_bound)
    if num_index == 0:
        return np.zeros([num_points, 3])
    if num_index < num_points:
        indices = np.concatenate([within_bound, np.ones(num_points - num_index, dtype=np.int32) * within_bound[0]])
        multiplicative_noise = 1 + np_random.randn(num_index)[:, None] * 0.01 * noise_level  # (num_index, 1)
        multiplicative_noise = np.concatenate(
            [multiplicative_noise, np.ones([num_points - num_index, 1]) * multiplicative_noise[0]], axis=0)
    else:
        indices = within_bound[np_random.permutation(num_index)[:num_points]]
        multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)

    processed_pc = pc[indices, :] * multiplicative_noise
    cloud = np.concatenate([processed_pc, cloud[indices, 3:]], axis=1)

    return cloud


def add_gaussian_noise(cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)
    return cloud * multiplicative_noise

#
# def batch_index_select(input_tensor, index, dim):
#     """Batch index_select
#     Code Source: https://github.com/Jiayuan-Gu/torkit3d/blob/master/torkit3d/nn/functional.py
#     Args:
#         input_tensor (torch.Tensor): [B, ...]
#         index (torch.Tensor): [B, N] or [B]
#         dim (int): the dimension to index
#     References:
#         https://discuss.pytorch.org/t/batched-index-select/9115/7
#         https://github.com/vacancy/AdvancedIndexing-PyTorch
#     """
#
#     if index.dim() == 1:
#         index = index.unsqueeze(1)
#         squeeze_dim = True
#     else:
#         assert (
#                 index.dim() == 2
#         ), "index is expected to be 2-dim (or 1-dim), but {} received.".format(
#             index.dim()
#         )
#         squeeze_dim = False
#     assert input_tensor.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(
#         input_tensor.size(0), index.size(0)
#     )
#     views = [1 for _ in range(input_tensor.dim())]
#     views[0] = index.size(0)
#     views[dim] = index.size(1)
#     expand_shape = list(input_tensor.shape)
#     expand_shape[dim] = -1
#     index = index.view(views).expand(expand_shape)
#     out = th.gather(input_tensor, dim, index)
#     if squeeze_dim:
#         out = out.squeeze(1)
#     return out
#
#
# def sample_pc(batch_pc: th.Tensor, num_points: int) -> th.Tensor:
#     """ pc: BxNxC, return: BxSxC"""
#     batch_size = batch_pc.shape[0]
#     device = batch_pc.device
#     point_size = batch_pc.shape[1]
#     sampled_pc = []
#     for i in range(batch_size):
#         random_perm = th.randperm(point_size, device=device)[:num_points]
#         sampled_pc.append(batch_pc[i][random_perm])
#     sampled_pc = th.stack(sampled_pc, dim=0)
#     return sampled_pc
#
#
# def transform_pc(pc: th.Tensor, pose: th.Tensor) -> th.Tensor:
#     """ pc: BxNx3, pose: Bx3x4 """
#     with th.no_grad:
#         transformed_pc = pose[:, :3, 3].unsqueeze(1) + th.matmul(pose[:, :3, :3], pc)
#     return transformed_pc
#
#
# def batch_process_relocate_pc(cloud: th.Tensor, camera_pose: th.Tensor, num_points: int) -> th.Tensor:
#     """ pc: BxNxC, camera_pose: Bx3x4 """
#     pc = cloud[..., :3]
#     batch = pc.shape[0]
#     with th.no_grad:
#         pc = transform_pc(pc, camera_pose)
#
#     bound = lab.RELOCATE_BOUND
#
#     # remove robot table
#     within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
#     within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
#     within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
#     within_bound = within_bound_x & within_bound_y & within_bound_z
#     within_indices = [th.nonzero(within_bound[i]) for i in range(batch)]
#     index_list = []
#     for i in range(batch):
#         indices = within_indices[i][:, 0]
#         num_index = len(indices)
#         if num_index < num_points:
#             indices = th.cat(
#                 [indices, th.zeros(num_points - num_index, dtype=indices.dtype, device=indices.device)])
#         else:
#             indices = indices[th.randperm(num_index, device=pc.device)[:num_points]]
#         index_list.append(indices)
#
#     batch_indices = th.stack(index_list)
#     batch_cloud = batch_index_select(cloud, batch_indices, dim=1)
#     return batch_cloud
