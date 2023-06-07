import sapien.core as sapien
from typing import Dict

import numpy as np


def fetch_texture(cam: sapien.CameraEntity, texture_name: str, return_torch=False):
    dlpack = cam.get_dl_tensor(texture_name)
    if not return_torch:
        assert texture_name not in ["Segmentation"]
        shape = sapien.dlpack.dl_shape(dlpack)
        output_array = np.zeros(shape, dtype=np.float32)
        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dlpack, output_array)
        sapien.dlpack.dl_cuda_sync()
        return output_array
    else:
        import torch
        return torch.from_dlpack(dlpack)


def generate_imagination_pc_from_obs(obs: Dict[str, np.ndarray]):
    pc = []
    color = []
    category = []
    for key, value in obs.items():
        if "point_cloud" in key:
            num_points = value.shape[0]
            pc.append(value)
            color.append(np.tile(np.array([0, 0, 255]), [num_points, 1]))
            category.append([0] * num_points)
        elif key == "imagination_robot":
            num_points = value.shape[0]
            pc.append(value)
            color.append(np.tile(np.array([255, 0, 0]), [num_points, 1]))
            category.append([1] * num_points)
        elif key == "imagination_goal":
            num_points = value.shape[0]
            pc.append(value)
            color.append(np.tile(np.array([0, 255, 0]), [num_points, 1]))
            category.append([2] * num_points)

    pc = np.concatenate(pc)
    color = np.concatenate(color).astype(np.uint8)
    category = np.concatenate(category)[:, None]
    return pc, color, category
