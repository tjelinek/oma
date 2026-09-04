import importlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import erosion, dilation

from configs.glopose_config import GloPoseConfig


def erode_segment_mask2(erosion_iterations, segment_masks):
    """

    :param erosion_iterations: int - iterations of erosion by 3x3 kernel
    :param segment_masks: Tensor of shape (N, 1, H, W)
    :return: Eroded segment mask of the same shape
    """

    kernel = torch.ones(3, 3).to(segment_masks.device)
    eroded_segment_masks = segment_masks.clone()

    for _ in range(erosion_iterations):
        eroded_segment_masks = erosion(eroded_segment_masks, kernel)
    return eroded_segment_masks


def dilate_mask(dilation_iterations, mask_tensor):
    """

    :param dilation_iterations: int - iterations of erosion by 3x3 kernel
    :param mask_tensor: Tensor of shape (1, N, 1, H, W)
    :return: Eroded segment mask of the same shape
    """

    kernel = torch.ones(3, 3).to(mask_tensor.device)
    dilated_segment_masks = mask_tensor.clone()[0]

    for _ in range(dilation_iterations):
        dilated_segment_masks = dilation(dilated_segment_masks, kernel)
    return dilated_segment_masks[None]


def mesh_normalize(vertices):
    mesh_max = torch.max(vertices, dim=1, keepdim=True)[0]
    mesh_min = torch.min(vertices, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    vertices = vertices - mesh_middle
    bs = vertices.shape[0]
    mesh_biggest = torch.max(vertices.view(bs, -1), dim=1)[0]
    vertices = vertices / mesh_biggest.view(bs, 1, 1)  # * 0.45
    return vertices


def normalize_vertices(vertices: torch.Tensor):
    vertices = vertices - vertices.mean(axis=0)
    max_abs_val = torch.max(torch.abs(vertices))
    magnification = 1.0 / max_abs_val if max_abs_val != 0 else 1.0
    vertices *= magnification

    return vertices


def compute_occlusion_mask(observed_occlusion: torch.Tensor, occlusion_threshold: float) -> torch.Tensor:
    return torch.le(observed_occlusion, occlusion_threshold)


def compute_segmentation_mask(observed_segmentation: torch.Tensor, segmentation_threshold: float) -> torch.Tensor:
    return torch.gt(observed_segmentation, segmentation_threshold)


def compute_not_occluded_foreground_mask(observed_occlusion: torch.Tensor, observed_segmentation: torch.Tensor,
                                         occlusion_threshold: float, segmentation_threshold: float) -> torch.Tensor:
    not_occluded_binary_mask = compute_occlusion_mask(observed_occlusion, occlusion_threshold)
    segmentation_binary_mask = compute_segmentation_mask(observed_segmentation, segmentation_threshold)
    return (not_occluded_binary_mask * segmentation_binary_mask).squeeze()


def get_foreground_and_segment_mask(observed_occlusion: torch.Tensor, observed_segmentation: torch.Tensor,
                                    occlusion_threshold: float, segmentation_threshold: float):
    not_occluded_binary_mask = compute_occlusion_mask(observed_occlusion, occlusion_threshold)
    segmentation_binary_mask = compute_segmentation_mask(observed_segmentation, segmentation_threshold)
    not_occluded_foreground_mask = compute_not_occluded_foreground_mask(observed_occlusion, observed_segmentation,
                                                                        occlusion_threshold, segmentation_threshold)

    return not_occluded_binary_mask, segmentation_binary_mask, not_occluded_foreground_mask


def get_not_occluded_foreground_points(observed_occlusion: torch.Tensor, observed_segmentation: torch.Tensor,
                                       occlusion_threshold: float, segmentation_threshold: float):
    _, _, not_occluded_foreground_mask = get_foreground_and_segment_mask(observed_occlusion, observed_segmentation,
                                                                         occlusion_threshold, segmentation_threshold)

    src_pts_yx = torch.nonzero(not_occluded_foreground_mask).to(torch.float32)

    return src_pts_yx, not_occluded_foreground_mask


def load_config(config_path: Path) -> GloPoseConfig:
    config_path = Path(config_path)

    spec = importlib.util.spec_from_file_location("module.name", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config_module
    spec.loader.exec_module(config_module)

    config_instance: GloPoseConfig = config_module.get_config()

    return config_instance


def print_cuda_occupied_memory(device='cuda:0'):
    # Set the device
    torch.cuda.set_device(device)

    # Get the current memory usage and the total memory.
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
    print(f"Reserved Memory: {reserved_memory / 1024 ** 2:.2f} MB")


def tensor_index_to_coordinates_xy(src_pts_yx):
    src_pts_xy = src_pts_yx.clone()
    src_pts_xy[:, [0, 1]] = src_pts_yx[:, [1, 0]]

    return src_pts_xy


def coordinates_xy_to_tensor_index(src_pts_xy):
    src_pts_yx = src_pts_xy.clone()
    src_pts_yx[:, [0, 1]] = src_pts_xy[:, [1, 0]]

    return src_pts_yx


def homogenize_3x4_transformation_matrix(T_3x4):
    T_4x4 = torch.eye(4, dtype=T_3x4.dtype).to(T_3x4.device).expand(*T_3x4.shape[:-2], 4, 4)
    T_4x4[..., :3, :] = T_3x4

    return T_4x4


def homogenize_3x3_camera_intrinsics(T_3x3):
    T_4x4 = torch.eye(4, dtype=T_3x3.dtype).to(T_3x3.device).expand(*T_3x3.shape[:-2], 4, 4)
    T_4x4[..., :3, :3] = T_3x3

    return T_4x4


def pad_to_multiple(image, multiple):
    height, width = image.shape[-2:]
    pad_h = multiple - (height % multiple)
    pad_w = multiple - (width % multiple)
    padded_image = F.pad(image, (0, pad_w, 0, pad_h))

    return padded_image, pad_h, pad_w


def extract_intrinsics_from_tensor(intrinsics: torch.Tensor) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts fx, fy, cx, cy from a 3x3 camera intrinsics matrix.

    Args:
        intrinsics (np.ndarray): A 3x3 camera intrinsics matrix.

    Returns:
        tuple: (fx, fy, cx, cy) where:
            - fx: Focal length along the x-axis (pixels)
            - fy: Focal length along the y-axis (pixels)
            - cx: Principal point x-coordinate (pixels)
            - cy: Principal point y-coordinate (pixels)
    """
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("Intrinsics matrix must be 3x3")

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    return fx, fy, cx, cy
