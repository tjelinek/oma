import torch
from kornia.geometry import Se3


def Se3_obj_from_epipolar_Se3_cam(Se3_cam: Se3, Se3_world_to_cam: Se3) -> Se3:
    return Se3_world_to_cam.inverse() * Se3_cam * Se3_world_to_cam


def Se3_epipolar_cam_from_Se3_obj(Se3_obj: Se3, Se3_world_to_cam: Se3) -> Se3:
    return Se3_world_to_cam * Se3_obj * Se3_world_to_cam.inverse()


def Se3_last_cam_to_world_from_Se3_obj(Se3_obj: Se3, Se3_world_to_cam: Se3) -> Se3:
    Se3_cam = (Se3_world_to_cam * Se3_obj).inverse()

    return Se3_cam


def pixel_coords_to_unit_coords(image_width: int, image_height: int, pts_yx: torch.Tensor, dtype=torch.float32) \
        -> torch.Tensor:
    return pts_yx.to(dtype) / torch.Tensor([image_height, image_width]).to(pts_yx.device)


def Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(Se3_cam_to_obj: Se3) -> Se3:
    Se3_cam_to_obj_1 = Se3_cam_to_obj[[0]]
    Se3_cam_to_obj_1_expanded = Se3.from_matrix(Se3_cam_to_obj_1.matrix().expand_as(Se3_cam_to_obj.matrix()))
    Se3_obj_1_to_obj_i = Se3_cam_to_obj_1_expanded.inverse() * Se3_cam_to_obj
    return Se3_obj_1_to_obj_i


def Se3_obj_relative_to_Se3_cam2obj(Se3_obj_relative: Se3, Se3_obj_ref_to_cam: Se3) -> Se3:
    Se3_cam2obj = Se3_obj_ref_to_cam.inverse() * Se3_obj_relative
    return Se3_cam2obj


def scale_Se3(Se3_pose: Se3, scale: float) -> Se3:
    return Se3(Se3_pose.rotation, Se3_pose.translation * scale)
