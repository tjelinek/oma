from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
from kornia.geometry import normalize_quaternion, So3, Se3
from kornia.geometry.conversions import axis_angle_to_quaternion
from kornia.geometry.quaternion import Quaternion
from kornia.image import ImageSize

from configs.glopose_config import RendererConfig
from utils.general import mesh_normalize, normalize_vertices

EncoderResult = namedtuple('EncoderResult', ['translations', 'quaternions', 'vertices', 'texture_maps',
                                             'lights'])


class Encoder(nn.Module):
    def __init__(self, renderer: RendererConfig, input_frames: int, ivertices, face_features, width, height, n_feat, texture_maps_init=None):
        super(Encoder, self).__init__()
        self.renderer = renderer
        self.input_frames = input_frames

        # Translation initialization
        translation_init = torch.Tensor(self.renderer.tran_init).repeat(input_frames, 1)

        # Quaternion initialization
        qinit = Quaternion.from_axis_angle(torch.Tensor(self.renderer.rot_init).repeat(input_frames, 1)).q

        # Initial rotations and translations
        self.register_buffer('initial_translation', translation_init.clone())
        self.register_buffer('initial_quaternion', qinit.clone())

        # Offsets initialization
        quaternion_offsets = Quaternion.identity(input_frames)
        self.register_buffer('quaternion_offsets', qinit)

        translation_offsets = torch.zeros(input_frames, 3)
        self.register_buffer('translation_offsets', translation_init)

        self.translation = nn.Parameter(torch.zeros_like(translation_offsets))
        quat = Quaternion.identity(input_frames)

        self.quaternion = nn.Parameter(quat.q)

        self.lights = None

        # Vertices initialization
        if self.renderer.optimize_shape:
            self.vertices = nn.Parameter(torch.zeros(1, ivertices.shape[0], 3))

        # Face features and texture map
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        if texture_maps_init is None:
            self.texture_map = nn.Parameter(torch.ones(1, n_feat, self.renderer.texture_size, self.renderer.texture_size))
        else:
            self.texture_map = nn.Parameter(texture_maps_init.clone())

        # Normalize and register ivertices
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)

        # Aspect ratio
        self.aspect_ratio = height / width

    def forward(self, opt_frames):

        if self.renderer.optimize_shape:
            vertices = self.ivertices + self.vertices
            if self.renderer.mesh_normalize:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:, None, :]  # make center of mass in origin
        else:
            vertices = self.ivertices

        translation = self.get_total_translation_at_frame_vectorized()
        quaternion = self.get_total_rotation_at_frame_vectorized()
        translation[0] = translation[0].detach()
        quaternion[0] = quaternion[0].detach()

        noopt = list(set(range(self.input_frames)) - set(opt_frames))

        translation[noopt] = translation[noopt].detach()
        quaternion[noopt] = quaternion[noopt].detach()

        quaternion = quaternion[:opt_frames[-1] + 1]
        translation = translation[:opt_frames[-1] + 1]

        result = EncoderResult(translations=translation,
                               quaternions=quaternion,
                               vertices=vertices,
                               texture_maps=self.texture_map,
                               lights=self.lights)
        return result

    def set_encoder_poses(self, rotations: torch.Tensor, translations: torch.Tensor):
        rotation_quaternion = axis_angle_to_quaternion(rotations)

        self.quaternion = torch.nn.Parameter(rotation_quaternion)
        self.translation = torch.nn.Parameter(translations)

    def get_total_rotation_at_frame_vectorized(self):
        q_so3 = So3(Quaternion(self.quaternion))
        q_so3_offset = So3(Quaternion(self.quaternion_offsets))

        offset_initial_quaternion = q_so3_offset * q_so3

        total_rotation_quaternion = normalize_quaternion(offset_initial_quaternion.q.q)

        return total_rotation_quaternion

    def get_se3_at_frame_vectorized(self) -> Se3:

        so3_offset = So3(Quaternion(self.quaternion_offsets))
        so3_optimized = So3(Quaternion(normalize_quaternion(self.quaternion)))

        total_rotation = so3_offset * so3_optimized

        translation_at_frame_vectorized = self.translation_offsets + self.translation

        total_se3 = Se3(total_rotation, translation_at_frame_vectorized)

        return total_se3

    def get_total_translation_at_frame_vectorized(self):
        return self.translation_offsets + self.translation

    def compute_next_offset(self, stepi):

        self.translation_offsets[stepi] = self.translation_offsets[stepi - 1] + self.translation[stepi - 1].detach()

        so3_offset = So3(Quaternion(normalize_quaternion(self.quaternion_offsets[stepi - 1])))
        so3_optim = So3(Quaternion(normalize_quaternion(self.quaternion[stepi - 1]).detach()))
        so3_new_offset = so3_offset * so3_optim

        self.quaternion_offsets[stepi] = so3_new_offset.q.q

    def frames_and_flow_frames_inference(self, keyframes, flow_frames) -> Tuple[EncoderResult, EncoderResult]:

        joined_frames = sorted(set(keyframes + flow_frames))
        not_optimized_frames = set(flow_frames) - set(keyframes)
        optimized_frames = list(sorted(set(joined_frames) - not_optimized_frames))

        joined_frames_idx = {frame: idx for idx, frame in enumerate(joined_frames)}

        frames_join_idx = [joined_frames_idx[frame] for frame in keyframes]
        flow_frames_join_idx = [joined_frames_idx[frame] for frame in flow_frames]

        joined_encoder_result: EncoderResult = self.forward(optimized_frames)

        optimized_translations = joined_encoder_result.translations[joined_frames]
        optimized_quaternions = joined_encoder_result.quaternions[joined_frames]

        keyframes_translations = optimized_translations[frames_join_idx]
        keyframes_quaternions = optimized_quaternions[frames_join_idx]
        flow_frames_translations = optimized_translations[flow_frames_join_idx]
        flow_frames_quaternions = optimized_quaternions[flow_frames_join_idx]

        encoder_result = EncoderResult(translations=keyframes_translations,
                                       quaternions=keyframes_quaternions,
                                       vertices=joined_encoder_result.vertices,
                                       texture_maps=joined_encoder_result.texture_maps,
                                       lights=joined_encoder_result.lights)

        encoder_result_flow_frames = EncoderResult(translations=flow_frames_translations,
                                                   quaternions=flow_frames_quaternions,
                                                   vertices=joined_encoder_result.vertices,
                                                   texture_maps=joined_encoder_result.texture_maps,
                                                   lights=joined_encoder_result.lights)

        return encoder_result, encoder_result_flow_frames


def init_gt_encoder(gt_mesh: 'kaolin.rep.SurfaceMesh', gt_texture: torch.Tensor, gt_Se3_obj1_to_obj_i: Se3,
                    image_shape: ImageSize, renderer: RendererConfig, input_frames: int, device: str) -> Encoder:
    from kaolin.rep import SurfaceMesh
    assert type(gt_mesh) is SurfaceMesh
    gt_rotations = gt_Se3_obj1_to_obj_i.quaternion.to_axis_angle()
    gt_translations = gt_Se3_obj1_to_obj_i.translation

    ivertices = normalize_vertices(gt_mesh.vertices).numpy()
    iface_features = gt_mesh.uvs[gt_mesh.face_uvs_idx].numpy()
    gt_encoder = Encoder(renderer, input_frames, ivertices, iface_features,
                         image_shape.width, image_shape.height, 3).to(device)
    for name, param in gt_encoder.named_parameters():
        if isinstance(param, torch.Tensor):
            param.detach_()
    gt_encoder.set_encoder_poses(gt_rotations, gt_translations)
    gt_encoder.gt_texture = gt_texture

    return gt_encoder
