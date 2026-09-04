from collections import namedtuple
from typing import Tuple

import kaolin
import torch
import torch.nn as nn
from kornia.geometry import Se3, Quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix

from data_structures.keyframe_buffer import SyntheticFlowObservation
from models.encoder import EncoderResult, Encoder
from configs.glopose_config import RendererConfig
from utils.flow import normalize_rendered_flows

MeshRenderResult = namedtuple('MeshRenderResult', ['face_normals',
                                                   'face_vertices_cam',
                                                   'red_index',
                                                   'ren_mask',
                                                   'ren_mesh_vertices_features',
                                                   'ren_mesh_vertices_world_coords',
                                                   'ren_mesh_vertices_camera_coords',
                                                   'ren_mesh_vertices_image_coords',
                                                   'ren_face_normals'])

RenderedFlowResult = namedtuple('RenderedFlowResult', ['theoretical_flow',
                                                       'rendered_flow_segmentation',
                                                       'rendered_flow_occlusion'])

RenderingResult = namedtuple('RenderingResult', ['rendered_image',
                                                 'rendered_image_segmentation',
                                                 'rendered_face_world_coords',
                                                 'rendered_face_camera_coords',
                                                 'rendered_face_normals'])


class RenderingKaolin(nn.Module):
    def __init__(self, renderer: RendererConfig, faces: torch.Tensor, width: int, height: int):
        super().__init__()

        self.renderer = renderer
        self.height = height
        self.width = width

        self.fov = torch.pi / 4  # 45 degrees
        camera_proj = kaolin.render.camera.generate_perspective_projection(self.fov, self.width / self.height)
        self.register_buffer('camera_proj', camera_proj)

        camera_position = torch.Tensor(self.renderer.camera_position)[None]
        # camera_position[:, 2] *= -1.  # Compensating for different kaolin coordinate system
        self.register_buffer('camera_trans', camera_position)
        self.register_buffer('obj_center', torch.Tensor(self.renderer.obj_center)[None])
        camera_up_direction = torch.Tensor(self.renderer.camera_up)[None]
        self.register_buffer('camera_up', camera_up_direction)

        camera_rot, _ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center,
                                                                                camera_up_direction)

        self.register_buffer('camera_rot', camera_rot)

        self.intrinsics = (
            kaolin.render.camera.PinholeIntrinsics.from_fov(width, height, self.fov, x0=width / 2, y0=height / 2,
                                                            fov_direction=kaolin.render.camera.CameraFOV.VERTICAL))

        camera_intrinsics = pinhole_intrinsics_to_tensor(self.intrinsics).cuda()

        self.register_buffer('camera_intrinsics', camera_intrinsics)
        self.set_faces(faces)

    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))
        self.register_buffer('face_indices', self.faces.clone().detach().to(dtype=torch.long, device='cuda'))

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps,
                lights=None) -> RenderingResult:
        batch_size = quaternion.shape[0]

        rotation_matrix = quaternion_to_rotation_matrix(quaternion)

        unit_vertices_batched = unit_vertices.repeat(batch_size, 1, 1)
        face_features_batched = face_features.repeat(batch_size, 1, 1, 1)
        texture_maps_batched = texture_maps.repeat(batch_size, 1, 1, 1)

        dibr_result = self.render_mesh_with_dibr(face_features_batched, rotation_matrix, translation,
                                                 unit_vertices_batched)

        ren_features = kaolin.render.mesh.texture_mapping(dibr_result.ren_mesh_vertices_features,
                                                          texture_maps_batched, mode='bilinear')

        if lights is not None:
            im_normals = dibr_result.face_normals[0, dibr_result.red_index, :]
            lighting = None
            for li in range(lights.shape[0]):
                lighting_r = \
                    kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 0])[
                        ..., None]
                lighting_g = \
                    kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 1])[
                        ..., None]
                lighting_b = \
                    kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 2])[
                        ..., None]
                lighting_one = torch.cat((lighting_r, lighting_g, lighting_b), 3)
                if lighting is None:
                    lighting = lighting_one
                else:
                    lighting += lighting_one
            lighting[dibr_result.red_index[..., None][:, :, :, [0, 0, 0]] < 0] = 1
            ren_features = ren_features * lighting
        rendering_rgb = ren_features.permute(0, 3, 1, 2)

        renderings = rendering_rgb
        segmentations = dibr_result.ren_mask.unsqueeze(1)
        rendered_object_camera_coords = dibr_result.ren_mesh_vertices_camera_coords.permute(0, 3, 1, 2)
        rendered_object_world_coords = dibr_result.ren_mesh_vertices_world_coords.permute(0, 3, 1, 2)
        rendered_object_face_normals_camera_coords = dibr_result.ren_face_normals.permute(0, 3, 1, 2)

        rendering_result = RenderingResult(rendered_image=renderings,
                                           rendered_image_segmentation=segmentations,
                                           rendered_face_world_coords=rendered_object_world_coords,
                                           rendered_face_camera_coords=rendered_object_camera_coords,
                                           rendered_face_normals=rendered_object_face_normals_camera_coords)
        return rendering_result

    def compute_theoretical_flow(self, encoder_out_pose_2, encoder_out_pose_1, flow_arcs_indices) -> RenderedFlowResult:
        """
        Computes the theoretical flow between consecutive frames.

        Args:
            encoder_out_pose_2 (EncoderResult): The encoder result for the current frame.
            encoder_out_pose_1 (EncoderResult): The encoder results for the previous frames.
            flow_arcs_indices (Sorted[Tuple[int, int]]): Indexes in encoder_out_prev_frames and encoder_out given as a
                                                         sorted collection of tuples.

        Returns:
            torch.Tensor: The computed theoretical flow between consecutive frames. The output flow is respective to the
                          coordinates range [0, 1].
        """

        if len(flow_arcs_indices) == 0:
            theoretical_flow = torch.zeros(1, 0, 2, self.height, self.width).cuda()
            rendered_flow_segmentation = torch.zeros(1, 0, 1, self.height, self.width).cuda()
            occlusion_masks = torch.zeros_like(rendered_flow_segmentation).cuda()
            flow_result = RenderedFlowResult(theoretical_flow, rendered_flow_segmentation, occlusion_masks)
            return flow_result

        batches = self.rotations_translations_batched(encoder_out_pose_1, encoder_out_pose_2, flow_arcs_indices)
        (rotation_matrix_1_batch, rotation_matrix_2_batch,
         translation_vector_1_batch, translation_vector_2_batch) = batches

        batch_size = translation_vector_1_batch.shape[0]
        vertices_1 = encoder_out_pose_1.vertices

        batched_tensors = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1)
        camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch = batched_tensors

        batched_tensors_template = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1)
        camera_rot_batch_template, camera_trans_batch_template, _, _, _ = batched_tensors_template

        # Rotate and translate the vertices using the given rotation_matrix and translation_vector
        vertices_1_batch = kaolin.render.camera.rotate_translate_points(vertices_1_batch,
                                                                        rotation_matrix_1_batch, obj_center_batch)
        vertices_2_batch = kaolin.render.camera.rotate_translate_points(vertices_2_batch,
                                                                        rotation_matrix_2_batch, obj_center_batch)

        vertices_1_batch = vertices_1_batch + translation_vector_1_batch.unsqueeze(1)
        vertices_2_batch = vertices_2_batch + translation_vector_2_batch.unsqueeze(1)

        prepared_vertices_1 = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices_1_batch, faces=self.faces,
                                                                        camera_proj=self.camera_proj,
                                                                        camera_rot=camera_rot_batch_template,
                                                                        camera_trans=camera_trans_batch_template)
        face_vertices_cam_1, face_vertices_image_1, face_normals_1 = prepared_vertices_1

        prepared_vertices_2 = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices_2_batch, faces=self.faces,
                                                                        camera_proj=self.camera_proj,
                                                                        camera_rot=camera_rot_batch,
                                                                        camera_trans=camera_trans_batch)
        face_vertices_cam_2, face_vertices_image_2, face_normals_2 = prepared_vertices_2

        # Extract the z-coordinates of the face vertices in camera space
        face_vertices_z_1 = face_vertices_cam_1[:, :, :, -1]

        # Extract the z-components of the face normals
        face_normals_z_1 = face_normals_1[:, :, -1]
        # This implementation is correct, but due to low mesh resolution, it does not work
        face_occlusion_indication = 1. * (face_normals_2[:, :, -1] < 0)
        face_occlusion_indication_features = face_occlusion_indication[..., None, None].repeat(1, 1, 3, 1)

        face_vertices_image_motion = face_vertices_image_2 - face_vertices_image_1  # Vertices are in [-1, 1] range

        features_for_rendering = torch.cat([face_vertices_image_motion,
                                            face_occlusion_indication_features,
                                            face_vertices_cam_2], dim=-1).float()

        ren_outputs_1, ren_mask_1, red_index_1 = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                       face_vertices_z_1,
                                                                                       face_vertices_image_1,
                                                                                       features_for_rendering,
                                                                                       face_normals_z_1,
                                                                                       sigmainv=self.renderer.sigmainv,
                                                                                       boxlen=0.02, knum=30,
                                                                                       multiplier=1000)

        theoretical_flow = ren_outputs_1[..., :2]

        theoretical_flow[..., 0] = theoretical_flow[..., 0] * 0.5
        theoretical_flow[..., 1] = -theoretical_flow[..., 1] * 0.5  # Correction for transform into image

        theoretical_flow = theoretical_flow.permute(0, 3, 1, 2).unsqueeze(0)  # torch.Size([1, N, 2, H, W])
        flow_render_mask = ren_mask_1.unsqueeze(1).unsqueeze(0)  # torch.Size([1, N, 1, H, W])
        occlusion_mask = ren_outputs_1[..., 2].detach().unsqueeze(1).unsqueeze(0)  # torch.Size([1, N, 1, H, W])

        return RenderedFlowResult(theoretical_flow, flow_render_mask, occlusion_mask)

    def get_batched_tensors_for_flow_rendering(self, batch_size, vertices_1):
        vertices_1_batch = vertices_1.repeat(batch_size, 1, 1)
        vertices_2_batch = vertices_1_batch
        obj_center_batch = self.obj_center.repeat(batch_size, 1)
        camera_rot_batch = self.camera_rot.repeat(batch_size, 1, 1)
        camera_trans_batch = self.camera_trans.repeat(batch_size, 1)

        return camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch

    def get_occlusion_mask_using_rendered_coordinates(self, rendered_pose1_with_pose2_coordinates,
                                                      rendered_pose2_with_pose2_coordinates, theoretical_flow):
        theoretical_flow_discrete = self.theoretical_flow_kaolin_to_image_warp(theoretical_flow)

        x_coords, x_coords_new, y_coords, y_coords_new = (
            self.get_original_and_warped_coordinates_from_flow(theoretical_flow_discrete))

        position_difference = (rendered_pose2_with_pose2_coordinates[0, y_coords_new, x_coords_new] -
                               rendered_pose1_with_pose2_coordinates[0, y_coords, x_coords])
        position_difference_norm = torch.linalg.vector_norm(position_difference, dim=-1)

        occlusion_mask = torch.zeros(1, 1, self.height, self.width).cuda()
        occlusion_mask[0, 0, y_coords, x_coords] = (position_difference_norm > 1e-1).float()

        return occlusion_mask

    def get_occlusion_mask_using_rendered_indices(self, rendered_pose1_with_pose2_indices,
                                                  rendered_pose2_with_pose2_indices, theoretical_flow):
        theoretical_flow_discrete = self.theoretical_flow_kaolin_to_image_warp(theoretical_flow)

        x_coords, x_coords_new, y_coords, y_coords_new = (
            self.get_original_and_warped_coordinates_from_flow(theoretical_flow_discrete))

        indices_eq_tensor = 1. * (rendered_pose2_with_pose2_indices == rendered_pose1_with_pose2_indices)

        occlusion_mask = torch.zeros(1, 1, self.height, self.width).cuda()
        occlusion_mask[0, 0, y_coords, x_coords] = indices_eq_tensor[0]

        return occlusion_mask

    def get_original_and_warped_coordinates_from_flow(self, theoretical_flow_discrete):
        x_coord_delta = theoretical_flow_discrete[..., 0]
        y_coord_delta = theoretical_flow_discrete[..., 1]
        x_coords, y_coords = torch.meshgrid(torch.arange(self.height), torch.arange(self.width))
        x_coords = x_coords.long().cuda()
        y_coords = y_coords.long().cuda()
        x_coords_new = torch.clamp(x_coords + x_coord_delta, 0, self.width).long()
        y_coord_new = torch.clamp(y_coords + y_coord_delta, 0, self.height).long()
        return x_coords, x_coords_new, y_coords, y_coord_new

    def theoretical_flow_kaolin_to_image_warp(self, theoretical_flow):
        theoretical_flow_discrete = theoretical_flow.clone()
        theoretical_flow_discrete[..., 0] *= self.height * 0.5
        theoretical_flow_discrete[..., 1] *= -self.width * 0.5
        theoretical_flow_discrete = theoretical_flow_discrete[0].long()
        return theoretical_flow_discrete

    def compute_theoretical_flow_using_rendered_vertices(self, rendering_result_frame_1: RenderingResult,
                                                         encoder_out_frame_2: EncoderResult,
                                                         encoder_out_frame_1: EncoderResult,
                                                         flow_arcs_indices,
                                                         ctx=None) -> RenderedFlowResult:

        rendered_vertices_frame_1 = rendering_result_frame_1.rendered_face_world_coords
        rendered_mask_frame_1 = rendering_result_frame_1.rendered_image_segmentation

        indices_pose_1_list = [frame_i_prev for frame_i_prev, _ in flow_arcs_indices]
        indices_pose_1 = torch.tensor(indices_pose_1_list, dtype=torch.long).cuda()
        rendered_vertices_frame_1_batched = torch.index_select(rendered_vertices_frame_1, 1, indices_pose_1)
        rendered_mask_frame_1_batched = torch.index_select(rendered_mask_frame_1, 1, indices_pose_1)

        batches = self.rotations_translations_batched(encoder_out_frame_1, encoder_out_frame_2, flow_arcs_indices)
        (rotation_matrix_1_batch, rotation_matrix_2_batch,
         translation_vector_1_batch, translation_vector_2_batch) = batches

        batch_size = translation_vector_1_batch.shape[0]
        vertices_1 = encoder_out_frame_1.vertices

        batched_tensors = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1)
        camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch = batched_tensors

        rendered_vertices_frame = rendered_vertices_frame_1_batched.permute(0, 1, 3, 4, 2)
        rendered_vertices_frame_norm = rendered_vertices_frame.norm(dim=-1)

        zero_vertices_positions = tuple(rendered_vertices_frame_norm.eq(0).nonzero().T)

        vertices_flattened = rendered_vertices_frame.flatten(start_dim=2, end_dim=-2)[0]

        vertices_1_nonzero = kaolin.render.camera.rotate_translate_points(vertices_flattened, rotation_matrix_1_batch,
                                                                          obj_center_batch)
        vertices_2_nonzero = kaolin.render.camera.rotate_translate_points(vertices_flattened, rotation_matrix_2_batch,
                                                                          obj_center_batch)

        vertices_1_nonzero = vertices_1_nonzero + translation_vector_1_batch.unsqueeze(1)
        vertices_2_nonzero = vertices_2_nonzero + translation_vector_2_batch.unsqueeze(1)

        vertices_1_camera = kaolin.render.camera.rotate_translate_points(vertices_1_nonzero, camera_rot_batch,
                                                                         camera_trans_batch)
        vertices_1_image = kaolin.render.camera.perspective_camera(vertices_1_camera, self.camera_proj)

        vertices_2_camera = kaolin.render.camera.rotate_translate_points(vertices_2_nonzero, camera_rot_batch,
                                                                         camera_trans_batch)
        vertices_2_image = kaolin.render.camera.perspective_camera(vertices_2_camera, self.camera_proj)

        vertices_flow = vertices_2_image - vertices_1_image

        if ctx is not None:
            ctx.x_world = rendered_vertices_frame_1_batched.detach().clone()

            vertices_1_camera_ren = vertices_1_camera.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_camera = vertices_1_camera_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

            vertices_1_image_ren = vertices_1_image.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_image = vertices_1_image_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

            vertices_2_image_ren = vertices_2_image.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_prime_image = vertices_2_image_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

        theoretical_flow = vertices_flow.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1])).unsqueeze(0)

        theoretical_flow_new = theoretical_flow.clone()  # Create a new tensor with the same values
        theoretical_flow_new[..., 0] = theoretical_flow[..., 0] * 0.5
        theoretical_flow_new[..., 1] = -theoretical_flow[..., 1] * 0.5  # Correction for transform into image
        theoretical_flow_new[zero_vertices_positions].zero_()
        theoretical_flow = theoretical_flow_new.permute(0, 1, 4, 2, 3)  # torch.Size([1, N, H, W, 2])

        flow_segmentation = rendered_mask_frame_1_batched

        # TODO implement mock occlusion as real occlusion
        mock_occlusion = torch.zeros(flow_segmentation.shape).cuda()

        flow_result = RenderedFlowResult(theoretical_flow, flow_segmentation, mock_occlusion)

        return flow_result

    def rendering_result_for_frame(self, encoder: Encoder, frame_i) -> RenderingResult:
        frames = [frame_i]
        encoder_result, _ = encoder.frames_and_flow_frames_inference(frames, frames)

        rendering_res = self.forward(encoder_result.translations, encoder_result.quaternions, encoder_result.vertices,
                                     encoder.face_features, encoder.texture_map)

        return rendering_res

    def render_flow_for_frame(self, encoder, flow_arc_source, flow_arc_target) -> SyntheticFlowObservation:
        keyframes = [flow_arc_source, flow_arc_target]
        flow_frames = [flow_arc_source, flow_arc_target]
        encoder_result, encoder_result_flow_frames = encoder.frames_and_flow_frames_inference(keyframes,
                                                                                              flow_frames)
        rendered_flow_res = self.compute_theoretical_flow(encoder_result, encoder_result_flow_frames,
                                                          flow_arcs_indices=[(0, 1)])

        synthetic_flow_cpu = SyntheticFlowObservation(
            observed_flow=rendered_flow_res.theoretical_flow,
            observed_flow_segmentation=rendered_flow_res.rendered_flow_segmentation,
            observed_flow_occlusion=rendered_flow_res.rendered_flow_occlusion,
            flow_source_frames=[flow_arc_source],
            flow_target_frames=[flow_arc_target],
        )

        return synthetic_flow_cpu

    @staticmethod
    def rotations_translations_batched(encoder_out_frame_1, encoder_out_frame_2, flow_arcs_indices):

        indices_pose_1_list = [frame_i_prev for frame_i_prev, _ in flow_arcs_indices]
        indices_pose_2_list = [frame_i for _, frame_i in flow_arcs_indices]
        # Convert lists to tensors
        indices_pose_1 = torch.tensor(indices_pose_1_list, dtype=torch.long).cuda()
        indices_pose_2 = torch.tensor(indices_pose_2_list, dtype=torch.long).cuda()
        # Batch gather translations
        translation_vector_1_batch = torch.index_select(encoder_out_frame_1.translations, 0, indices_pose_1)
        translation_vector_2_batch = torch.index_select(encoder_out_frame_2.translations, 0, indices_pose_2)
        # Batch convert quaternions to rotation matrices
        quaternion_batch_1 = torch.index_select(encoder_out_frame_1.quaternions, 0, indices_pose_1)
        quaternion_batch_2 = torch.index_select(encoder_out_frame_2.quaternions, 0, indices_pose_2)

        rotation_matrix_1_batch = quaternion_to_rotation_matrix(quaternion_batch_1).to(torch.float)
        rotation_matrix_2_batch = quaternion_to_rotation_matrix(quaternion_batch_2).to(torch.float)

        return rotation_matrix_1_batch, rotation_matrix_2_batch, translation_vector_1_batch, translation_vector_2_batch

    def render_mesh_with_dibr(self, face_features, rotation_matrix, translation_vector, unit_vertices) \
            -> MeshRenderResult:
        # Rotate and translate the vertices using the given rotation_matrix and translation_vector
        obj_center_batched = self.obj_center.repeat(rotation_matrix.shape[0], 1).unsqueeze(-1)
        vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, obj_center_batched)

        # Apply the translation to the vertices
        vertices = vertices + translation_vector.unsqueeze(1)

        # Prepare the vertices for rendering by computing their camera coordinates, image coordinates, and face normals
        camera_rot_batched = self.camera_rot.repeat(translation_vector.shape[0], 1, 1)
        camera_trans_batched = self.camera_trans.repeat(translation_vector.shape[0], 1)

        prepared_vertices = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices,
                                                                      faces=self.faces,
                                                                      camera_rot=camera_rot_batched,
                                                                      camera_trans=camera_trans_batched,
                                                                      camera_proj=self.camera_proj)

        vertices_world_coordinates = kaolin.ops.mesh.index_vertices_by_faces(unit_vertices, self.faces)

        face_vertices_cam, face_vertices_image, face_normals = prepared_vertices
        face_normals_feature = face_normals.unsqueeze(-1)

        # Extract the z-coordinates of the face vertices in camera space
        face_vertices_z = face_vertices_cam[:, :, :, -1]

        # Extract the z-components of the face normals
        face_normals_z = face_normals[:, :, -1]

        features_for_rendering = torch.cat((face_features,
                                            face_vertices_cam,
                                            face_vertices_image,
                                            vertices_world_coordinates,
                                            face_normals_feature), dim=-1)

        # Perform dibr rasterization
        ren_outputs, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                 face_vertices_z,
                                                                                 face_vertices_image,
                                                                                 features_for_rendering,
                                                                                 face_normals_z,
                                                                                 sigmainv=self.renderer.sigmainv,
                                                                                 boxlen=0.02, knum=30, multiplier=1000)

        # Extract ren_mesh_vertices_features and ren_mesh_vertices_coords from the combined output tensor
        split_tuple = torch.split(ren_outputs, [face_features.shape[-1],
                                                face_vertices_cam.shape[-1],
                                                face_vertices_image.shape[-1],
                                                vertices_world_coordinates.shape[-1],
                                                face_normals_feature.shape[-1]], dim=-1)

        (ren_mesh_vertices_features, ren_mesh_vertices_camera_coords, ren_mesh_vertices_image_coords,
         ren_mesh_vertices_world_coords, ren_face_normals_features) = split_tuple

        return MeshRenderResult(face_normals, face_vertices_cam, red_index, ren_mask,
                                ren_mesh_vertices_features,
                                ren_mesh_vertices_world_coords,
                                ren_mesh_vertices_camera_coords,
                                ren_mesh_vertices_image_coords,
                                ren_face_normals_features)

    def get_rgb_texture(self, translation, quaternion, unit_vertices, face_features, input_batch):
        tex = torch.zeros(1, 3, self.renderer.texture_size, self.renderer.texture_size)
        cnt = torch.zeros(self.renderer.texture_size, self.renderer.texture_size)
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi])

            rendering_result = self.render_mesh_with_dibr(face_features, rotation_matrix, translation_vector,
                                                          unit_vertices)

            coord = torch.round((1 - rendering_result.ren_mesh_vertices_features) * self.renderer.texture_size).to(int)
            coord[coord >= self.renderer.texture_size] = self.renderer.texture_size - 1
            coord[coord < 0] = 0
            xc = coord[0, :, :, 1].reshape([coord.shape[1] * coord.shape[2]])
            yc = (self.renderer.texture_size - 1 - coord[0, :, :, 0]).reshape([coord.shape[1] * coord.shape[2]])
            cr = input_batch[0, frmi, 0].reshape([coord.shape[1] * coord.shape[2]])
            cg = input_batch[0, frmi, 1].reshape([coord.shape[1] * coord.shape[2]])
            cb = input_batch[0, frmi, 2].reshape([coord.shape[1] * coord.shape[2]])
            for ki in range(xc.shape[0]):
                cnt[xc[ki], yc[ki]] = cnt[xc[ki], yc[ki]] + 1
                tex[0, 0, xc[ki], yc[ki]] = tex[0, 0, xc[ki], yc[ki]] + cr[ki]
                tex[0, 1, xc[ki], yc[ki]] = tex[0, 1, xc[ki], yc[ki]] + cg[ki]
                tex[0, 2, xc[ki], yc[ki]] = tex[0, 2, xc[ki], yc[ki]] + cb[ki]
        tex_final = tex / cnt[None, None]
        return tex_final


def infer_normalized_renderings(renderer: RenderingKaolin, encoder_face_features, encoder_result,
                                encoder_result_flow_frames, flow_arcs_indices, input_image_width, input_image_height) \
        -> Tuple[torch.Tensor, torch.Tensor, RenderedFlowResult]:
    rendering_result = renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                        encoder_result.vertices, encoder_face_features, encoder_result.texture_maps,
                                        encoder_result.lights)

    rendering = rendering_result.rendered_image
    rendering_mask = rendering_result.rendered_image_segmentation

    flow_result = renderer.compute_theoretical_flow(encoder_result, encoder_result_flow_frames, flow_arcs_indices)

    theoretical_flow, rendered_flow_segmentation, occlusion_masks = flow_result

    # Renormalization compensating for the fact that we render into bounding box that is smaller than the actual image
    normalized_theoretical_flow = normalize_rendered_flows(theoretical_flow, renderer.width, renderer.height,
                                                           input_image_width, input_image_height)
    flow_result = flow_result._replace(theoretical_flow=normalized_theoretical_flow)

    return rendering, rendering_mask, flow_result


def pinhole_intrinsics_to_tensor(intrinsics: kaolin.render.camera.PinholeIntrinsics) -> torch.Tensor:
    intrinsics_tensor = torch.Tensor([[intrinsics.focal_x, 0., intrinsics.x0],
                                      [0., intrinsics.focal_y, intrinsics.y0],
                                      [0., 0., 1.]])
    if len(intrinsics_tensor.shape) == 2:
        intrinsics_tensor = intrinsics_tensor

    return intrinsics_tensor


def get_Se3_obj2cam_from_kaolin_params(camera_trans: torch.Tensor, camera_up: torch.Tensor,
                                       obj_center: torch.Tensor) -> Se3:
    R_obj_to_cam, t_obj_to_cam = (
        kaolin.render.camera.generate_rotate_translate_matrices(camera_position=camera_trans,
                                                                camera_up_direction=camera_up,
                                                                look_at=obj_center))

    Se3_obj_to_cam = Se3(Quaternion.from_matrix(R_obj_to_cam), t_obj_to_cam)
    return Se3_obj_to_cam
