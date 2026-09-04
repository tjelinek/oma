import itertools
import logging
from pathlib import Path
from typing import List, Optional

import imageio
import networkx as nx
import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb
import torch
import trimesh
from PIL import Image
from kornia.geometry import Se3
from matplotlib import pyplot as plt
from data_structures.data_graph import DataGraph
from data_structures.rerun_annotations import RerunAnnotations
from configs.glopose_config import GloPoseConfig
from eval.eval_point_cloud import _load_ply_without_edges
from onboarding.colmap_utils import world2cam_from_reconstruction
from utils.data_utils import load_texture, load_mesh_using_trimesh
from utils.general import normalize_vertices, extract_intrinsics_from_tensor
from visualizations.rerun_utils import (init_rerun_recording, register_matching_series_lines,
                                        visualize_certainty_map, log_matching_correspondences,
                                        log_colmap_point_projections)
from utils.image_utils import overlay_mask

logger = logging.getLogger(__name__)


def build_onboarding_blueprint(config: GloPoseConfig) -> rrb.Blueprint:
    """Build the standard onboarding rerun blueprint. Reusable for both normal and merged runs."""
    if config.onboarding.frame_filter in ('dense_matching', 'RANSAC', 'depth'):
        reliability_plot_name = ("Depth Geometric-Inlier Reliability"
                                 if config.onboarding.frame_filter == 'depth'
                                 else f"{config.onboarding.frame_filter} Matching Reliability")
        match_reliability_statistics = rrb.TimeSeriesView(
            name=reliability_plot_name,
            origin=RerunAnnotations.matching_reliability_plot,
            axis_y=rrb.ScalarAxis(range=(0.0, 1.2), zoom_lock=True),
            plot_legend=rrb.PlotLegend(visible=True))
    else:
        max_range = 3.0 * config.onboarding.sift_filter_good_to_add_matches
        match_reliability_statistics = rrb.TimeSeriesView(
            name="SIFT Num of Matches",
            origin=RerunAnnotations.matching_reliability_plot,
            axis_y=rrb.ScalarAxis(range=(0.0, max_range), zoom_lock=False),
            plot_legend=rrb.PlotLegend(visible=True))

    blueprint = rrb.Blueprint(
        rrb.Tabs(
            contents=[
                rrb.Tabs(
                    contents=[
                        rrb.Vertical(
                            contents=[
                                rrb.Horizontal(
                                    contents=[
                                        rrb.Spatial2DView(name="Template Image Current",
                                                          origin=RerunAnnotations.template_image),
                                        rrb.Spatial2DView(name="Observed Image",
                                                          origin=RerunAnnotations.observed_image),
                                    ],
                                    name='Observed Images'
                                ),
                                rrb.Grid(
                                    contents=[
                                        rrb.Spatial2DView(name=f"Keyframe {i}",
                                                          origin=f'{RerunAnnotations.keyframe_images}/{i}')
                                        for i in range(27)
                                    ],
                                    grid_columns=9,
                                    name='Templates'
                                ),
                                rrb.GraphView(
                                    name='View Graph',
                                    origin=RerunAnnotations.view_graph,
                                ),
                            ],
                            row_shares=[0.3, 0.5, 0.2],
                            name='Keyframe Images'
                        ),
                        rrb.GraphView(
                            name='Keyframe Graph',
                            origin=RerunAnnotations.keyframe_graph,
                        ),
                        rrb.GraphView(
                            name='View Graph',
                            origin=RerunAnnotations.view_graph,
                        ),
                    ],
                    name='Keyframes'
                ),
                rrb.Tabs(
                    contents=[
                        rrb.Spatial3DView(
                            origin=RerunAnnotations.colmap_visualization,
                            name='COLMAP',
                            # Sparse view: hide the densified cloud so the two
                            # tabs show sparse vs densified side by side.
                            contents=[
                                '+ $origin/**',
                                f'- {RerunAnnotations.colmap_pointcloud_densified}',
                            ],
                            background=[255, 255, 255]
                        ),
                        rrb.Spatial3DView(
                            origin=RerunAnnotations.colmap_visualization,
                            name='Densified',
                            contents=[
                                '+ $origin/**',
                                f'- {RerunAnnotations.colmap_pointcloud}',
                            ],
                            background=[255, 255, 255]
                        ),
                        rrb.Spatial3DView(
                            origin=RerunAnnotations.space_visualization,
                            name='3D Ground Truth',
                            background=[255, 255, 255]
                        ),
                        rrb.Grid(
                            contents=[
                                rrb.Spatial2DView(
                                    name=f"Keyframe {i}",
                                    origin=f'{RerunAnnotations.colmap_point_projections}/{i}'
                                )
                                for i in range(1, 28)
                            ],
                            grid_columns=9,
                            name='Point Projections'
                        ),
                        rrb.Horizontal(
                            contents=[
                                rrb.GraphView(
                                    name='COLMAP Co-visibility',
                                    origin=RerunAnnotations.colmap_covisibility_graph,
                                ),
                                rrb.GraphView(
                                    name='Initial Viewgraph',
                                    origin=RerunAnnotations.initial_viewgraph,
                                ),
                            ],
                            name='Graph Comparison'
                        ),
                    ],
                    name='3D Space'
                ),
                rrb.Grid(
                    contents=[
                        rrb.TimeSeriesView(name="Pose Estimation (w.o. flow computation)",
                                           origin=RerunAnnotations.pose_estimation_timing),
                    ],
                    grid_columns=2,
                    name='Timings'
                ),
                rrb.Tabs(
                    contents=[
                        rrb.Vertical(
                            contents=[
                                rrb.Horizontal(
                                    contents=[
                                        rrb.Spatial2DView(
                                            name=f"{config.onboarding.filter_matcher} Matches High Certainty",
                                            origin=RerunAnnotations.matches_high_certainty),
                                        rrb.Spatial2DView(
                                            name=f"{config.onboarding.filter_matcher} Matches Low Certainty",
                                            origin=RerunAnnotations.matches_low_certainty),
                                        *([rrb.Spatial2DView(
                                            name=f"{config.onboarding.filter_matcher} Matching Certainty",
                                            origin=RerunAnnotations.matching_certainty)]
                                          if config.onboarding.frame_filter in ('dense_matching', 'RANSAC', 'depth') else [])
                                    ],
                                    name='Matching'
                                ),
                                match_reliability_statistics,
                            ],
                            row_shares=[0.8, 0.2],
                            name='Matching'
                        ),
                        *([rrb.Vertical(
                            contents=[
                                rrb.Horizontal(
                                    contents=[
                                        rrb.Spatial2DView(
                                            name=f"{config.onboarding.filter_matcher} Matches High Certainty",
                                            origin=RerunAnnotations.matches_high_certainty_matchable),
                                        rrb.Spatial2DView(
                                            name=f"{config.onboarding.filter_matcher} Matches Low Certainty",
                                            origin=RerunAnnotations.matches_low_certainty_matchable),
                                        rrb.Spatial2DView(name="Template",
                                                          origin=RerunAnnotations.matchability)
                                    ],
                                    name='Matching'
                                ),
                                rrb.TimeSeriesView(name="Matchable Area Share",
                                                   origin=RerunAnnotations.matching_matchability_plot,
                                                   axis_y=rrb.ScalarAxis(range=(0.0, 1.2), zoom_lock=True),
                                                   plot_legend=rrb.PlotLegend(visible=True)),
                                rrb.TimeSeriesView(name=f"{config.onboarding.filter_matcher} Min Certainty",
                                                   origin=RerunAnnotations.matching_min_roma_certainty_plot,
                                                   axis_y=rrb.ScalarAxis(range=(0.0, 1.2), zoom_lock=True),
                                                   plot_legend=rrb.PlotLegend(visible=True)),
                            ],
                            row_shares=[4, 1, 1],
                            name='Matchability'
                        )] if config.onboarding.frame_filter == 'dense_matching' and config.onboarding.matchability_based_reliability
                          else []),
                        *([rrb.TimeSeriesView(
                            name="Depth: Estimated vs GT Relative Pose Error",
                            origin=RerunAnnotations.depth_pose_error_plot,
                            axis_y=rrb.ScalarAxis(range=(0.0, 180.0), zoom_lock=False),
                            plot_legend=rrb.PlotLegend(visible=True))]
                          if config.onboarding.frame_filter == 'depth' else []),
                    ],
                    name='Matching'
                ),
                rrb.Tabs(
                    contents=[
                        rrb.Grid(
                            contents=[
                                rrb.TimeSeriesView(name="RANSAC - Frontview",
                                                   origin=RerunAnnotations.ransac_stats),
                                rrb.TimeSeriesView(name="Pose - Rotation",
                                                   origin=RerunAnnotations.obj_rot_1st_to_last),
                                rrb.TimeSeriesView(name="Pose - Translation",
                                                   origin=RerunAnnotations.obj_tran_1st_to_last),
                            ],
                            grid_columns=2,
                            name='Epipolar'
                        ),
                        rrb.Grid(
                            contents=[
                                rrb.TimeSeriesView(name="Camera Rotation Ref -> Last",
                                                   origin=RerunAnnotations.cam_rot_ref_to_last),
                                rrb.TimeSeriesView(name="Camera Translation Ref -> Last",
                                                   origin=RerunAnnotations.cam_tran_ref_to_last),
                                rrb.TimeSeriesView(name="Object Rotation Ref -> Last",
                                                   origin=RerunAnnotations.obj_rot_ref_to_last),
                                rrb.TimeSeriesView(name="Object Translation Ref -> Last",
                                                   origin=RerunAnnotations.obj_tran_ref_to_last),
                            ],
                            grid_columns=2,
                            name='Pose'
                        ),
                    ],
                    name='Pose'
                ),
            # Tab 6: Model Merging (used by separate merge strategy)
            rrb.Tabs(
                rrb.Spatial3DView(
                    name='Before Alignment',
                    origin=RerunAnnotations.merge_before,
                    background=[255, 255, 255],
                ),
                rrb.Spatial3DView(
                    name='After Procrustes',
                    origin=RerunAnnotations.merge_after_procrustes,
                    background=[255, 255, 255],
                ),
                rrb.Spatial3DView(
                    name='Before ICP',
                    origin=RerunAnnotations.merge_before_icp,
                    background=[255, 255, 255],
                ),
                rrb.Spatial3DView(
                    name='After ICP',
                    origin=RerunAnnotations.merge_after_icp,
                    background=[255, 255, 255],
                ),
                rrb.Spatial3DView(
                    name='After Alignment (Merged)',
                    origin=RerunAnnotations.merge_after,
                    background=[255, 255, 255],
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        *[rrb.Spatial2DView(name=f'Match Pair {i}',
                                            origin=f'{RerunAnnotations.merge_match_pairs}/{i}')
                          for i in range(5)],
                    ),
                    rrb.TextLogView(name='Match Info', origin=RerunAnnotations.merge_match_info),
                    row_shares=[0.8, 0.2],
                    name='Match Pair Images',
                ),
                name='Model Merging',
            ),
            ],
            name=f'Results - {config.run.sequence}'
        )
    )
    return blueprint


def log_reconstruction_to_rerun(reconstruction: pycolmap.Reconstruction, gt_model_path: Path | None = None,
                                gt_Se3_world2cam: dict | None = None):
    """Log a COLMAP reconstruction (pointcloud + cameras + GT track) to rerun. Standalone, no WriteResults needed."""
    import io
    from matplotlib import pyplot as plt

    # 3D point cloud
    if reconstruction.points3D:
        points_3d_coords = np.stack([p.xyz for p in reconstruction.points3D.values()], axis=0)
        points_3d_colors = np.stack([p.color for p in reconstruction.points3D.values()], axis=0)
        rr.log(RerunAnnotations.colmap_pointcloud,
               rr.Points3D(points_3d_coords, colors=points_3d_colors), static=True)
        rr.log(RerunAnnotations.space_reconstruction_pointcloud,
               rr.Points3D(points_3d_coords, colors=points_3d_colors, radii=0.001), static=True)

    # Camera poses
    for image_id, image in sorted(reconstruction.images.items()):
        cam_from_world = image.cam_from_world()
        pred_t = torch.tensor(cam_from_world.translation)
        pred_q_xyzw = torch.tensor(cam_from_world.rotation.quat)

        camera = reconstruction.cameras[image.camera_id]
        K = camera.calibration_matrix()
        entity = f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}'

        rr.log(entity,
               rr.Transform3D(translation=pred_t,
                              rotation=rr.Quaternion(xyzw=pred_q_xyzw),
                              from_parent=True),
               static=True)
        rr.log(entity,
               rr.Pinhole(resolution=[camera.width, camera.height],
                          focal_length=[K[0, 0], K[1, 1]],
                          principal_point=[K[0, 2], K[1, 2]]),
               static=True)

    # GT camera track
    if gt_Se3_world2cam is not None and len(gt_Se3_world2cam) >= 2:
        gt_centers = np.stack([
            gt_Se3_world2cam[i].inverse().translation.numpy(force=True)
            for i in sorted(gt_Se3_world2cam.keys())
        ])
        n_gt = len(gt_centers)
        cmap_gt = plt.get_cmap('Reds')
        gradient = np.linspace(1., 0.5, n_gt)
        colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_gt)]) * 255).astype(np.uint8)
        strips_gt = np.stack([gt_centers[:-1], gt_centers[1:]], axis=1)
        object_size = 1.0
        if reconstruction.points3D:
            object_size = np.max(np.linalg.norm(points_3d_coords - np.mean(points_3d_coords, axis=0), axis=1))
        strips_radii = [0.005 * object_size] * n_gt
        rr.log(RerunAnnotations.colmap_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt, colors=colors_gt, radii=strips_radii), static=True)

    # GT mesh
    if gt_model_path is not None and gt_model_path.exists():
        try:
            if gt_model_path.suffix.lower() == '.ply':
                cleaned = _load_ply_without_edges(gt_model_path)
                mesh = trimesh.load(io.BytesIO(cleaned), file_type='ply', force='mesh')
            else:
                mesh = trimesh.load(gt_model_path, force='mesh')
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            mesh3d_kwargs = dict(vertex_positions=vertices, triangle_indices=faces)
            if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.material is not None:
                uv = mesh.visual.uv
                if uv is not None:
                    vertex_texcoords = np.asarray(uv, dtype=np.float32).copy()
                    vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]
                    mesh3d_kwargs['vertex_texcoords'] = vertex_texcoords
                    material = mesh.visual.material
                    texture_image = None
                    if hasattr(material, 'image') and material.image is not None:
                        texture_image = np.asarray(material.image.convert('RGB'), dtype=np.uint8)
                    elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                        texture_image = np.asarray(material.baseColorTexture.convert('RGB'), dtype=np.uint8)
                    if texture_image is not None:
                        mesh3d_kwargs['albedo_texture'] = texture_image
                    else:
                        mesh3d_kwargs.pop('vertex_texcoords', None)
                        mesh3d_kwargs['vertex_colors'] = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3]
                else:
                    mesh3d_kwargs['vertex_colors'] = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3]
            elif isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                mesh3d_kwargs['vertex_colors'] = np.asarray(mesh.visual.vertex_colors)[:, :3]
            else:
                mesh3d_kwargs['vertex_colors'] = np.full((len(vertices), 3), 180, dtype=np.uint8)
            rr.log(RerunAnnotations.space_gt_mesh, rr.Mesh3D(**mesh3d_kwargs), static=True)
        except Exception as e:
            logger.warning("Failed to load GT mesh from %s: %s", gt_model_path, e)



def _estimate_object_radius(reconstruction: pycolmap.Reconstruction) -> float:
    """Estimate a reasonable point radius from the reconstruction extent."""
    if not reconstruction.points3D:
        return 0.001
    pts = np.stack([p.xyz for p in reconstruction.points3D.values()], axis=0)
    extent = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
    return max(extent * 0.002, 0.001)


def _log_reconstruction_pointcloud(entity: str, reconstruction: pycolmap.Reconstruction,
                                    color: tuple, radius: float | None = None):
    """Log a reconstruction's point cloud to rerun with a uniform color."""
    if not reconstruction.points3D:
        return
    pts = np.stack([p.xyz for p in reconstruction.points3D.values()], axis=0)
    colors = np.full((len(pts), 3), color, dtype=np.uint8)
    if radius is None:
        radius = _estimate_object_radius(reconstruction)
    rr.log(entity, rr.Points3D(pts, colors=colors, radii=radius), static=True)


def _log_reconstruction_cameras(entity_prefix: str, reconstruction: pycolmap.Reconstruction,
                                 color: tuple):
    """Log camera poses from a reconstruction as colored line strips (camera track)."""
    centers = []
    for image_id in sorted(reconstruction.images.keys()):
        image = reconstruction.images[image_id]
        cam_center = image.cam_from_world().inverse().translation
        centers.append(cam_center)
    if len(centers) < 2:
        return
    centers = np.array(centers)
    strips = np.stack([centers[:-1], centers[1:]], axis=1)
    rr.log(entity_prefix,
           rr.LineStrips3D(strips=strips, colors=[color] * len(strips), radii=0.002),
           static=True)
    rr.log(f'{entity_prefix}/positions',
           rr.Points3D(centers, colors=[color] * len(centers), radii=0.005),
           static=True)


def log_merge_to_rerun(rec_target: pycolmap.Reconstruction, rec_source: pycolmap.Reconstruction,
                        merged_rec: pycolmap.Reconstruction, align_info: dict,
                        target_images_dir: Path, source_images_dir: Path,
                        target_segs_dir: Path, source_segs_dir: Path,
                        gt_model_path: Path | None = None,
                        gt_Se3_world2cam: dict | None = None):
    """Log merge visualization data to rerun.

    Shows:
    - Target (down) point cloud in blue, source (up) in red — before alignment
    - Merged point cloud with original colors
    - Camera tracks for both reconstructions
    - GT mesh and GT camera track
    - Source images used for matching (from align_info)
    """
    import io

    RA = RerunAnnotations

    def _log_merge_stage(prefix: str, rec_fixed: pycolmap.Reconstruction,
                         rec_moved: pycolmap.Reconstruction | None):
        """Log a blue/red point cloud pair + GT mesh/track to a given entity prefix."""
        _log_reconstruction_pointcloud(f'{prefix}/target_pointcloud', rec_fixed, color=(70, 130, 255))
        _log_reconstruction_cameras(f'{prefix}/target_cameras', rec_fixed, color=(70, 130, 255))
        if rec_moved is not None:
            _log_reconstruction_pointcloud(f'{prefix}/source_pointcloud', rec_moved, color=(255, 70, 70))
            _log_reconstruction_cameras(f'{prefix}/source_cameras', rec_moved, color=(255, 70, 70))
        if gt_mesh_kwargs is not None:
            rr.log(f'{prefix}/gt_mesh', rr.Mesh3D(**gt_mesh_kwargs), static=True)
        if gt_strips is not None:
            rr.log(f'{prefix}/gt_camera_track',
                   rr.LineStrips3D(strips=gt_strips, colors=[(200, 50, 50)] * len(gt_strips), radii=0.003),
                   static=True)

    # Pre-load GT mesh once
    gt_mesh_kwargs = None
    if gt_model_path is not None and gt_model_path.exists():
        try:
            if gt_model_path.suffix.lower() == '.ply':
                cleaned = _load_ply_without_edges(gt_model_path)
                mesh = trimesh.load(io.BytesIO(cleaned), file_type='ply', force='mesh')
            else:
                mesh = trimesh.load(gt_model_path, force='mesh')
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            gt_mesh_kwargs = dict(vertex_positions=vertices, triangle_indices=faces,
                                  vertex_colors=np.full((len(vertices), 3), 180, dtype=np.uint8))
        except Exception as e:
            logger.warning("Failed to load GT mesh for merge viz: %s", e)

    # Pre-compute GT camera track
    gt_strips = None
    if gt_Se3_world2cam is not None and len(gt_Se3_world2cam) >= 2:
        gt_centers = np.stack([
            gt_Se3_world2cam[i].inverse().translation.numpy(force=True)
            for i in sorted(gt_Se3_world2cam.keys())
        ])
        gt_strips = np.stack([gt_centers[:-1], gt_centers[1:]], axis=1)

    # Before alignment: unmodified target (down) and source (up)
    _log_merge_stage(RA.merge_before, rec_target, rec_source)

    # After Procrustes (None if Procrustes was not applied)
    rec_after_procrustes = align_info.get('rec_after_procrustes')
    _log_merge_stage(RA.merge_after_procrustes, rec_target, rec_after_procrustes)

    # Before ICP (= after Procrustes if applied, or unmodified source if Procrustes failed)
    rec_before_icp = align_info.get('rec_before_icp')
    _log_merge_stage(RA.merge_before_icp, rec_target, rec_before_icp)

    # After ICP
    rec_after_icp = align_info.get('rec_after_icp')
    _log_merge_stage(RA.merge_after_icp, rec_target, rec_after_icp)

    # After alignment: final merged with original colors
    if merged_rec.points3D:
        pts = np.stack([p.xyz for p in merged_rec.points3D.values()], axis=0)
        colors = np.stack([p.color for p in merged_rec.points3D.values()], axis=0)
        radius = _estimate_object_radius(merged_rec)
        rr.log(RA.merge_after_pointcloud, rr.Points3D(pts, colors=colors, radii=radius), static=True)
    _log_reconstruction_cameras(RA.merge_after_cameras, merged_rec, color=(70, 200, 70))
    if gt_mesh_kwargs is not None:
        rr.log(RA.merge_after_gt_mesh, rr.Mesh3D(**gt_mesh_kwargs), static=True)
    if gt_strips is not None:
        rr.log(RA.merge_after_gt_camera_track,
               rr.LineStrips3D(strips=gt_strips, colors=[(200, 50, 50)] * len(gt_strips), radii=0.003),
               static=True)

    # Log match pair images with correspondences (if available in align_info)
    match_pairs = align_info.get('match_pairs', [])
    for i, pair in enumerate(match_pairs[:5]):
        src_stem = Path(pair['src_name']).stem
        tgt_stem = Path(pair['tgt_name']).stem
        src_img_path = source_images_dir / f'{src_stem}_image.png'
        tgt_img_path = target_images_dir / f'{tgt_stem}_image.png'
        if not src_img_path.exists() or not tgt_img_path.exists():
            continue

        src_img = np.asarray(Image.open(src_img_path))
        tgt_img = np.asarray(Image.open(tgt_img_path))
        h1, w1 = src_img.shape[:2]

        # Overlay segmentation masks (darken background like in onboarding)
        src_seg_path = source_segs_dir / f'{src_stem}_seg.png'
        tgt_seg_path = target_segs_dir / f'{tgt_stem}_seg.png'
        if src_seg_path.exists():
            src_seg = np.asarray(Image.open(src_seg_path).convert('L')).astype(np.float32) / 255.0
            src_img = overlay_mask(src_img, ~src_seg.astype(bool), alpha=0.7, color=(0, 0, 0))
        if tgt_seg_path.exists():
            tgt_seg = np.asarray(Image.open(tgt_seg_path).convert('L')).astype(np.float32) / 255.0
            tgt_img = overlay_mask(tgt_img, ~tgt_seg.astype(bool), alpha=0.7, color=(0, 0, 0))

        # Stack images vertically: source on top, target on bottom
        combined = np.concatenate([src_img, tgt_img], axis=0)
        pair_entity = f'{RA.merge_match_pairs}/{i}'
        rr.log(pair_entity, rr.Image(combined), static=True)

        # Log info text to shared text log panel (each at a different time so all are visible)
        reliability = pair.get('reliability', 0)
        threshold = pair.get('reliability_threshold', 0)
        passes = "PASS" if reliability >= threshold else "FAIL"
        info_text = (f"[Pair {i}] src={pair['src_name']} -> tgt={pair['tgt_name']}  |  "
                     f"matches={pair['num_matches']}  "
                     f"med_cert={pair.get('median_certainty', 0):.3f}  "
                     f"reliability={reliability:.3f} ({passes}, thr={threshold:.3f})  "
                     f"3D_corr={pair.get('num_correspondences', 0)}")
        rr.set_time('match_pair', sequence=i)
        rr.log(RA.merge_match_info, rr.TextLog(info_text))

        # Draw correspondences as line strips from match_pts if available
        match_pts = pair.get('match_pts')
        if match_pts is not None:
            src_2d, tgt_2d = match_pts
            # Offset target points by source image height
            tgt_2d_shifted = tgt_2d.copy()
            tgt_2d_shifted[:, 1] += h1
            # Subsample for visualization (max 200 lines)
            n = len(src_2d)
            step = max(1, n // 200)
            src_sub = src_2d[::step]
            tgt_sub = tgt_2d_shifted[::step]
            strips = np.stack([src_sub, tgt_sub], axis=1).astype(np.float32)
            rr.log(f'{pair_entity}/correspondences',
                   rr.LineStrips2D(strips, colors=[(0, 255, 0)] * len(strips), radii=0.5),
                   static=True)


class WriteResults:

    def __init__(self, write_folder, tracking_config: GloPoseConfig, data_graph: DataGraph):

        self.data_graph: DataGraph = data_graph

        self.logged_templates_3d_space: List = list()
        self.logged_keyframe_graph: nx.DiGraph = nx.DiGraph()

        self.config: GloPoseConfig = tracking_config

        self.write_folder = Path(write_folder)

        self.observations_path = self.write_folder / Path('images')
        self.segmentation_path = self.write_folder / Path('segments')
        self.ransac_path = self.write_folder / Path('ransac')
        self.exported_mesh_path = self.write_folder / Path('3d_model')

        self.init_directories()

        self.template_fields: List[str] = []

        self.rerun_init()

    def init_directories(self):
        if not self.config.visualization.write_to_rerun:
            self.observations_path.mkdir(exist_ok=True, parents=True)
            self.segmentation_path.mkdir(exist_ok=True, parents=True)
            self.ransac_path.mkdir(exist_ok=True, parents=True)
            self.exported_mesh_path.mkdir(exist_ok=True, parents=True)
            (self.write_folder / 'templates').mkdir(exist_ok=True, parents=True)

    def rerun_init(self):
        rerun_file = (self.write_folder /
                      f'rerun_{self.config.run.experiment_name}_{self.config.run.sequence}_{self.config.run.special_hash}.rrd')

        self.template_fields = {
            RerunAnnotations.chained_pose_polar_template,
            RerunAnnotations.chained_pose_long_flow_template,
            RerunAnnotations.chained_pose_short_flow_template,

            RerunAnnotations.cam_delta_r_short_flow_template,
            RerunAnnotations.cam_delta_t_short_flow_template,
            RerunAnnotations.cam_delta_r_long_flow_template,
            RerunAnnotations.cam_delta_t_long_flow_template,

            RerunAnnotations.long_short_chain_diff_template,

            RerunAnnotations.cam_rot_ref_to_last_template,
            RerunAnnotations.cam_tran_ref_to_last_template,
            RerunAnnotations.obj_rot_ref_to_last_template,
            RerunAnnotations.obj_tran_ref_to_last_template,

            RerunAnnotations.translation_scale
        }

        blueprint = build_onboarding_blueprint(self.config)

        axes_colors = {
            'x': (0, 127, 0),
            'y': (0, 51, 102),
            'z': (102, 0, 102),
        }

        gt_axes_colors = {
            'x': (0, 255, 0),
            'y': (102, 178, 255),
            'z': (255, 155, 255),
        }

        rerun_name = f'{self.config.run.sequence}-{self.config.run.experiment_name}-{self.config.run.special_hash}'
        init_rerun_recording(rerun_name, rerun_file, blueprint)

        if self.config.onboarding.frame_filter in ('dense_matching', 'RANSAC'):
            register_matching_series_lines()
        elif self.config.onboarding.frame_filter == 'SIFT':
            rr.log(RerunAnnotations.min_matches_sift,
                   rr.SeriesLines(colors=[255, 0, 0], names="min matches"), static=True)
            rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
                   rr.SeriesLines(colors=[0, 255, 0], names="good to add matches"), static=True)
            rr.log(RerunAnnotations.matches_sift,
                   rr.SeriesLines(colors=[0, 0, 255], names="matches"), static=True)

        annotations = set()
        for axis, c in axes_colors.items():
            annotations |= set(map(
                lambda annotation: (annotation, c),
                [
                    RerunAnnotations.obj_tran_1st_to_last_axes[axis],
                    RerunAnnotations.obj_tran_ref_to_last_axes[axis],
                    RerunAnnotations.cam_tran_ref_to_last_axes[axis],
                    RerunAnnotations.obj_rot_1st_to_last_axes[axis],
                    RerunAnnotations.obj_rot_ref_to_last_axes[axis],
                    RerunAnnotations.cam_rot_ref_to_last_axes[axis],
                    RerunAnnotations.chained_pose_long_flow_axes[axis],
                    RerunAnnotations.chained_pose_short_flow_axes[axis],
                    RerunAnnotations.translation_scale_gt_axes[axis],
                ]
            ))

        for axis, c in gt_axes_colors.items():
            annotations |= set(map(
                lambda annotation: (annotation, c),
                [
                    RerunAnnotations.obj_rot_1st_to_last_gt_axes[axis],
                    RerunAnnotations.obj_rot_ref_to_last_gt_axes[axis],
                    RerunAnnotations.cam_rot_ref_to_last_gt_axes[axis],
                    RerunAnnotations.obj_tran_1st_to_last_gt_axes[axis],
                    RerunAnnotations.obj_tran_ref_to_last_gt_axes[axis],
                    RerunAnnotations.cam_tran_ref_to_last_gt_axes[axis],
                ]
            ))

            for rerun_annotation, color in annotations:
                rr.log(rerun_annotation, rr.SeriesLines(colors=color,
                                                       names=rerun_annotation.split('/')[-1]), static=True)

        for template_annotation in self.template_fields:
            rr.log(template_annotation,
                   rr.SeriesPoints(
                       colors=[255, 0, 0],
                       names="new template",
                       markers="circle",
                       marker_sizes=4,
                   ),
                   static=True)

        rr.log(RerunAnnotations.observed_image_segmentation,
               rr.AnnotationContext([(1, "blue", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotations.template_image_segmentation,
               rr.AnnotationContext([(1, "blue", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)

    def visualize_keyframes(self, frame_i: int, keyframe_graph: nx.Graph):
        rr.set_time("frame", sequence=frame_i)

        kfs = set(keyframe_graph.nodes)
        not_logged_keyframes = kfs - set(self.logged_keyframe_graph.nodes)
        for kf_idx in sorted(not_logged_keyframes):
            keyframe_node = self.data_graph.get_frame_data(kf_idx)
            template = keyframe_node.frame_observation.observed_image[0].permute(1, 2, 0).detach().cpu()

            annotation = f'{RerunAnnotations.keyframe_images}/{len(self.logged_keyframe_graph.nodes)}'
            template_path = self.write_folder / 'templates' / f'{len(keyframe_graph.nodes)}'

            self.log_image(frame_i, template, annotation, template_path)

            # Matchability logging
            if self.config.onboarding.matchability_based_reliability:
                matchability_mask = keyframe_node.matchability_mask
                matchability_image = (~matchability_mask.unsqueeze(0).permute(1, 2, 0)).to(torch.float).numpy(
                    force=True)
                template = (template.numpy(force=True) * 255.0).astype(np.uint8)
                matchability_image_overlay = overlay_mask(template, matchability_image, 1.0, color=(0, 0, 0))
                rr_matchability_image = rr.Image(template).compress(jpeg_quality=self.config.visualization.jpeg_quality)
                rr.log(RerunAnnotations.matchability, rr_matchability_image)

            self.logged_keyframe_graph.add_node(kf_idx)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphNodes(node_ids=list(keyframe_graph.nodes),
                                                              labels=[str(kf) for kf in keyframe_graph.nodes]))

        # not_logged_keyframe_edges = set(keyframe_graph.edges) - set(self.logged_keyframe_graph.edges)
        # self.logged_keyframe_graph.add_edges_from(not_logged_keyframe_edges)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphEdges(edges=[(u, v) for (u, v) in keyframe_graph.edges]))

    def visualize_pose_graph(self, frame_i: int, keyframe_graph: nx.Graph):
        rr.set_time("frame", sequence=frame_i)

        # Create a directed graph from the pose graph
        pose_graph = nx.DiGraph()
        pose_graph.add_nodes_from(self.data_graph.G.nodes)
        # pose_graph.add_edges_from((u, v) for (u, v) in self.data_graph.G.edges
        #                           if self.data_graph.get_edge_observations(u, v).is_match_reliable)
        pose_graph.add_edges_from((n, self.data_graph.get_frame_data(n).matching_source_keyframe)
                                  for n in self.data_graph.G.nodes)
        pose_graph.remove_edges_from((n, n) for n in self.data_graph.G.nodes)

        white_node = [255, 255, 255]
        red_node = [255, 0, 0]

        kfs = set(keyframe_graph.nodes)
        all_nodes = list(kfs)
        node_labels = {kf: str(kf) for kf in kfs}

        for kf in kfs:
            neighbors = sorted({e[0] for e in pose_graph.in_edges(kf)} - set(kfs))
            pose_graph.remove_edges_from((kf, n) for n in neighbors if n not in kfs)
            pose_graph.remove_edges_from((n, kf) for n in neighbors if n not in kfs)

            for k, g in itertools.groupby(enumerate(neighbors),
                                          key=lambda t: t[1] - t[0]):
                g = list(g)
                start = g[0][1]
                end = g[-1][1]
                pose_graph.add_edge(start, kf)
                all_nodes.append(start)
                if start != end:
                    node_labels[start] = f'{start}..{end}'
                else:
                    node_labels[start] = str(start)

        all_nodes_sorted = sorted(all_nodes)
        # Define y-axis positions for keyframes and ordinary frames
        positions = [
            (i * 100, 200.0) if n in kfs else (i * 100, 0.0) for i, n in enumerate(all_nodes_sorted)
        ]

        # Define colors for keyframes and ordinary frames
        colors = [
            red_node if n in kfs else white_node for n in all_nodes_sorted
        ]

        # Log nodes with their positions, colors, and labels
        rr.log(
            RerunAnnotations.view_graph,
            rr.GraphNodes(
                node_ids=all_nodes_sorted,
                positions=positions,
                labels=[node_labels[n] for n in all_nodes_sorted],
                colors=colors,
            )
        )

        all_nodes_set = set(all_nodes)
        # Log edges of the graph
        rr.log(
            RerunAnnotations.view_graph,
            rr.GraphEdges(edges=[(u, v) for (u, v) in pose_graph.edges])
        )

    @torch.no_grad()
    def write_results(self, frame_i, keyframe_graph):

        self.visualize_keyframes(frame_i, keyframe_graph)
        self.visualize_pose_graph(frame_i, keyframe_graph)
        self.visualize_observed_data(frame_i)

        self.visualize_flow_with_matching_rerun(frame_i)

        # self.visualize_3d_camera_space(frame_i, keyframe_graph)

    def visualize_colmap_track(self, frame_i: int, colmap_reconstruction: pycolmap.Reconstruction,
                               visualize_also_gt_poses: bool,
                               colmap_images_dir: Path | None = None,
                               colmap_segmentations_dir: Path | None = None,
                               gt_model_path: Path | None = None):
        rr.set_time("frame", sequence=frame_i)

        # A reconstruction may register images but contain no triangulated 3D points
        # (degenerate/empty tracks). np.stack on an empty list raises ValueError, so
        # skip the point-cloud logging in that case rather than crashing the sequence.
        # points_3d_coords stays None then and every later use must check for it.
        points_3d_coords = None
        points_3d_colors = None
        if len(colmap_reconstruction.points3D) > 0:
            points_3d_coords = np.stack([p.xyz for p in colmap_reconstruction.points3D.values()], axis=0)
            points_3d_colors = np.stack([p.color for p in colmap_reconstruction.points3D.values()], axis=0)
            rr.log(RerunAnnotations.colmap_pointcloud, rr.Points3D(points_3d_coords, colors=points_3d_colors), static=True)

        if colmap_images_dir is not None:
            log_colmap_point_projections(colmap_reconstruction, colmap_images_dir, colmap_segmentations_dir)

        all_image_names = [str(self.data_graph.get_frame_data(i).image_filename)
                           for i in range(len(self.data_graph.G.nodes))]

        pred_Se3_world2cam_colmap_frames = world2cam_from_reconstruction(colmap_reconstruction)
        pred_Se3_world2cam = {all_image_names.index(colmap_reconstruction.images[colmap_idx].name): Se3_pose
                              for colmap_idx, Se3_pose in pred_Se3_world2cam_colmap_frames.items()}

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        if visualize_also_gt_poses:
            gt_Se3_world2cam = self.accumulate_Se3_attributes(all_frames_from_0, 'gt_Se3_world2cam')

            gt_t_world2cam = gt_Se3_world2cam.inverse().translation.numpy(force=True)
            pred_t_world2cam = np.stack([pred_Se3_world2cam[frm].inverse().t.numpy(force=True)
                                         for frm in sorted(pred_Se3_world2cam)])

            cmap_gt = plt.get_cmap('Reds')
            cmap_pred = plt.get_cmap('Blues')
            gradient = np.linspace(1., 0.5, self.config.input.input_frames)
            colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)
            colors_pred = (np.asarray([cmap_pred(gradient[i])[:3] for i in range(len(pred_t_world2cam))]) * 255).astype(
                np.uint8)

            strips_gt = np.stack([gt_t_world2cam[:-1], gt_t_world2cam[1:]], axis=1)
            strips_pred = np.stack([pred_t_world2cam[:-1], pred_t_world2cam[1:]], axis=1)

            object_size = (np.max(np.linalg.norm(points_3d_coords - np.mean(points_3d_coords, axis=0), axis=1))
                           if points_3d_coords is not None else 1.0)
            strips_radii = [0.005 * object_size] * n_poses

            rr.log(RerunAnnotations.colmap_gt_camera_track,
                   rr.LineStrips3D(strips=strips_gt,  # gt_t_world2cam
                                   colors=colors_gt,
                                   radii=strips_radii),
                   static=True)

            # rr.log(RerunAnnotations.colmap_pred_camera_track,
            #        rr.LineStrips3D(strips=strips_pred,  # gt_t_world2cam
            #                        colors=colors_pred,
            #                        radii=strips_radii),
            #        static=True)

        image_id_to_poses = {}
        image_name_to_image_id = {image.name: image_id for image_id, image in colmap_reconstruction.images.items()}

        G_reliable = nx.Graph()

        for image_id, image in sorted(colmap_reconstruction.images.items(), key=lambda x: x[0]):
            frame_index = all_image_names.index(image.name)

            pred_t_world2cam = torch.tensor(image.cam_from_world().translation)
            pred_q_world2cam_xyzw = torch.tensor(image.cam_from_world().rotation.quat)

            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Transform3D(translation=pred_t_world2cam,
                               rotation=rr.Quaternion(xyzw=pred_q_world2cam_xyzw),
                               from_parent=True),
                static=True
            )

            camera_params = colmap_reconstruction.cameras[image.camera_id]
            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Pinhole(resolution=[camera_params.width, camera_params.height],
                           focal_length=[camera_params.params[0], camera_params.params[1]],
                           camera_xyz=None  # rr.ViewCoordinates.RUB
                           ),
                static=True
            )

            frame_node = self.data_graph.get_frame_data(frame_index)

            # gt_t_world2cam = frame_node.gt_Se3_cam2obj.t
            # print(f'Frame: {frame_index}, gt: {gt_t_world2cam.numpy(force=True).round(3)},'
            #       f'pred: {image.cam_from_world.translation.round(3)}')

            image_id_to_poses[image_id] = pred_t_world2cam

            for reliable_node_idx in frame_node.reliable_sources:
                reliable_node_data = self.data_graph.get_frame_data(reliable_node_idx)
                reliable_node_name = reliable_node_data.image_filename.name

                if reliable_node_name in image_name_to_image_id:
                    reliable_node_image_id = image_name_to_image_id[reliable_node_name]
                    G_reliable.add_edge(image_id, reliable_node_image_id)

        strips = []
        for im_id1, im_id2 in G_reliable.edges:
            im1_t = image_id_to_poses[im_id1]
            im2_t = image_id_to_poses[im_id2]

            strips.append([im1_t, im2_t])

        # --- Visualization 1: GT model + reconstruction pointcloud in "3D Ground Truth" ---
        # Log reconstruction pointcloud into the 3D Ground Truth view
        if points_3d_coords is not None:
            rr.log(RerunAnnotations.space_reconstruction_pointcloud,
                   rr.Points3D(points_3d_coords, colors=points_3d_colors, radii=0.001),
                   static=True)

        # Load and log GT mesh if available
        if gt_model_path is not None and gt_model_path.exists():
            self._log_gt_mesh(gt_model_path)

        # --- Visualization 2: COLMAP co-visibility graph + our initial viewgraph ---
        self._log_graph_comparison(colmap_reconstruction, image_name_to_image_id, G_reliable)

    def _log_gt_mesh(self, gt_model_path: Path):
        """Load GT mesh and log it as rr.Mesh3D to the 3D Ground Truth view.

        Supports textured meshes (UV + albedo texture), vertex-colored meshes,
        and falls back to a default grey if neither is available.
        """
        import io

        try:
            if gt_model_path.suffix.lower() == '.ply':
                cleaned = _load_ply_without_edges(gt_model_path)
                mesh = trimesh.load(io.BytesIO(cleaned), file_type='ply', force='mesh')
            else:
                mesh = trimesh.load(gt_model_path, force='mesh')

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)

            mesh3d_kwargs = dict(
                vertex_positions=vertices,
                triangle_indices=faces,
            )

            if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.material is not None:
                # Textured mesh: extract UV coordinates and albedo texture image
                uv = mesh.visual.uv
                if uv is not None:
                    vertex_texcoords = np.asarray(uv, dtype=np.float32).copy()
                    # Flip V for OpenGL convention (trimesh uses bottom-left origin)
                    vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]
                    mesh3d_kwargs['vertex_texcoords'] = vertex_texcoords

                    # Get the texture image from the material
                    material = mesh.visual.material
                    texture_image = None
                    if hasattr(material, 'image') and material.image is not None:
                        texture_image = np.asarray(material.image.convert('RGB'), dtype=np.uint8)
                    elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                        texture_image = np.asarray(material.baseColorTexture.convert('RGB'), dtype=np.uint8)

                    if texture_image is not None:
                        mesh3d_kwargs['albedo_texture'] = texture_image
                    else:
                        # Has UVs but no texture image — fall back to vertex colors
                        mesh3d_kwargs.pop('vertex_texcoords', None)
                        mesh3d_kwargs['vertex_colors'] = np.asarray(
                            mesh.visual.to_color().vertex_colors)[:, :3]
                else:
                    mesh3d_kwargs['vertex_colors'] = np.asarray(
                        mesh.visual.to_color().vertex_colors)[:, :3]

            elif isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
                mesh3d_kwargs['vertex_colors'] = vertex_colors
            else:
                mesh3d_kwargs['vertex_colors'] = np.full((len(vertices), 3), 180, dtype=np.uint8)

            rr.log(RerunAnnotations.space_gt_mesh, rr.Mesh3D(**mesh3d_kwargs), static=True)

        except Exception as e:
            logger.warning("Failed to load GT mesh from %s: %s", gt_model_path, e)

    def _log_graph_comparison(self, colmap_reconstruction: pycolmap.Reconstruction,
                              image_name_to_image_id: dict, G_reliable: nx.Graph):
        """Log COLMAP co-visibility graph and our initial viewgraph side by side."""
        # Build COLMAP co-visibility graph from shared 3D point tracks.
        # Count shared 3D points per image pair, then keep only edges above the median.
        from collections import Counter
        pair_counts: Counter = Counter()
        for point3D in colmap_reconstruction.points3D.values():
            observer_ids = sorted(elem.image_id for elem in point3D.track.elements)
            for i in range(len(observer_ids)):
                for j in range(i + 1, len(observer_ids)):
                    pair_counts[(observer_ids[i], observer_ids[j])] += 1

        G_covis = nx.Graph()
        for image_id in colmap_reconstruction.images:
            G_covis.add_node(image_id)

        if pair_counts:
            counts = np.array(list(pair_counts.values()))
            threshold = max(int(np.median(counts)), 1)
            for (id_a, id_b), count in pair_counts.items():
                if count >= threshold:
                    G_covis.add_edge(id_a, id_b, weight=count)

        # Log COLMAP co-visibility graph
        covis_node_ids = sorted(G_covis.nodes)
        covis_labels = [colmap_reconstruction.images[nid].name.replace('.png', '')
                        for nid in covis_node_ids]
        rr.log(
            RerunAnnotations.colmap_covisibility_graph,
            rr.GraphNodes(node_ids=covis_node_ids, labels=covis_labels),
            static=True
        )
        rr.log(
            RerunAnnotations.colmap_covisibility_graph,
            rr.GraphEdges(edges=[(u, v) for u, v in G_covis.edges]),
            static=True
        )

        # Log our initial viewgraph (reliable matching graph)
        reliable_node_ids = sorted(G_reliable.nodes)
        reliable_labels = [colmap_reconstruction.images[nid].name.replace('.png', '')
                           if nid in colmap_reconstruction.images else str(nid)
                           for nid in reliable_node_ids]
        rr.log(
            RerunAnnotations.initial_viewgraph,
            rr.GraphNodes(node_ids=reliable_node_ids, labels=reliable_labels),
            static=True
        )
        rr.log(
            RerunAnnotations.initial_viewgraph,
            rr.GraphEdges(edges=[(u, v) for u, v in G_reliable.edges]),
            static=True
        )

    def visualize_3d_camera_space(self, frame_i: int, keyframe_graph: nx.DiGraph):

        rr.set_time("frame", sequence=frame_i)

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        if (frame_i == 1 and self.config.renderer.gt_mesh_path is not None
                and self.config.renderer.gt_texture_path is not None):
            gt_texture = load_texture(Path(self.config.renderer.gt_texture_path),
                                      self.config.renderer.texture_size)
            gt_texture_int = (gt_texture[0].permute(1, 2, 0) * 255).to(torch.uint8)

            gt_mesh = load_mesh_using_trimesh(Path(self.config.renderer.gt_mesh_path))

            normalized_vertices = normalize_vertices(torch.Tensor(gt_mesh.vertices))

            vertex_texcoords = gt_mesh.visual.uv
            vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

            rr.log(
                RerunAnnotations.space_gt_mesh,
                rr.Mesh3D(
                    triangle_indices=gt_mesh.faces,
                    albedo_texture=gt_texture_int,
                    vertex_texcoords=vertex_texcoords,
                    vertex_positions=normalized_vertices
                )
            )

        gt_cam2obj_se3 = self.accumulate_Se3_attributes(all_frames_from_0, 'gt_Se3_cam2obj')
        pred_cam2obj_se3 = self.accumulate_Se3_attributes(all_frames_from_0, 'pred_Se3_cam2obj')

        gt_q_xyzw_cam2obj = gt_cam2obj_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        pred_q_xyzw_cam2obj = pred_cam2obj_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        gt_t_cam2obj = gt_cam2obj_se3.translation.numpy(force=True)
        pred_t_cam2obj = pred_cam2obj_se3.translation.numpy(force=True)

        rr.set_time("frame", sequence=frame_i)

        rr.log(
            RerunAnnotations.space_predicted_camera_pose,
            rr.Transform3D(translation=pred_t_cam2obj[-1],
                           rotation=rr.Quaternion(xyzw=pred_q_xyzw_cam2obj[-1]),
                           )
        )
        rr.log(
            RerunAnnotations.space_gt_camera_pose,
            rr.Transform3D(translation=gt_t_cam2obj[-1],
                           rotation=rr.Quaternion(xyzw=gt_q_xyzw_cam2obj[-1]),
                           )
        )

        cmap_gt = plt.get_cmap('Greens')
        cmap_pred = plt.get_cmap('Blues')
        gradient = np.linspace(1., 0., self.config.input.input_frames)
        colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)
        colors_pred = (np.asarray([cmap_pred(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)

        strips_gt = np.stack([gt_t_cam2obj[:-1], gt_t_cam2obj[1:]], axis=1)
        strips_pred = np.stack([pred_t_cam2obj[:-1], pred_t_cam2obj[1:]], axis=1)

        strips_radii_factor = (max(torch.max(torch.cat([gt_t_cam2obj, pred_t_cam2obj]).norm(dim=1)).item(), 5.) / 5.)
        strips_radii = [0.01 * strips_radii_factor] * n_poses

        rr.log(RerunAnnotations.space_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt,  # gt_t_cam2obj
                               colors=colors_gt,
                               radii=strips_radii))

        rr.log(RerunAnnotations.space_predicted_camera_track,
               rr.LineStrips3D(strips=strips_pred,  # pred_t_cam2obj
                               colors=colors_pred,
                               radii=strips_radii))

        datagraph_camera_node = self.data_graph.get_frame_data(frame_i)
        template_frame_idx = datagraph_camera_node.matching_source_keyframe
        datagraph_template_node = self.data_graph.get_frame_data(template_frame_idx)

        template_node_Se3_cam2obj = datagraph_template_node.pred_Se3_cam2obj
        pred_template_node_t_cam2obj = template_node_Se3_cam2obj.translation.squeeze().numpy(force=True)

        rr.log(RerunAnnotations.space_predicted_closest_keypoint,
               rr.LineStrips3D(strips=[[pred_t_cam2obj[-1],
                                        pred_template_node_t_cam2obj]],
                               colors=[[255, 0, 0]],
                               radii=[0.025 * strips_radii_factor]))

        if len(datagraph_camera_node.reliable_sources) > 1:
            for reliable_template_idx in datagraph_camera_node.reliable_sources:
                datagraph_template_node = self.data_graph.get_frame_data(reliable_template_idx)

                template_node_Se3_cam2obj = datagraph_template_node.pred_Se3_cam2obj
                pred_template_node_t_cam2obj = template_node_Se3_cam2obj.translation.squeeze().numpy(force=True)

                rr.log(f'{RerunAnnotations.space_predicted_reliable_templates}/{reliable_template_idx}',
                       rr.LineStrips3D(strips=[[pred_t_cam2obj[-1],
                                                pred_template_node_t_cam2obj]],
                                       colors=[[255, 255, 0]],
                                       radii=[0.025 * strips_radii_factor]))

        for i, keyframe_node_idx in enumerate(sorted(keyframe_graph.nodes)):

            if keyframe_node_idx not in self.logged_templates_3d_space:
                template_idx = len(self.logged_templates_3d_space)

                keyframe_node = self.data_graph.get_frame_data(keyframe_node_idx)
                template = (keyframe_node.frame_observation.observed_image[0].permute(1, 2, 0).numpy(
                    force=True) * 255.).astype(np.uint8)

                self.logged_templates_3d_space.append(keyframe_node_idx)
                template_image_grid_annotation = (f'{RerunAnnotations.space_predicted_camera_keypoints}/'
                                                  f'{template_idx}')
                rr.log(template_image_grid_annotation,
                       rr.Image(template).compress(jpeg_quality=self.config.visualization.jpeg_quality))

                for template_annotation in self.template_fields:
                    rr.log(template_annotation, rr.Scalars(0.0))

                template_frame_data = self.data_graph.get_frame_data(keyframe_node_idx)
                keyframe_pred_Se3_cam2obj = template_frame_data.pred_Se3_cam2obj

                keyframe_pred_q_cam2obj = keyframe_pred_Se3_cam2obj.quaternion.q[:, [1, 2, 3, 0]].squeeze()
                keyframe_pred_t_cam2obj = keyframe_pred_Se3_cam2obj.translation.squeeze()

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Transform3D(translation=keyframe_pred_t_cam2obj.numpy(force=True),
                                   rotation=rr.Quaternion(xyzw=keyframe_pred_q_cam2obj.numpy(force=True)))
                )
                frame_data = self.data_graph.get_frame_data(keyframe_node_idx)
                fx, fy, cx, cy = extract_intrinsics_from_tensor(frame_data.gt_pinhole_K)

                image_width = frame_data.image_shape.width
                image_height = frame_data.image_shape.height

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Pinhole(
                        resolution=[image_width, image_height],
                        focal_length=[float(fx.item()),
                                      float(fy.item())],
                        camera_xyz=rr.ViewCoordinates.RUB,
                    ),
                )

    def visualize_flow_with_matching_rerun(self, frame_i):

        datagraph_camera_data = self.data_graph.get_frame_data(frame_i)
        new_flow_arc = (datagraph_camera_data.matching_source_keyframe, frame_i)
        flow_arc_source, flow_arc_target = new_flow_arc

        if self.config.onboarding.frame_filter in ('passthrough', 'linear_transition'):
            # Passthrough-style selection: no per-frame (source, frame_i) flow edge is
            # created during filtering, so there is nothing to visualize here.
            return

        if self.config.onboarding.frame_filter in ('vggt_covis', 'vggt_trackvis'):
            # VGGT pair-score filters: edges carry only a reliability scalar (no match
            # points, no certainty maps), and only for pairs the filter actually scored.
            if self.data_graph.G.has_edge(flow_arc_source, flow_arc_target):
                arc = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)
                target_data = self.data_graph.get_frame_data(flow_arc_target)
                if arc.reliability_score is not None:
                    rr.log(RerunAnnotations.matching_reliability, rr.Scalars(arc.reliability_score))
                if target_data.current_flow_reliability_threshold is not None:
                    rr.log(RerunAnnotations.matching_reliability_threshold_roma,
                           rr.Scalars(target_data.current_flow_reliability_threshold))
            return

        arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

        template_data = self.data_graph.get_frame_data(flow_arc_source)
        target_data = self.data_graph.get_frame_data(flow_arc_target)

        if self.config.onboarding.frame_filter in ('dense_matching', 'RANSAC', 'depth'):
            reliability = arc_observation.reliability_score
            rr.log(RerunAnnotations.matching_reliability, rr.Scalars(reliability))
            rr.log(RerunAnnotations.matching_reliability_threshold_roma,
                   rr.Scalars(target_data.current_flow_reliability_threshold))
            if self.config.onboarding.frame_filter == 'depth':
                # Estimated (sim3d) vs GT relative camera pose error, when GT is available.
                if arc_observation.depth_rotation_error_deg is not None:
                    rr.log(RerunAnnotations.depth_rotation_error,
                           rr.Scalars(arc_observation.depth_rotation_error_deg))
                if arc_observation.depth_translation_error_deg is not None:
                    rr.log(RerunAnnotations.depth_translation_error,
                           rr.Scalars(arc_observation.depth_translation_error_deg))
            if self.config.onboarding.matchability_based_reliability:
                matchability_share = template_data.relative_area_matchable
                min_roma_certainty = template_data.roma_certainty_threshold
                rr.log(RerunAnnotations.matching_matchability_plot_share_matchable, rr.Scalars(matchability_share))
                rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty, rr.Scalars(min_roma_certainty))
        elif self.config.onboarding.frame_filter == 'SIFT':
            rr.log(RerunAnnotations.matches_sift, rr.Scalars(arc_observation.num_matches))
            rr.log(RerunAnnotations.min_matches_sift, rr.Scalars(self.config.onboarding.sift_filter_min_matches))
            rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
                   rr.Scalars(self.config.onboarding.sift_filter_good_to_add_matches))

        if frame_i == 0 or (frame_i % self.config.visualization.large_images_write_frequency != 0):
            return

        template_image = template_data.frame_observation.observed_image.squeeze()
        target_image = target_data.frame_observation.observed_image.squeeze()

        template_target_image = torch.cat([template_image, target_image], dim=-2)
        template_target_image_np = (template_target_image.permute(1, 2, 0).numpy(force=True) * 255.).astype(np.uint8)
        rerun_image = rr.Image(template_target_image_np).compress(jpeg_quality=self.config.visualization.jpeg_quality)
        rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
        rr.log(RerunAnnotations.matches_low_certainty, rerun_image)

        # Matchability visualization
        if self.config.onboarding.matchability_based_reliability:
            matchability_mask = template_data.matchability_mask
            matchability_mask_padding = torch.ones_like(matchability_mask)
            matchability_mask_pad = torch.cat([matchability_mask, matchability_mask_padding], dim=0)
            matchability_mask_pad_np = matchability_mask_pad.numpy(force=True)
            template_target_image_matchable_np = overlay_mask(template_target_image_np * 255.,
                                                              ~matchability_mask_pad_np, 1.0, (0, 0, 0))

            mathability_image_rerun = rr.Image(template_target_image_matchable_np).compress(
                jpeg_quality=self.config.visualization.jpeg_quality)
            rr.log(RerunAnnotations.matches_high_certainty_matchable, mathability_image_rerun)
            rr.log(RerunAnnotations.matches_low_certainty_matchable, mathability_image_rerun)

        if self.config.onboarding.frame_filter in ('dense_matching', 'RANSAC', 'depth'):
            certainties = arc_observation.src_dst_certainty_roma.numpy(force=True)
            threshold = template_data.roma_certainty_threshold
            if threshold is None:
                threshold = self.config.onboarding.min_certainty_threshold

            src_pts_yx = arc_observation.src_pts_xy_roma[:, [1, 0]].numpy(force=True)
            dst_pts_yx = arc_observation.dst_pts_xy_roma[:, [1, 0]].numpy(force=True)

            if arc_observation.roma_flow_warp_certainty is not None:
                visualize_certainty_map(arc_observation.roma_flow_warp_certainty,
                                        template_target_image.shape, template_target_image_np,
                                        RerunAnnotations.matching_certainty,
                                        self.config.visualization.jpeg_quality)
        elif self.config.onboarding.frame_filter == 'SIFT':
            src_pts_xy = arc_observation.src_pts_xy_roma
            dst_pts_xy = arc_observation.dst_pts_xy_roma

            if src_pts_xy is not None and dst_pts_xy is not None and len(src_pts_xy) > 0:
                src_pts_yx = src_pts_xy[:, [1, 0]].numpy(force=True)
                dst_pts_yx = dst_pts_xy[:, [1, 0]].numpy(force=True)
            else:
                src_pts_yx = np.zeros((0, 2))
                dst_pts_yx = np.zeros((0, 2))
            certainties = np.ones(src_pts_yx.shape[0])
            threshold = 0.0  # all SIFT matches are inliers
        else:
            return

        template_image_size = template_data.image_shape
        log_matching_correspondences(src_pts_yx, dst_pts_yx, certainties, threshold,
                                     template_image_size.height,
                                     RerunAnnotations.matches_high_certainty,
                                     RerunAnnotations.matches_low_certainty, 20)

        if self.config.onboarding.matchability_based_reliability and self.config.onboarding.frame_filter == 'dense_matching':
            matchable_certainties = arc_observation.src_dst_certainty_roma_matchable.numpy(force=True)
            matchable_src_yx = arc_observation.src_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            matchable_dst_yx = arc_observation.dst_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            log_matching_correspondences(matchable_src_yx, matchable_dst_yx, matchable_certainties,
                                         threshold, template_image_size.height,
                                         RerunAnnotations.matches_high_certainty_matchable,
                                         RerunAnnotations.matches_low_certainty_matchable, 20)

    def accumulate_Se3_attributes(self, frame_indices, attr_name: str) -> Se3:

        Ts_cam2obj = []

        for frame in frame_indices:
            frame_data = self.data_graph.get_frame_data(frame)
            Ts_cam2obj.append(getattr(frame_data, attr_name).matrix().squeeze())

        T_cam2obj = torch.stack(Ts_cam2obj, dim=0).to(self.config.run.device)
        Se3_cam2obj = Se3.from_matrix(T_cam2obj)

        return Se3_cam2obj

    def visualize_observed_data(self, frame_i):

        if frame_i % self.config.visualization.large_images_write_frequency != 0:
            return

        observed_image_annotation = RerunAnnotations.observed_image
        observed_image_segmentation_annotation = RerunAnnotations.observed_image_segmentation

        prev_visualized_frame_idx = self.config.visualization.large_images_write_frequency
        # Save the images to disk
        prev_frame = self.data_graph.get_frame_data(frame_i - prev_visualized_frame_idx) \
            if frame_i >= prev_visualized_frame_idx else None
        current_datagraph_node = self.data_graph.get_frame_data(frame_i)
        last_frame_observation = current_datagraph_node.frame_observation

        new_image_path = self.observations_path / Path(f'image_{frame_i}.png')
        last_observed_image = last_frame_observation.observed_image.squeeze().permute(1, 2, 0)

        self.log_image(frame_i, last_observed_image, observed_image_annotation, new_image_path)

        rr.set_time("frame", sequence=frame_i)

        if frame_i == 0 or prev_frame.matching_source_keyframe != current_datagraph_node.matching_source_keyframe:
            template_frame_node = self.data_graph.get_frame_data(current_datagraph_node.matching_source_keyframe)
            template_frame_observation = template_frame_node.frame_observation
            template = template_frame_observation.observed_image.squeeze().permute(1, 2, 0)
            template_segment = template_frame_observation.observed_segmentation.numpy(force=True)
            template_path = Path('')
            self.log_image(frame_i, template, RerunAnnotations.template_image, template_path)
            rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(template_segment))

        image_segmentation = last_frame_observation.observed_segmentation.numpy(force=True)
        rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(image_segmentation))

    def log_image(self, frame: int, image: torch.Tensor, rerun_annotation: str, save_path: Optional[Path] = None,
                  ignore_dimensions=False):
        if not ignore_dimensions:
            assert len(image.shape) == 3 and image.shape[-1] == 3

        if self.config.visualization.write_to_rerun:
            rr.set_time("frame", sequence=frame)
            image_np = (image.numpy(force=True) * 255.).astype(np.uint8) if image.dtype != torch.uint8 else image.numpy(
                force=True)
            rr.log(rerun_annotation, rr.Image(image_np).compress(jpeg_quality=self.config.visualization.jpeg_quality))
        else:
            image_np = image.numpy(force=True)
            imageio.imwrite(save_path, image_np)

    def log_pyplot(self, frame: int, fig: plt.plot, save_path: Path, rerun_annotation: str, **kwargs):

        if self.config.visualization.write_to_rerun:
            fig.canvas.draw()

            image_bytes_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_np = image_bytes_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image_np)
            rr.set_time("frame", sequence=frame)
            rr.log(rerun_annotation, rr.Image(image).compress(jpeg_quality=self.config.visualization.jpeg_quality))
        else:
            plt.savefig(str(save_path), **kwargs)

        plt.close()
