from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

import imageio.v3 as iio

from data_structures.rerun_annotations import RerunAnnotations
from utils.image_utils import overlay_mask


def init_rerun_recording(name: str, output_path: Path, blueprint: rrb.Blueprint):
    """Initialize a rerun recording: init, save to .rrd file, and send blueprint."""
    rr.init(name)
    rr.save(output_path)
    rr.send_blueprint(blueprint)


def register_matching_series_lines():
    """Register shared series line styles for matching reliability/matchability plots."""
    rr.log(RerunAnnotations.matching_reliability_threshold_roma,
           rr.SeriesLines(colors=[255, 0, 0], names="min reliability"), static=True)
    rr.log(RerunAnnotations.matching_reliability,
           rr.SeriesLines(colors=[0, 0, 255], names="reliability"), static=True)
    rr.log(RerunAnnotations.matching_matchability_plot_share_matchable,
           rr.SeriesLines(colors=[255, 0, 0], names="share of matchable fg"), static=True)
    rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty,
           rr.SeriesLines(colors=[0, 0, 255], names="min match certainty"), static=True)


def log_correspondences_rerun(cmap, src_yx, target_yx, rerun_annotation, source_image_height, sample_size=None):
    if sample_size is not None:
        random_indices = torch.randperm(min(sample_size, src_yx.shape[0]))
        src_yx = src_yx[random_indices]
        target_yx = target_yx[random_indices]

    if len(src_yx.shape) == 1 or src_yx.shape[1] == 0:
        return  # No matches to draw
    target_yx_2nd_image = target_yx
    target_yx_2nd_image[:, 0] = source_image_height + target_yx_2nd_image[:, 0]

    line_strips_xy = np.stack([src_yx[:, [1, 0]], target_yx_2nd_image[:, [1, 0]]], axis=1)

    num_points = line_strips_xy.shape[0]
    colors = [cmap(i / num_points)[:3] for i in range(num_points)]
    colors = (np.array(colors) * 255).astype(int).tolist()

    rr.log(
        rerun_annotation,
        rr.LineStrips2D(
            strips=line_strips_xy,
            colors=colors,
        ),
    )


def visualize_certainty_map(certainty_map: torch.Tensor,
                            target_image_shape: tuple,
                            template_target_image_np: np.ndarray,
                            certainty_annotation: str,
                            jpeg_quality: int = 75):
    """Reshape a (H, W*2) certainty map into column layout, resize to match the concatenated
    template+target image, overlay as heatmap, and log to rerun.

    Args:
        certainty_map: (H, W*2) or (1, H, W*2) certainty map from matcher
        target_image_shape: (C, H, W) shape of the concatenated template+target image tensor
        template_target_image_np: (H*2, W, 3) uint8 numpy image (or ones for black background)
        certainty_annotation: rerun annotation path for the certainty view
        jpeg_quality: JPEG compression quality for rerun
    """
    roma_h, roma_w = certainty_map.shape[-2], certainty_map.shape[-1] // 2
    certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(certainty_map.device)
    certainty_map_column[:roma_h, :roma_w] = certainty_map[..., :roma_h, :roma_w]
    certainty_map_column[roma_h:, :roma_w] = certainty_map[..., :roma_h, roma_w:]
    certainty_map_column = certainty_map_column[None]
    certainty_map_resized = TF.resize(certainty_map_column, size=list(target_image_shape[1:]))

    certainty_map_np = certainty_map_resized.numpy(force=True)
    background = np.ones_like(template_target_image_np)
    certainty_overlay = overlay_mask(background, certainty_map_np)

    rr.log(certainty_annotation,
           rr.Image(certainty_overlay).compress(jpeg_quality=jpeg_quality))


def log_matching_correspondences(src_pts_yx: np.ndarray, dst_pts_yx: np.ndarray,
                                 certainties: np.ndarray, threshold: float,
                                 source_image_height: int,
                                 high_certainty_annotation: str,
                                 low_certainty_annotation: str,
                                 sample_size: int = 2000):
    """Split points by certainty threshold and log inlier/outlier correspondences to rerun.

    Args:
        src_pts_yx: (N, 2) source points in (y, x) order
        dst_pts_yx: (N, 2) destination points in (y, x) order
        certainties: (N,) certainty values
        threshold: certainty threshold for inlier/outlier split
        source_image_height: height of the source (template) image for vertical offset
        high_certainty_annotation: rerun annotation path for inliers
        low_certainty_annotation: rerun annotation path for outliers
        sample_size: max number of correspondences to draw per group
    """
    above_threshold_mask = certainties >= threshold

    inliers_source_yx = src_pts_yx[above_threshold_mask]
    inliers_target_yx = dst_pts_yx[above_threshold_mask]
    outliers_source_yx = src_pts_yx[~above_threshold_mask]
    outliers_target_yx = dst_pts_yx[~above_threshold_mask]

    cmap_inliers = plt.get_cmap('Greens')
    log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                              high_certainty_annotation, source_image_height, sample_size)
    cmap_outliers = plt.get_cmap('Reds')
    log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                              low_certainty_annotation, source_image_height, sample_size)


def log_colmap_point_projections(reconstruction,
                                  images_dir: Path,
                                  segmentations_dir: Path | None = None,
                                  jpeg_quality: int = 75):
    """Log 2D projections of COLMAP 3D points onto each keyframe image in Rerun.

    Projects all 3D points through each image's camera model to get 2D positions,
    rather than relying on stored point2D.xy (which may be in a different coordinate
    space for external reconstruction methods like VGGT).

    If segmentations_dir is provided, points are colored green (inside mask) or red (outside mask).

    Args:
        reconstruction: pycolmap.Reconstruction with registered images and 3D points
        images_dir: directory containing the keyframe images (filenames match image.name)
        segmentations_dir: optional directory with segmentation masks ({image.name}.png)
        jpeg_quality: JPEG compression quality for rerun image logging
    """
    if len(reconstruction.points3D) == 0:
        return

    # Gather all 3D points once
    point3d_ids = list(reconstruction.points3D.keys())
    points_3d = np.stack([reconstruction.points3D[pid].xyz for pid in point3d_ids], axis=0)
    points_3d_colors = np.stack([reconstruction.points3D[pid].color for pid in point3d_ids], axis=0)

    for image_id, image in sorted(reconstruction.images.items(), key=lambda x: x[0]):
        image_path = images_dir / image.name
        if not image_path.exists():
            continue

        image_np = iio.imread(image_path)
        h, w = image_np.shape[:2]

        # Project all 3D points through this image's camera
        camera = reconstruction.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()

        # Transform to camera coordinates
        points_cam = (cam_from_world.rotation.matrix() @ points_3d.T).T + cam_from_world.translation
        # Keep only points in front of camera
        in_front = points_cam[:, 2] > 0
        if not np.any(in_front):
            continue

        # Project to 2D using camera model
        points_2d = np.stack([camera.img_from_cam(p) for p in points_cam[in_front]], axis=0)

        # Keep only points within image bounds
        in_bounds = ((points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h))
        points_xy = points_2d[in_bounds]
        visible_colors = points_3d_colors[in_front][in_bounds]

        if len(points_xy) == 0:
            continue

        # Color by segmentation mask membership if available
        if segmentations_dir is not None:
            seg_path = segmentations_dir / f'{image.name}.png'
            if seg_path.exists():
                seg_mask = iio.imread(seg_path)
                if seg_mask.ndim == 3:
                    seg_mask = seg_mask[..., 0]

                px = np.clip(points_xy[:, 0].astype(int), 0, w - 1)
                py = np.clip(points_xy[:, 1].astype(int), 0, h - 1)
                inside = seg_mask[py, px] > 127

                colors = np.where(inside[:, None],
                                  np.array([[0, 200, 0, 255]], dtype=np.uint8),
                                  np.array([[200, 0, 0, 255]], dtype=np.uint8))
            else:
                colors = np.full((len(points_xy), 4), [0, 200, 0, 255], dtype=np.uint8)
        else:
            colors = visible_colors

        entity_path = f'{RerunAnnotations.colmap_point_projections}/{image_id}'
        rr.log(entity_path, rr.Image(image_np).compress(jpeg_quality=jpeg_quality), static=True)
        rr.log(entity_path + '/points',
               rr.Points2D(positions=points_xy, colors=colors, radii=2.0), static=True)
