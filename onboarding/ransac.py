import numpy as np
import pycolmap

from configs.glopose_config import RANSACConfig
from kornia.image import ImageSize
from onboarding.colmap_utils import colmap_K_params_vec


def estimate_inlier_mask(
    src_pts_xy: np.ndarray,
    dst_pts_xy: np.ndarray,
    ransac_config: RANSACConfig,
    K1: np.ndarray | None = None,
    K2: np.ndarray | None = None,
    source_shape: ImageSize | None = None,
    target_shape: ImageSize | None = None,
    confidences: np.ndarray | None = None,
) -> np.ndarray | None:
    """Estimate essential/fundamental matrix via RANSAC and return a boolean inlier mask.

    Returns None if estimation fails or too few matches are provided.
    """
    if len(src_pts_xy) < ransac_config.min_num_matches:
        return None

    method = ransac_config.method
    if method == 'pycolmap':
        return _estimate_pycolmap(src_pts_xy, dst_pts_xy, ransac_config, K1, K2, source_shape, target_shape)
    elif method in ('magsac++', 'ransac', '8point'):
        return _estimate_opencv(src_pts_xy, dst_pts_xy, ransac_config, K1)
    elif method == 'pygcransac':
        return _estimate_pygcransac(src_pts_xy, dst_pts_xy, ransac_config, K1, K2, source_shape, target_shape,
                                    confidences)
    else:
        raise ValueError(f"Unknown RANSAC method '{method}'. "
                         f"Options: 'pycolmap', 'magsac++', 'ransac', '8point', 'pygcransac'")


def _estimate_pycolmap(
    src_pts_xy: np.ndarray,
    dst_pts_xy: np.ndarray,
    config: RANSACConfig,
    K1: np.ndarray | None,
    K2: np.ndarray | None,
    source_shape: ImageSize | None,
    target_shape: ImageSize | None,
) -> np.ndarray | None:
    ransac_opts = pycolmap.RANSACOptions()
    ransac_opts.max_error = config.max_error
    ransac_opts.confidence = config.confidence

    if K1 is not None and K2 is not None:
        K_params1 = colmap_K_params_vec(K1)
        K_params2 = colmap_K_params_vec(K2)

        camera1 = pycolmap.Camera(
            camera_id=1, model=pycolmap.CameraModelId.PINHOLE,
            width=source_shape.width, height=source_shape.height, params=K_params1)
        camera2 = pycolmap.Camera(
            camera_id=1, model=pycolmap.CameraModelId.PINHOLE,
            width=target_shape.width, height=target_shape.height, params=K_params2)

        result = pycolmap.estimate_essential_matrix(src_pts_xy, dst_pts_xy, camera1, camera2, ransac_opts)
    else:
        result = pycolmap.estimate_fundamental_matrix(src_pts_xy, dst_pts_xy, ransac_opts)

    if result is None:
        return None
    return result.get('inlier_mask')


def _estimate_opencv(
    src_pts_xy: np.ndarray,
    dst_pts_xy: np.ndarray,
    config: RANSACConfig,
    K1: np.ndarray | None,
) -> np.ndarray | None:
    import cv2

    method_map = {
        'magsac++': cv2.USAC_MAGSAC,
        'ransac': cv2.RANSAC,
        '8point': cv2.USAC_FM_8PTS,
    }
    cv2_method = method_map[config.method]

    if K1 is not None:
        _, mask = cv2.findEssentialMat(
            src_pts_xy, dst_pts_xy, K1,
            method=cv2_method, threshold=config.max_error, prob=config.confidence)
    else:
        _, mask = cv2.findFundamentalMat(
            src_pts_xy, dst_pts_xy,
            method=cv2_method, ransacReprojThreshold=config.max_error, confidence=config.confidence)

    if mask is None:
        return None
    return mask.ravel().astype(np.bool_)


def _estimate_pygcransac(
    src_pts_xy: np.ndarray,
    dst_pts_xy: np.ndarray,
    config: RANSACConfig,
    K1: np.ndarray | None,
    K2: np.ndarray | None,
    source_shape: ImageSize | None,
    target_shape: ImageSize | None,
    confidences: np.ndarray | None,
) -> np.ndarray | None:
    try:
        import pygcransac
    except ImportError:
        raise ImportError(
            "pygcransac is required for method='pygcransac'. "
            "Install it with: pip install pygcransac")

    if K1 is None or K2 is None:
        raise ValueError("pygcransac requires camera intrinsics (K1 and K2)")
    if source_shape is None or target_shape is None:
        raise ValueError("pygcransac requires source_shape and target_shape")

    correspondences = np.ascontiguousarray(np.concatenate([src_pts_xy, dst_pts_xy], axis=1))

    if config.use_prosac and confidences is not None:
        ordering = confidences.argsort()[::-1]
        confidences = confidences[ordering]
        correspondences = correspondences[ordering]

        _, mask = pygcransac.findEssentialMatrix(
            correspondences, K1, K2,
            source_shape.height, source_shape.width,
            target_shape.height, target_shape.width,
            confidences,
            threshold=config.max_error, min_iters=config.min_iters, sampler=1)
    else:
        _, mask = pygcransac.findEssentialMatrix(
            correspondences, K1, K2,
            source_shape.height, source_shape.width,
            target_shape.height, target_shape.width,
            config.confidence,
            threshold=config.max_error, min_iters=config.min_iters)

    if mask is None:
        return None
    return mask.astype(np.bool_)
