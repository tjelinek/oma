"""Adapter for SpatialTrackerV2 (SpaTrackV2) point tracking.

Sole location importing SpaTrackerV2 internals. SpatialTrackerV2
(arXiv:2507.12462) is a feed-forward 3D point tracker: a VGGT-style front-end
(VGGT4Track) predicts per-frame depth, intrinsics and camera poses from RGB,
and the tracker head decomposes observed motion into scene geometry, camera
ego-motion and residual object motion. We consume its 2D track projections +
visibility through the same track() contract as CoTrackerAdapter, so
PointTrackingMatchingProvider can select it via BaseTrackingConfig.tracker.

Import note: the repo's code lives under a top-level `models` package
(repositories/SpaTrackerV2/models/...). Both GloPose's `models/` and the
repo's `models/` are namespace packages (no __init__.py), so appending the
repo root to sys.path merges them — `models.SpaTrackV2.*` resolves without
shadowing GloPose modules. Its vggt4track module imports `vggt.*`, served by
repositories/vggt via the existing vggt adapter path helper.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

SPATRACK2_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'SpaTrackerV2'
# Released checkpoints, mirrored locally (compute nodes run HF_HUB_OFFLINE=1).
DEFAULT_WEIGHTS_DIR = os.environ.get('SPATRACKV2_WEIGHTS_DIR', 'weights/spatrackv2')


def _ensure_spatrack2_on_path():
    from adapters.vggt_adapter import _ensure_vggt_on_path
    _ensure_vggt_on_path()
    repo = str(SPATRACK2_REPO)
    if repo not in sys.path:
        sys.path.append(repo)  # append: GloPose top-level packages keep priority


class _BAConfigCompat:
    """BundleAdjustmentConfig shim for pycolmap 4: set_constant_cam_pose was
    renamed to set_constant_rig_from_world_pose (frame_id == image_id in our
    1:1 frame builder), and the variable-point bookkeeping methods were removed
    (points are variable by default) — we only track ids so ba_pycolmap's
    '< 50 points' early-exit check keeps working."""

    def __init__(self, cfg):
        object.__setattr__(self, '_cfg', cfg)
        object.__setattr__(self, '_variable_pids', set())

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_cfg'), name)

    def set_constant_cam_pose(self, image_id):
        self._cfg.set_constant_rig_from_world_pose(image_id)

    def add_variable_point(self, point3D_id):
        self._variable_pids.add(point3D_id)

    @property
    def variable_point3D_ids(self):
        return self._variable_pids

    @property
    def constant_point3D_ids(self):
        return self._cfg.constant_points


class _PycolmapCompat:
    """Delegating pycolmap proxy for SpaTrackerV2's bundled ba.py, written
    against pycolmap ≤3.x. Installed into that module's namespace only — the
    global pycolmap module is left untouched."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def BundleAdjustmentConfig(self):
        return _BAConfigCompat(self._mod.BundleAdjustmentConfig())


def _batch_matrix_to_pycolmap_v4(points3d, extrinsics, intrinsics, tracks, masks,
                                 image_size, max_points3D_val=3000, shared_camera=False,
                                 camera_type="SIMPLE_PINHOLE", extra_params=None,
                                 cam_tracks_static=None, query_pts=None):
    """pycolmap-4 port of ba.py::batch_matrix_to_pycolmap. The original builds
    pycolmap.Image(id=..., cam_from_world=...), both illegal in pycolmap 4
    (id renamed to image_id, cam_from_world read-only — poses live on Frames);
    we reuse GloPose's Rig->Frame->Image builder instead. Semantics (0-based
    image/camera ids, per-frame cameras, track wiring) match the original."""
    import pycolmap
    from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

    N, P, _ = tracks.shape
    assert len(extrinsics) == N and len(intrinsics) == N and len(points3d) == P

    extrinsics = extrinsics.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    if extra_params is not None:
        extra_params = extra_params.cpu().numpy()
    tracks = tracks.cpu().numpy()
    masks = masks.cpu().numpy()
    points3d = points3d.cpu().numpy()
    image_size = image_size.cpu().numpy()
    if cam_tracks_static is not None:
        cam_tracks_static = cam_tracks_static.cpu().numpy()

    reconstruction = pycolmap.Reconstruction()

    valid_idx = np.nonzero(masks.sum(0) >= 2)[0]  # tracks with >= 2 inliers

    point3d_ids = []
    for vidx in valid_idx:
        point3d_id = reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), np.zeros(3))
        point3d_ids.append(point3d_id)

    if cam_tracks_static is not None:
        extra_residual = []
        for id_x, vidx in enumerate(valid_idx):
            query_i = query_pts[:, :, vidx]
            extra_residual.append({
                "point3D_id": point3d_ids[id_x],
                "image_ids": [int(query_i[0, 0, 0])],
                "observed_depth": [query_i[0, 0, -1]],
            })
    else:
        extra_residual = None

    num_points3D = len(valid_idx)
    camera = None
    for fidx in range(N):
        if camera is None or (not shared_camera):
            if camera_type == "SIMPLE_RADIAL":
                pycolmap_intri = np.array([intrinsics[fidx][0, 0], intrinsics[fidx][0, 2],
                                           intrinsics[fidx][1, 2], extra_params[fidx][0]])
            elif camera_type == "SIMPLE_PINHOLE":
                pycolmap_intri = np.array([intrinsics[fidx][0, 0], intrinsics[fidx][0, 2],
                                           intrinsics[fidx][1, 2]])
            else:
                raise ValueError(f"Camera type {camera_type} is not supported yet")
            camera = pycolmap.Camera(model=camera_type, width=int(image_size[0]),
                                     height=int(image_size[1]), params=pycolmap_intri,
                                     camera_id=fidx)
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3])

        points2D_list = []
        point2D_idx = 0
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]
            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all() \
                    and masks[fidx][original_track_idx]:
                points2D_list.append(pycolmap.Point2D(tracks[fidx][original_track_idx],
                                                      point3D_id))
                reconstruction.points3D[point3D_id].track.add_element(fidx, point2D_idx)
                point2D_idx += 1

        add_posed_image_to_reconstruction(
            reconstruction, fidx, camera.camera_id, f"image_{fidx}", cam_from_world,
            points2D=make_point2d_list(points2D_list))

    return reconstruction, valid_idx, extra_residual


def _solve_bundle_adjustment_v4(reconstruction, ba_options, ba_config=None,
                                extra_residual=None):
    """pycolmap-4 port of ba.py::solve_bundle_adjustment: the manual
    create_solver_options + pyceres.solve dance was replaced by
    BundleAdjuster.solve(). extra_residual was already dead code upstream."""
    import pycolmap
    adjuster = pycolmap.create_default_bundle_adjuster(
        ba_options, getattr(ba_config, '_cfg', ba_config), reconstruction)
    # .ceres_summary is the pyceres SolverSummary whose fields (num_residuals_reduced,
    # initial/final_cost, ...) ba.py::log_ba_summary reads.
    return adjuster.solve().ceres_summary


def _pycolmap_to_batch_matrix_v4(reconstruction, device="cuda",
                                 camera_type="SIMPLE_PINHOLE"):
    """pycolmap-4 port of ba.py::pycolmap_to_batch_matrix — Image.cam_from_world
    became a method (was a property)."""
    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))
    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz
    points3D = torch.from_numpy(points3D).to(device)

    extrinsics, intrinsics = [], []
    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None
    for i in range(num_images):
        pyimg = reconstruction.images[i]
        pycam = reconstruction.cameras[pyimg.camera_id]
        extrinsics.append(pyimg.cam_from_world().matrix())
        intrinsics.append(pycam.calibration_matrix())
        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    extrinsics = torch.from_numpy(np.stack(extrinsics)).to(device)
    intrinsics = torch.from_numpy(np.stack(intrinsics)).to(device)
    if camera_type == "SIMPLE_RADIAL":
        extra_params = torch.from_numpy(np.stack(extra_params)).to(device)[:, None]
    return points3D, extrinsics, intrinsics, extra_params


def _weighted_procrustes_torch_safe(X, Y, W=None, RT=None):
    """Port of spatrack_modules/utils.py::weighted_procrustes_torch with the
    degenerate-solve branch fixed: upstream hits a live `pdb.set_trace()` when
    the SVD rotation is non-orthonormal (e.g. all-zero weights on masked
    frames), which kills non-interactive jobs with bdb.BdbQuit. We instead fall
    back to identity for the degenerate (B, T) entries and keep the rest."""
    device = X.device
    B, T, N, _ = Y.shape

    if W is None:
        W = torch.ones(B, 1, N, device=device)
    elif W.dim() == 3:  # (B, T, N)
        W = W.unsqueeze(-1)
    else:  # (B, 1, N)
        W = W.unsqueeze(-1).expand(B, T, N, 1)

    X = X.expand(B, T, N, 3)

    sum_W = torch.sum(W, dim=2, keepdim=True)
    centroid_X = torch.sum(W * X, dim=2) / sum_W.squeeze(-1)
    centroid_Y = torch.sum(W * Y, dim=2) / sum_W.squeeze(-1)

    X_centered = X - centroid_X.unsqueeze(2)
    Y_centered = Y - centroid_Y.unsqueeze(2)

    X_weighted = X_centered * W
    H = torch.matmul(X_weighted.transpose(2, 3), Y_centered)

    U, S, Vt = torch.linalg.svd(H)

    det = torch.det(torch.matmul(U, Vt))
    Vt_corrected = Vt.clone()
    B_idx, T_idx = torch.nonzero(det < 0, as_tuple=True)
    Vt_corrected[B_idx, T_idx, -1, :] *= -1

    R = torch.matmul(U, Vt_corrected).inverse()
    t = centroid_Y - torch.matmul(R, centroid_X.unsqueeze(-1)).squeeze(-1)

    ok = torch.isfinite(R).all(-1).all(-1) & ((torch.det(R) - 1).abs() < 1e-3)  # (B, T)
    eye = torch.eye(3, device=device, dtype=R.dtype).expand_as(R)
    R = torch.where(ok[..., None, None], R, eye)
    t = torch.where(ok[..., None], t, torch.zeros_like(t))

    w2c = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    w2c[:, :, :3, :3] = R
    w2c[:, :, :3, 3] = t
    try:
        c2w_traj = torch.inverse(w2c)
    except Exception:
        c2w_traj = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)

    return c2w_traj


def _install_pycolmap4_compat(ba_mod):
    import pycolmap
    if not isinstance(ba_mod.pycolmap, _PycolmapCompat):
        ba_mod.pycolmap = _PycolmapCompat(pycolmap)
    ba_mod.batch_matrix_to_pycolmap = _batch_matrix_to_pycolmap_v4
    ba_mod.solve_bundle_adjustment = _solve_bundle_adjustment_v4
    ba_mod.pycolmap_to_batch_matrix = _pycolmap_to_batch_matrix_v4


class SpaTrack2Adapter:

    def __init__(self, device: str, weights_dir: str | None = None):
        _ensure_spatrack2_on_path()
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        from models.SpaTrackV2.models.predictor import Predictor

        self.device = device
        wdir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        front_src = str(wdir / 'front') if (wdir / 'front').exists() \
            else 'Yuxihenry/SpatialTrackerV2_Front'
        offline_src = str(wdir / 'offline') if (wdir / 'offline').exists() \
            else 'Yuxihenry/SpatialTrackerV2-Offline'
        self.front = VGGT4Track.from_pretrained(front_src).eval().to(device)
        self.tracker = Predictor.from_pretrained(offline_src)
        self.tracker.eval()
        self.tracker.to(device)

        import models.SpaTrackV2.models.tracker3D.spatrack_modules.ba as _spat_ba
        _install_pycolmap4_compat(_spat_ba)
        # weighted_procrustes_torch is used both as a module global (utils.py's
        # key_fr_wprocrustes) and as a from-import (TrackRefiner) — patch both.
        import models.SpaTrackV2.models.tracker3D.spatrack_modules.utils as _spat_utils
        import models.SpaTrackV2.models.tracker3D.TrackRefiner as _spat_refiner
        _spat_utils.weighted_procrustes_torch = _weighted_procrustes_torch_safe
        _spat_refiner.weighted_procrustes_torch = _weighted_procrustes_torch_safe

    @torch.no_grad()
    def track(self, video: torch.Tensor, queries_xy: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Track query points through a video chunk (same contract as CoTrackerAdapter).

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]; frame 0 is the query frame.
            queries_xy: (N, 2) float tensor of (x, y) query positions in frame 0.

        Returns:
            tracks: (T, N, 2) tracked (x, y) positions per frame, in input pixels.
            certainty: (T, N) float in [0, 1] — the tracker's sigmoided visibility.
        """
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

        torch.cuda.empty_cache()
        T, _, H, W = video.shape
        # SpaTrackerV2's refiner computes second temporal derivatives (shape
        # (B, T-2, N)) and crashes on empty-tensor .max() for 2-frame chunks —
        # pad to 3 by duplicating the last frame and drop its outputs below.
        t_pad = 0
        if T == 2:
            video = torch.cat([video, video[-1:]], dim=0)
            t_pad, T = 1, 3
        vid255 = (video * 255.0).float()
        proc = preprocess_image(vid255)[None]  # (1, T, 3, h, w), 0-255

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            preds = self.front(proc.to(self.device) / 255.0)
        extrs = preds['poses_pred'].squeeze(0).cpu().numpy()
        intrs = preds['intrs'].squeeze(0).cpu().numpy()
        depth = preds['points_map'][..., 2].squeeze(0).cpu().numpy()
        unc_metric = preds['unc_metric'].squeeze(0).cpu().numpy() > 0.5

        vidproc = proc.squeeze(0)  # (T, 3, h, w), 0-255, CPU
        h, w = vidproc.shape[-2:]
        sx, sy = w / W, h / H
        q = queries_xy.float().cpu().numpy() * np.array([sx, sy])
        query_xyt = np.concatenate([np.zeros((len(q), 1)), q], axis=1)  # (N, 3) [t, x, y]

        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            (_c2w, _intrs, _pmap, _cdepth, _track3d, track2d, vis, conf, _video) = \
                self.tracker.forward(vidproc, depth=depth, intrs=intrs, extrs=extrs,
                                     queries=query_xyt, fps=1, full_point=False,
                                     iters_track=4, query_no_BA=True, fixed_cam=False,
                                     stage=1, unc_metric=unc_metric,
                                     support_frame=T - 1, replace_ratio=0.2)
        torch.cuda.empty_cache()

        track2d = track2d.float().cpu()
        vis = vis.float().cpu()
        if track2d.dim() == 4:  # strip batch dim if present
            track2d = track2d[0]
        if vis.dim() == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]  # forward_stream returns vis_pred as (T, N, 1)
        elif vis.dim() == 3:
            vis = vis[0]  # (B, T, N)
        if t_pad:
            track2d, vis = track2d[:-t_pad], vis[:-t_pad]
        tracks = (track2d[..., :2] / torch.tensor([sx, sy])).to(self.device)
        certainty = vis.clamp(0, 1).to(self.device)
        if os.environ.get('GLOPOSE_TRACK_DEBUG'):
            qs = torch.tensor([0.1, 0.5, 0.9])
            c = conf.float().cpu().reshape(-1)
            print(f'[SpaTrack2-debug] endpoint vis q10/50/90: '
                  f'{[round(x, 3) for x in torch.quantile(certainty[-1], qs).tolist()]} '
                  f'| conf q10/50/90: {[round(x, 3) for x in torch.quantile(c, qs).tolist()]} '
                  f'| drift p50: {(tracks[-1] - tracks[0]).norm(dim=1).median().item():.1f}px',
                  flush=True)
        return tracks, certainty
