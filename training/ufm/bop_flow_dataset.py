"""
BOP-PBR dense-flow dataset for fine-tuning UFM on object-only (masked) matching.

For a static multi-object PBR scene the dense optical flow of a (static) object
between two views is fully determined by the source-view depth, the camera
intrinsics, and the RELATIVE camera pose between the two views. We recover that
relative pose from the target object's own pose (`cam_R_m2c`, `cam_t_m2c`) in the
two frames -- BOP classic `train_pbr` does NOT store world camera extrinsics
(`scene_camera.json` has only `cam_K` + `depth_scale`), so we cannot go through a
world frame. For a static object o:

    cam_A -> cam_B :  R_AB = R_m2cB @ R_m2cA^T,  t_AB = t_m2cB - R_AB @ t_m2cA

This is self-contained per object and works for every BOP-PBR dataset.

BOP-PBR `scene_gt` lists the same instances in the same order in every frame of a
scene, so an instance INDEX identifies the same physical instance across frames
(this is what `mask_visib/{frame:06d}_{inst:06d}.png` indexes too). We therefore
group samples by instance index and pair two frames of the SAME index -- no
cross-view matching needed.

Each sample returns, at a fixed (W, H) working resolution:
  - img0, img1     : (3, H, W) float32 in [0, 1], background blacked out
  - gt_flow        : (2, H, W) float32, source->target pixel flow (x, y)
  - covisibility   : (H, W) float32 in {0, 1}, GT for the occlusion/covis head
                     (1 = source object pixel depth-consistently visible in target)
  - flow_valid     : (H, W) float32 in {0, 1}, where the EPE loss is supervised
  - obj_mask0      : (H, W) float32, source object mask (diagnostics / viz)

Normalization to the encoder's `data_norm_type` is applied later, in the training
step, so the dataset stays model-agnostic.
"""

from __future__ import annotations

import os

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Datasets with a `train_pbr` split. HANDAL and HOPE are deliberately held
# out for evaluation, so they are NOT in the default training pool. ycbv and hot3d
# are ALSO excluded: HO3D's objects ARE the YCB objects and HOT3D is itself an eval
# dataset, so training on their renders would break the unseen-object protocol.
DEFAULT_TRAIN_DATASETS = ["hb", "icbin", "itodd", "tless", "tudl"]
HELDOUT_EVAL_DATASETS = ["handal", "hope"]


@dataclass
class BopFlowDatasetConfig:
    bop_root: str = os.environ.get("OMA_DATA_ROOT", "data") + "/bop"
    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_DATASETS))
    split: str = "train_pbr"
    width: int = 560
    height: int = 420
    mask_background: bool = True
    # A target instance must occupy at least this visible fraction in each frame.
    min_visib_fract: float = 0.30
    # Min number of covisible object pixels (native res) for a pair to be kept.
    min_covis_px: int = 2000
    # Relative depth tolerance for the covisibility (occlusion) test.
    depth_rel_tol: float = 0.02
    # Forward-backward cycle-consistency tolerance (pixels). A correspondence is
    # covisible only if it survives this check -- removes occlusion boundaries and
    # depth-discontinuity pixels that the depth test admits.
    cycle_tol_px: float = 1.5
    # Pair sampling: frame-index gap between the two views (in #frames).
    max_frame_gap: int = 30
    min_frame_gap: int = 1
    # Cap on indexed instance groups per dataset, to bound __init__ time.
    max_groups_per_dataset: int | None = None
    seed: int = 0
    # Object-disjoint train/val split (rigorous model selection without touching the
    # HANDAL/HOPE test sets). Each (dataset, obj_id) is deterministically assigned to
    # train or val by a stable hash, so train and val are exact complements and never
    # share an object. 'all' = no split (e.g. for final test datasets).
    obj_split: str = "all"          # 'train' | 'val' | 'all'
    val_obj_fraction: float = 0.2
    obj_split_seed: int = 1234
    # Scene-disjoint (image-disjoint) train/val split -- the "true" validation axis.
    # Each (dataset, scene) is deterministically assigned to train or val by a stable
    # hash, so NO val render is ever seen in training (unlike obj_split, which shares
    # multi-object scene renders across the split). This is the primary val axis for
    # UFM fine-tuning model selection; HANDAL/HOPE stay a pure one-shot test.
    scene_split: str = "all"        # 'train' | 'val' | 'all'
    val_scene_fraction: float = 0.2
    scene_split_seed: int = 1234
    # Covisibility-head training (VGGT covis filter): additionally return the BACKWARD
    # covisibility (frame1 pixels visible in frame0) and per-frame supervision regions
    # (object mask AND valid sensor depth -- real-split depth holes must not become
    # false "not covisible" labels). Off for UFM flow fine-tuning.
    covis_pair_labels: bool = False


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _K(cam: dict) -> np.ndarray:
    return np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)


def _pose_m2c(gt_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.array(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3)
    return R, t


def _snap_unit_factor(raw_ratio: float) -> float:
    """Snap a depth->pose unit ratio to the nearest power of 1000 in {1e-3,1,1e3}.
    `raw_ratio` ~ object_origin_depth(pose units) / median_surface_depth(depth units)."""
    if 0.3 <= raw_ratio <= 3.0:
        return 1.0
    if 3e-4 <= raw_ratio <= 3e-3:
        return 1e-3
    if 3e2 <= raw_ratio <= 3e3:
        return 1e3
    return 1.0      # unknown / object-extent dominated -- assume same units


def relative_camera_pose(Ra: np.ndarray, ta: np.ndarray,
                         Rb: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cam_A -> cam_B from a static object's model->cam pose in both frames."""
    R_AB = Rb @ Ra.T
    t_AB = tb - R_AB @ ta
    return R_AB, t_AB


def compute_flow_and_covisibility(
    depth0: np.ndarray, K0: np.ndarray,
    depth1: np.ndarray, K1: np.ndarray,
    R_AB: np.ndarray, t_AB: np.ndarray,
    depth_rel_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render dense GT flow view0->view1 and a covisibility mask.

    Returns flow (2,H,W) float32, covis (H,W) bool, depthok0 (H,W) bool.
    """
    H, W = depth0.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    z0 = depth0
    depthok0 = z0 > 0

    x = (us - K0[0, 2]) / K0[0, 0] * z0
    y = (vs - K0[1, 2]) / K0[1, 1] * z0
    Xc0 = np.stack([x.ravel(), y.ravel(), z0.ravel()], axis=0)   # (3, N) cam0

    Xc1 = R_AB @ Xc0 + t_AB[:, None]                            # (3, N) cam1
    z1 = Xc1[2]
    eps = 1e-8
    u1 = K1[0, 0] * Xc1[0] / (z1 + eps) + K1[0, 2]
    v1 = K1[1, 1] * Xc1[1] / (z1 + eps) + K1[1, 2]

    u1i, v1i, z1i = u1.reshape(H, W), v1.reshape(H, W), z1.reshape(H, W)
    flow = np.stack([u1i - us, v1i - vs], axis=0).astype(np.float32)

    in_bounds = (u1i >= 0) & (u1i <= W - 1) & (v1i >= 0) & (v1i <= H - 1) & (z1i > 0)
    u1c = np.clip(np.round(u1i), 0, W - 1).astype(np.int64)
    v1c = np.clip(np.round(v1i), 0, H - 1).astype(np.int64)
    depth1_at = depth1[v1c, u1c]
    depth_consistent = np.abs(depth1_at - z1i) < (depth_rel_tol * np.maximum(z1i, eps))

    covis = depthok0 & in_bounds & depth_consistent & (depth1_at > 0)
    return flow, covis, depthok0


def forward_backward_cycle_error(fwd: np.ndarray, bwd: np.ndarray) -> np.ndarray:
    """Per-source-pixel cycle error |p0 - (p0 + fwd -> sample bwd)| in pixels.
    fwd, bwd: (2, H, W) flows for 0->1 and 1->0."""
    H, W = fwd.shape[1:]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    u1 = (xs + fwd[0]).astype(np.float32)
    v1 = (ys + fwd[1]).astype(np.float32)
    bwd_u = cv2.remap(bwd[0], u1, v1, interpolation=cv2.INTER_LINEAR, borderValue=1e6)
    bwd_v = cv2.remap(bwd[1], u1, v1, interpolation=cv2.INTER_LINEAR, borderValue=1e6)
    u0p = u1 + bwd_u
    v0p = v1 + bwd_v
    return np.sqrt((u0p - xs) ** 2 + (v0p - ys) ** 2)


class BopPbrFlowDataset(Dataset):
    """Object-centric masked flow pairs from BOP-PBR scenes."""

    def __init__(self, config: BopFlowDatasetConfig):
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.bop_root = Path(config.bop_root)
        self.groups: list[dict] = []          # one per (dataset, scene, inst_idx)
        self._build_index()
        if not self.groups:
            raise RuntimeError("No instance groups indexed -- check paths/filters.")

    def _scene_dir(self, dataset: str, scene: str) -> Path:
        return self.bop_root / dataset / self.cfg.split / scene

    def _obj_in_split(self, dataset: str, oid: int) -> bool:
        """Deterministic per-object train/val assignment via a stable hash.

        'all' keeps everything. Otherwise object `oid` of `dataset` is a val object iff
        its hash falls in the bottom `val_obj_fraction` of [0,1); train is the exact
        complement. Uses hashlib (not Python's salted hash()) so the split is identical
        across processes/runs -- train and val are guaranteed object-disjoint.
        """
        if self.cfg.obj_split == "all":
            return True
        key = f"{dataset}:{oid}:{self.cfg.obj_split_seed}".encode()
        bucket = (int(hashlib.md5(key).hexdigest(), 16) % 10_000) / 10_000.0
        is_val = bucket < self.cfg.val_obj_fraction
        return is_val if self.cfg.obj_split == "val" else not is_val

    def _scene_in_split(self, dataset: str, scene: str) -> bool:
        """Deterministic per-scene train/val assignment via a stable hash.

        The "true" (image-disjoint) validation axis: 'all' keeps everything, otherwise
        scene `scene` of `dataset` is a val scene iff its hash falls in the bottom
        `val_scene_fraction` of [0,1); train is the exact complement. Because BOP-PBR
        scenes are multi-object renders, splitting at the scene level guarantees no val
        render (pixels/background/lighting) is ever seen in training.
        """
        if self.cfg.scene_split == "all":
            return True
        key = f"{dataset}:{scene}:{self.cfg.scene_split_seed}".encode()
        bucket = (int(hashlib.md5(key).hexdigest(), 16) % 10_000) / 10_000.0
        is_val = bucket < self.cfg.val_scene_fraction
        return is_val if self.cfg.scene_split == "val" else not is_val

    def _build_index(self):
        for dataset in self.cfg.datasets:
            split_dir = self.bop_root / dataset / self.cfg.split
            if not split_dir.is_dir():
                continue
            scenes = sorted([p.name for p in split_dir.iterdir() if p.name.isdigit()])
            n_groups_ds = 0
            for scene in scenes:
                if not self._scene_in_split(dataset, scene):   # image-disjoint train/val
                    continue
                sdir = split_dir / scene
                try:
                    scene_gt = _load_json(sdir / "scene_gt.json")
                    scene_gt_info = _load_json(sdir / "scene_gt_info.json")
                except FileNotFoundError:
                    continue
                # `scene_gt` instance ordering/count is NOT guaranteed stable across
                # frames (it is for tless, but NOT for ycbv etc.). So we identify a
                # physical instance by an obj_id that is UNIQUE within a frame: then
                # its per-frame instance index is unambiguous and refers to the same
                # static object. obj_id -> list of (frame_key, inst_idx).
                obj_frames: dict[int, list[tuple[str, int]]] = {}
                for frame_key, entries in scene_gt.items():
                    infos = scene_gt_info.get(frame_key, [])
                    counts = Counter(e["obj_id"] for e in entries)
                    for inst_idx, entry in enumerate(entries):
                        oid = entry["obj_id"]
                        if counts[oid] != 1:          # ambiguous in this frame
                            continue
                        if inst_idx >= len(infos):
                            continue
                        if infos[inst_idx].get("visib_fract", 0.0) < self.cfg.min_visib_fract:
                            continue
                        obj_frames.setdefault(oid, []).append((frame_key, inst_idx))
                for oid, frames in obj_frames.items():
                    if len(frames) < 2:
                        continue
                    if not self._obj_in_split(dataset, oid):   # object-disjoint train/val
                        continue
                    self.groups.append({
                        "dataset": dataset, "scene": scene, "obj_id": oid,
                        "frames": sorted(frames, key=lambda fi: int(fi[0])),
                    })
                    n_groups_ds += 1
                    if (self.cfg.max_groups_per_dataset is not None
                            and n_groups_ds >= self.cfg.max_groups_per_dataset):
                        break
                if (self.cfg.max_groups_per_dataset is not None
                        and n_groups_ds >= self.cfg.max_groups_per_dataset):
                    break

    def __len__(self):
        return len(self.groups)

    @staticmethod
    def _imread(p: Path, flags: int) -> np.ndarray:
        """cv2.imread that reports a failure instead of returning None.

        A truncated or unreadable PNG makes cv2.imread return None *without*
        raising, and each caller then fails with a different exception type
        depending on what it does with the result: cvtColor raises cv2.error,
        `.astype` raises AttributeError, and `m > 0` raises TypeError. The
        retry window in __getitem__ can only catch what it lists, so the mask
        path (TypeError) escaped it and killed an 8000-step run at step 7350.
        Funnel every read through one FileNotFoundError instead.
        """
        img = cv2.imread(str(p), flags)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {p}")
        return img

    def _read_rgb(self, sdir: Path, frame_key: str) -> np.ndarray:
        for ext in ("jpg", "png"):
            p = sdir / "rgb" / f"{int(frame_key):06d}.{ext}"
            if p.exists():
                return cv2.cvtColor(self._imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"rgb for frame {frame_key} in {sdir}")

    def _read_depth(self, sdir: Path, frame_key: str, depth_scale: float) -> np.ndarray:
        p = sdir / "depth" / f"{int(frame_key):06d}.png"
        return self._imread(p, cv2.IMREAD_UNCHANGED).astype(np.float64) * depth_scale

    def _read_mask(self, sdir: Path, frame_key: str, inst_idx: int) -> np.ndarray:
        p = sdir / "mask_visib" / f"{int(frame_key):06d}_{inst_idx:06d}.png"
        return (self._imread(p, cv2.IMREAD_UNCHANGED) > 0).astype(np.float32)

    def _resize_sample(self, img, flow, covis, valid, objmask):
        W, H = self.cfg.width, self.cfg.height
        H0, W0 = img.shape[:2]
        sx, sy = W / W0, H / H0
        img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        flow_r = cv2.resize(flow.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR)
        flow_r[..., 0] *= sx
        flow_r[..., 1] *= sy
        flow_r = flow_r.transpose(2, 0, 1)
        covis_r = cv2.resize(covis.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        valid_r = cv2.resize(valid.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        obj_r = cv2.resize(objmask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return img_r, flow_r, covis_r, valid_r, obj_r

    def _sample_pair(self, group):
        """Pick an anchor (frame, inst) and a NEARBY (frame, inst) -- small
        baseline so the two views overlap. Each entry carries its own per-frame
        instance index (ordering is not stable across frames)."""
        frames = group["frames"]                     # list of (frame_key, inst_idx)
        anchor = frames[self.rng.randrange(len(frames))]
        na = int(anchor[0])
        candidates = [fi for fi in frames
                      if self.cfg.min_frame_gap <= abs(int(fi[0]) - na) <= self.cfg.max_frame_gap]
        if candidates:
            return anchor, self.rng.choice(candidates)
        nbr = min((fi for fi in frames if fi[0] != anchor[0]),
                  key=lambda fi: abs(int(fi[0]) - na))
        return anchor, nbr

    def __getitem__(self, idx):
        for attempt in range(8):
            g = self.groups[(idx + attempt) % len(self.groups)]
            # A single missing/unreadable rgb|depth|mask PNG (cv2.imread -> None) must not
            # kill a long run: treat any IO/decoding failure as a skipped pair and try the
            # next group. The 8-attempt window makes an all-empty sample vanishingly rare.
            try:
                out = self._try_build(g)
            except (cv2.error, AttributeError, FileNotFoundError, OSError, ValueError):
                out = None
            if out is not None:
                return out
        W, H = self.cfg.width, self.cfg.height
        z = torch.zeros
        out = {
            "img0": z(3, H, W), "img1": z(3, H, W), "gt_flow": z(2, H, W),
            "covisibility": z(H, W), "flow_valid": z(H, W), "obj_mask0": z(H, W),
            "meta": {"dataset": g["dataset"], "scene": g["scene"],
                     "obj_id": g["obj_id"], "empty": True},
        }
        if self.cfg.covis_pair_labels:
            out.update({"covisibility1": z(H, W), "obj_mask1": z(H, W),
                        "region0": z(H, W), "region1": z(H, W)})
        return out

    def _try_build(self, g):
        sdir = self._scene_dir(g["dataset"], g["scene"])
        scene_cam = _load_json(sdir / "scene_camera.json")
        scene_gt = _load_json(sdir / "scene_gt.json")

        (fa, ia), (fb, ib) = self._sample_pair(g)
        # Sanity: both endpoints must really be this obj_id at their own index.
        if (scene_gt[fa][ia]["obj_id"] != g["obj_id"]
                or scene_gt[fb][ib]["obj_id"] != g["obj_id"]):
            return None

        K0, K1 = _K(scene_cam[fa]), _K(scene_cam[fb])
        ds0 = scene_cam[fa].get("depth_scale", 1.0)
        ds1 = scene_cam[fb].get("depth_scale", 1.0)
        Ra, ta = _pose_m2c(scene_gt[fa][ia])
        Rb, tb = _pose_m2c(scene_gt[fb][ib])
        R_AB, t_AB = relative_camera_pose(Ra, ta, Rb, tb)
        R_BA, t_BA = relative_camera_pose(Rb, tb, Ra, ta)

        rgb0 = self._read_rgb(sdir, fa).astype(np.float32) / 255.0
        rgb1 = self._read_rgb(sdir, fb).astype(np.float32) / 255.0
        depth0 = self._read_depth(sdir, fa, ds0)
        depth1 = self._read_depth(sdir, fb, ds1)
        mask0 = self._read_mask(sdir, fa, ia)
        mask1 = self._read_mask(sdir, fb, ib)

        # Reconcile depth vs pose units (e.g. HOT3D depth is mm but cam_t_m2c is m).
        # Snap the depth->pose-unit factor to a power of 10 using the object's
        # origin depth vs its median surface depth.
        on0 = mask0 > 0.5
        if on0.sum() > 0 and (depth0[on0] > 0).any():
            med = float(np.median(depth0[on0 & (depth0 > 0)]))
            factor = _snap_unit_factor(abs(float(ta[2])) / max(med, 1e-9))
            if factor != 1.0:
                depth0 = depth0 * factor
                depth1 = depth1 * factor

        flow, covis, depthok0 = compute_flow_and_covisibility(
            depth0, K0, depth1, K1, R_AB, t_AB, self.cfg.depth_rel_tol)
        # Forward-backward consistency: keep only geometrically verified pixels.
        bwd, covis_bwd, depthok1 = compute_flow_and_covisibility(
            depth1, K1, depth0, K0, R_BA, t_BA, self.cfg.depth_rel_tol)
        cyc = forward_backward_cycle_error(flow, bwd)
        covis = covis & (cyc < self.cfg.cycle_tol_px)

        covis_obj = covis & (mask0 > 0.5)
        if int(covis_obj.sum()) < self.cfg.min_covis_px:
            return None

        if self.cfg.mask_background:
            rgb0 = rgb0 * mask0[..., None]
            rgb1 = rgb1 * mask1[..., None]

        valid = covis_obj.astype(np.float32)
        covis_gt = covis_obj.astype(np.float32)

        img0_r, flow_r, covis_r, valid_r, obj_r = self._resize_sample(
            rgb0, flow, covis_gt, valid, mask0)
        img1_r = cv2.resize(rgb1, (self.cfg.width, self.cfg.height),
                            interpolation=cv2.INTER_LINEAR)

        out = {
            "img0": torch.from_numpy(img0_r.transpose(2, 0, 1)).float(),
            "img1": torch.from_numpy(img1_r.transpose(2, 0, 1)).float(),
            "gt_flow": torch.from_numpy(flow_r).float(),
            "covisibility": torch.from_numpy(covis_r).float(),
            "flow_valid": torch.from_numpy(valid_r).float(),
            "obj_mask0": torch.from_numpy(obj_r).float(),
            "meta": {"dataset": g["dataset"], "scene": g["scene"], "obj_id": g["obj_id"],
                     "frame0": fa, "frame1": fb, "inst0": ia, "inst1": ib, "empty": False},
        }
        if self.cfg.covis_pair_labels:
            # Backward direction (frame1 pixels visible in frame0), cycle-verified the
            # same way, plus per-frame supervision regions = object mask AND valid depth
            # (sensor depth holes are "unknown", not "occluded").
            cyc_b = forward_backward_cycle_error(bwd, flow)
            covis1_obj = covis_bwd & (cyc_b < self.cfg.cycle_tol_px) & (mask1 > 0.5)
            region0 = (mask0 > 0.5) & depthok0
            region1 = (mask1 > 0.5) & depthok1
            WH = (self.cfg.width, self.cfg.height)
            for key, arr in (("covisibility1", covis1_obj), ("obj_mask1", mask1),
                             ("region0", region0), ("region1", region1)):
                r = cv2.resize(arr.astype(np.float32), WH, interpolation=cv2.INTER_NEAREST)
                out[key] = torch.from_numpy(r).float()
        return out


def collate_drop_meta(batch):
    metas = [b.pop("meta") for b in batch]
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["meta"] = metas
    return out
