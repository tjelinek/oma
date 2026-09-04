# Dataset Formats & Comparison Methods

Reference for the dataset layouts the OMa entry points read, and for the external
methods OMa is compared against.

---

## Table of Contents

1. [BOP Standard Format](#1-bop-standard-format)
2. [HANDAL](#2-handal)
3. [HO3D](#3-ho3d)
4. [NAVI](#4-navi)
5. [BOP Classic (T-LESS, LM-O, IC-BIN)](#5-bop-classic-t-less-lm-o-ic-bin)
6. [HOPE](#6-hope)
7. [Google Scanned Objects](#7-google-scanned-objects)
8. [Mast3r / Dust3r](#8-mast3r--dust3r)
9. [VGGT](#9-vggt)
10. [MapAnything](#10-mapanything)

---
## 1. BOP Standard Format

Datasets following the [BOP benchmark](https://bop.felk.cvut.cz/datasets/) convention share a common
folder layout and annotation schema. HANDAL, HOPE, T-LESS, LM-O, and IC-BIN all use this format
(with dataset-specific extensions).

### 1.1 Folder Layout

```
<bop_root>/<dataset>/
    <split>/                            # train, train_primesense, train_pbr, test, val, ...
        <scene_id>/                     # 6-digit zero-padded: 000001, 000002, ...
            rgb/
                000000.png              # or .jpg; 6-digit zero-padded frame ID
                000001.png
                ...
            depth/                      # 16-bit PNG depth maps
                000000.png
                ...
            mask/                       # full projected object masks (optional)
                000000_000000.png       # {frame_id}_{object_instance_index}.png
            mask_visib/                 # visible-part masks
                000000_000000.png
            scene_gt.json               # GT object poses per frame
            scene_camera.json           # camera intrinsics (+ optional extrinsics) per frame
            scene_gt_info.json          # bboxes, visibility fractions
    models/
        obj_000001.ply                  # one PLY mesh per object, units: mm
        ...
        models_info.json                # diameters, bounding box dimensions
    test_targets_bop19.json             # or val_targets_bop24.json, test_targets_bop24.json
```

For HOPE/HANDAL, additional onboarding directories:

```
    onboarding_static/
        <obj_id>_up/                    # turntable, camera above
        <obj_id>_down/                  # turntable, camera below
    onboarding_dynamic/
        <obj_id>/                       # hand-held, free motion
```

### 1.2 `scene_gt.json` -- Object Poses

```json
{
  "0": [
    {
      "obj_id": 1,
      "cam_R_m2c": [r00, r01, r02, r10, r11, r12, r20, r21, r22],
      "cam_t_m2c": [tx, ty, tz]
    }
  ]
}
```

| Field | Format | Description |
|-------|--------|-------------|
| Key | String frame ID (`"0"`, `"1"`, ...) | Matches image filename stem |
| `obj_id` | int | Object identifier (dataset-local) |
| `cam_R_m2c` | 9 floats, row-major 3x3 | Rotation: **model-to-camera** |
| `cam_t_m2c` | 3 floats | Translation: **model-to-camera**, in **mm** |

Multiple objects per frame are stored as a list. OMa loads this in
`utils/bop_challenge.py:read_obj2cam_Se3_from_gt()`.

### 1.3 `scene_camera.json` -- Camera Parameters

```json
{
  "0": {
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "depth_scale": 0.1,
    "cam_R_w2c": [r00, ..., r22],
    "cam_t_w2c": [tx, ty, tz],
    "width": 640,
    "height": 480
  }
}
```

| Field | Format | Description |
|-------|--------|-------------|
| `cam_K` | 9 floats, row-major 3x3 | Pinhole intrinsic matrix |
| `depth_scale` | float | Multiplier: raw depth pixel value x depth_scale = mm |
| `cam_R_w2c` | 9 floats (optional) | World-to-camera rotation (onboarding sequences only) |
| `cam_t_w2c` | 3 floats (optional) | World-to-camera translation in mm (onboarding sequences only) |
| `width`, `height` | int (optional) | Image dimensions |

`width`/`height` may be absent (OMa falls back to 0). `cam_R_w2c`/`cam_t_w2c` are only present
in onboarding sequences, not in standard BOP test/val splits.

OMa loads intrinsics in `utils/bop_challenge.py:get_pinhole_params()` and world-to-camera
extrinsics in `read_gt_Se3_world2cam()`.

### 1.4 `scene_gt_info.json` -- Visibility & Bounding Boxes

```json
{
  "0": [
    {
      "bbox_obj": [x, y, w, h],
      "bbox_visib": [x, y, w, h],
      "px_count_all": 1234,
      "px_count_valid": 1200,
      "px_count_visib": 1100,
      "visib_fract": 0.89
    }
  ]
}
```

OMa does **not** read this file directly.

### 1.5 Test Targets Files

```json
[
  {"im_id": 0, "scene_id": 1, "obj_id": 5, "inst_count": 1},
  ...
]
```

`obj_id` and `inst_count` are optional. OMa groups by `(im_id, scene_id)` in
`utils/bop_challenge.py:group_test_targets_by_image()`.

### 1.6 Object Models

PLY meshes at `models/obj_NNNNNN.ply`:
- Binary little-endian PLY with vertices (float xyz + uchar rgba) and triangle faces
- Units: **millimeters**
- `models_info.json` provides per-object `diameter`, `min_x/y/z`, `size_x/y/z` (all in mm)

### 1.7 Depth Maps

- 16-bit unsigned int PNG
- Raw value x `depth_scale` (from `scene_camera.json`) = depth in **mm**
- OMa applies an additional `depth_scale_to_meter` conversion (see per-dataset sections)

### 1.8 Image Naming

- RGB: `{frame_id:06d}.png` or `.jpg` (OMa checks both; see `pose_estimator.py:312`)
- Masks: `{frame_id:06d}_{object_instance_index:06d}.png`
- Depth: `{frame_id:06d}.png`
- Frame IDs may be non-contiguous (especially in HANDAL static sequences)

### 1.9 Coordinate System Conventions

- **Object coordinates**: defined by PLY model, origin at model center, units in mm
- **Camera coordinates**: standard CV convention (x-right, y-down, z-forward)
- **`cam_R_m2c`, `cam_t_m2c`**: transform points from object frame to camera frame
- **`cam_R_w2c`, `cam_t_w2c`**: transform points from world frame to camera frame

OMa conversion chain (in `utils/bop_challenge.py`):
1. `read_obj2cam_Se3_from_gt()` -> `Se3(R_m2c, t_m2c)` (object-to-camera)
2. `extract_gt_Se3_cam2obj()` -> invert + scale -> camera-to-object
3. Run scripts invert again -> `gt_Se3_world2cam` (since "world" = object frame in onboarding)

---

## 2. HANDAL

40 hand-held objects across 17 categories. Available in both native and BOP format.

### 2.1 Native Format

```
HANDAL/
    handal_dataset_<category>/          # 17 categories (mugs, hammers, spatulas, ...)
        models/
            models_info.json
            obj_000001.ply              # per-object PLY, units: mm
            ...
        models_parts/                   # HANDAL-specific
            obj_000001_handle.ply
            obj_000001_not.ply
        train/
            <sequence_id>/              # e.g., 001001 (first 3 digits = obj_id)
                rgb/                    # {frame_id:06d}.jpg
                mask/                   # {frame_id:06d}_{obj_idx:06d}.png
                mask_visib/
                mask_parts/             # HANDAL-specific: {fid}_{oid}_handle.png
                scene_gt.json
                scene_camera.json
                scene_gt_info.json
        test/
            ...
        dynamic/                        # present in ~10 of 17 categories
            002999_train/               # obj_id 002, "999" = dynamic marker
            ...
```

**No `depth/` directory** in native HANDAL.

### 2.2 BOP Format

```
bop/handal/
    onboarding_static/
        obj_NNNNNN_up/
        obj_NNNNNN_down/
    onboarding_dynamic/
        obj_NNNNNN/
    val/
        <scene_id>/
    test/
        <scene_id>/
    models/
        obj_000001.ply ... obj_000040.ply
    val_targets_bop24.json
    test_targets_bop24.json
```

### 2.3 Sequence Naming

| Format | Example | Meaning |
|--------|---------|---------|
| Native | `handal_dataset_mugs@001001` | Category `mugs`, obj 001, sequence 001 |
| BOP onboarding | `obj_000001_up` | Object 1, static upper turntable |
| BOP onboarding | `obj_000001_dynamic` | Object 1, hand-held dynamic |
| BOP val | `000001_000005` | Scene 1, object 5 |

### 2.4 Key Parameters

| Property | Static sequences | Dynamic sequences |
|----------|-----------------|-------------------|
| Resolution | 1920x1440 | 640x480 |
| Typical fx, fy | ~1590, ~1589 | ~567, ~567 |
| Frame IDs | Non-contiguous (0, 8, 15, 22, ...) | Consecutive (0, 1, 2, ...) |
| Frames/sequence | ~124-133 | ~400-522 |
| `depth_scale_to_meter` | 0.001 | 0.001 |
| `image_downsample` | 0.5 | 0.5 |
| `similarity_transformation` | `'kabsch'` | `'depths'` |

### 2.5 Object ID Scope

Object IDs are **not globally unique** across native categories (each category starts at 1).
In BOP format, objects are globally numbered `obj_000001` through `obj_000040`.

### 2.6 OMa Loading

- **Native**: `run_HANDAL.py` -- splits on `@`, sets `cam_scale=1.0`, `image_downsample=0.5`
- **BOP onboarding**: `run_bop_HANDAL_onboarding.py` -> `set_config_for_bop_onboarding()` ->
  `run_on_bop_sequences()`

---

## 3. HO3D

Hand-Object 3D dataset v3. 9 YCB objects manipulated by hands, captured with multi-camera Kinect rig.

### 3.1 Folder Layout

```
HO3D/
    train/                              # 55 sequences
        <sequence_name>/                # e.g., ABF10, MC1, SM2
            rgb/                        # {frame_id:04d}.jpg (4-digit, 640x480)
            depth/                      # {frame_id:04d}.png (16-bit, encoded)
            meta/                       # {frame_id:04d}.pkl and .npz
            seg/                        # {frame_id:04d}.png (320x240, half-res)
    evaluation/                         # 13 sequences
        <sequence_name>/
            rgb/ depth/ meta/
            segmentation_rendered/      # OMa-generated (no seg/ in eval split)
    models/                             # YCB object meshes
        003_cracker_box/
            textured.obj                # full mesh (~51MB)
            textured_simple.obj         # simplified mesh (~1.6MB)
            texture_map.png
            points.xyz                  # point cloud
        ...
    calibration/                        # multi-camera extrinsics (v3)
```

### 3.2 Annotation Format (`.pkl` meta files)

Each frame has a pickle dict with:

| Field | Shape | Description |
|-------|-------|-------------|
| `camMat` | `(3, 3)` float64 | Camera intrinsic matrix |
| `objRot` | `(3, 1)` or `(3,)` float32 | Rodrigues axis-angle rotation (obj-to-cam) |
| `objTrans` | `(3,)` float32 | Translation in **meters** (obj-to-cam) |
| `objName` | str | YCB object name, e.g., `'021_bleach_cleanser'` |
| `objLabel` | int | YCB numeric label |
| `objCorners3D` | `(8, 3)` float64 | 3D bbox in current camera frame (meters) |
| `objCorners3DRest` | `(8, 3)` float64 | 3D bbox in object rest pose (meters) |
| `handPose` | `(48,)` float32 | MANO hand pose (train only) |
| `handTrans` | `(3,)` float32 | Hand translation (train only) |
| `handBeta` | `(10,)` float32 | MANO shape params (train only) |
| `handJoints3D` | `(21, 3)` float64 | 3D hand joints (train); `(3,)` (eval) |

Evaluation split omits `handPose`, `handTrans`, `handBeta` and related contact fields.
`objRot` shape is inconsistent across sequences -- OMa uses `.squeeze()`.

### 3.3 Key Differences from BOP

| Aspect | HO3D | BOP standard |
|--------|------|--------------|
| **Annotations** | Per-frame `.pkl` files | `scene_gt.json` + `scene_camera.json` |
| **Rotation format** | Rodrigues axis-angle (3D) | 3x3 rotation matrix (flat 9 elements) |
| **Translation units** | **Meters** | **Millimeters** |
| **Image naming** | 4-digit zero-padded `.jpg` | 6-digit zero-padded `.png`/`.jpg` |
| **Image resolution** | 640x480 | Varies |
| **Model units** | **Meters** (YCB) | **Millimeters** |
| **Depth encoding** | 2-channel: `(ch2 + ch1*256) * 0.00012498664727900177` | 16-bit uint * depth_scale |

### 3.4 OMa Loading

`run_HO3D.py`:
- Iterates `.pkl` files in `meta/`, loads `camMat`, `objRot`, `objTrans`
- Converts `objRot` to quaternion via `Quaternion.from_axis_angle()`
- **Multiplies translations by 1000** (meters -> mm, to match OMa's internal mm convention)
- `skip_indices *= 10` (sequences are ~1000-1700 frames)
- Segmentation from `seg/` folder (channel 1 = green), falls back to `segmentation_rendered/`

### 3.5 Objects and Sequences

9 YCB objects appear: `003_cracker_box`, `004_sugar_box`, `006_mustard_bottle`,
`010_potted_meat_can`, `011_banana`, `019_pitcher_base`, `021_bleach_cleanser`, `025_mug`,
`035_power_drill`, `037_scissors`.

Sequence naming: `{subject_prefix}{camera_id}` (e.g., `ABF10` = subject ABF, camera 0).
55 train sequences, 13 evaluation sequences.

---

## 4. NAVI

Novel View synthesis and Appearance capture for object Instance recognition. 36 household objects,
video sequences with per-frame poses from COLMAP.

### 4.1 Folder Layout

```
NAVI/navi_v1.5/
    <object_name>/                      # 36 objects
        3d_scan/
            <object_name>.obj           # GT mesh (OBJ, large ~50-200MB)
            <object_name>.mtl
            <object_name>.jpg           # texture
            <object_name>.glb           # GLB format
        video-NN-<camera>-<video_id>/   # 136 video sequences total
            annotations.json
            images/                     # frame_NNNNN.jpg (5-digit, non-contiguous)
            masks/                      # frame_NNNNN.png (binary, palette mode)
            depth/                      # frame_NNNNN.png (uint16, mm)
            video.mp4
        multiview-NN-<camera>/          # 324 multiview sequences (not used by OMa)
        wild_set/                       # 35 wild sets (not used by OMa)
    custom_splits/
```

### 4.2 `annotations.json`

JSON array, one entry per frame:

```json
{
  "object_id": "3d_dollhouse_sink",
  "camera": {
    "q": [w, x, y, z],
    "t": [tx, ty, tz],
    "focal_length": 2654.89,
    "camera_model": "canon_t4i"
  },
  "filename": "frame_00000.jpg",
  "image_size": [1080, 1920],
  "split": "train",
  "occluded": false
}
```

| Field | Description |
|-------|-------------|
| `camera.q` | Quaternion `[w, x, y, z]` (Hamilton), **world-to-camera** rotation |
| `camera.t` | Translation `[tx, ty, tz]`, **world-to-camera**, in **mm** |
| `camera.focal_length` | Single focal length in pixels (fx = fy) |
| `image_size` | `[height, width]` |
| `occluded` | Per-frame occlusion flag (~7.6% of frames) |

Principal point is assumed at image center: `cx = width/2`, `cy = height/2`.

### 4.3 Key Parameters

| Property | Value |
|----------|-------|
| Resolution | 1080x1920 (portrait) or 1920x1080 (landscape) |
| Focal length range | 1577-3434 px (varies by camera + zoom) |
| Depth format | uint16 PNG, values in mm, zero = no depth |
| Model format | OBJ + texture, scan-quality |
| Translation units | mm |
| Frames per sequence | ~40-200 |
| Frame naming | `frame_NNNNN.jpg` (non-contiguous, sampled at ~15 frame intervals) |

### 4.4 OMa Loading

`run_NAVI.py`:
- Sequence format: `<object>@video-NN-<camera>-<video_id>`
- Only `video-*` sequences are discovered (not multiview or wild_set)
- Quaternion loaded directly into `kornia.geometry.Quaternion`
- Frame dicts reindexed to contiguous 0-based indices
- No `depth_scale_to_meter` conversion (already in mm)

---

## 5. BOP Classic (T-LESS, LM-O, IC-BIN)

These follow the standard BOP format (section 1) with dataset-specific quirks.

### 5.1 T-LESS (Texture-Less Objects)

```
bop/tless/
    train_primesense/                   # real sensor data for onboarding
    test_primesense/                    # test split
    models/
        obj_000001.ply ... obj_000030.ply   # 30 texture-less industrial parts
```

| Property | Value |
|----------|-------|
| Objects | 30 (texture-less, many with symmetries) |
| Splits | `train_primesense` (onboarding), `test_primesense` (eval) |
| Targets year | `bop19` |
| Templates per object | 491 (fewer than 642, due to symmetries) |
| `depth_scale_to_meter` | 0.001 |
| `skip_indices` | multiplied by 4 |
| Inference downsampling | 1.0 (none) |

### 5.2 LM-O (Linemod Occlusion)

```
bop/lmo/
    train/                              # onboarding
    test/                               # evaluation
    models/
        obj_000001.ply, obj_000005.ply, obj_000006.ply,
        obj_000008.ply ... obj_000012.ply    # 8 objects (non-contiguous IDs)
```

| Property | Value |
|----------|-------|
| Objects | 8 (IDs: 1, 5, 6, 8, 9, 10, 11, 12) |
| Splits | `train`, `test` |
| Targets year | `bop19` |
| Templates per object | 642 |
| `depth_scale_to_meter` | 0.001 |

### 5.3 IC-BIN (Bin-Picking)

```
bop/icbin/
    train/
    test/
    models/
        obj_000001.ply, obj_000002.ply  # 2 objects only
```

| Property | Value |
|----------|-------|
| Objects | 2 |
| Splits | `train`, `test` |
| Targets year | `bop19` |
| Heavy clutter/occlusion | Yes (bin-picking scenario) |

### 5.4 OMa Loading

`run_BOP_classic_onboarding.py`:
- Sequence code: `{dataset}@{split}@{scene_name}` (e.g., `tless@train_primesense@000001`)
- `depth_scale_to_meter = 0.001`, `skip_indices *= 4`
- Loads via `get_bop_images_and_segmentations()`, `read_gt_Se3_cam2obj_transformations()`,
  `read_pinhole_params()`

---

## 6. HOPE

28 household objects, BOP format with onboarding splits.

```
bop/hope/
    onboarding_static/
        obj_NNNNNN_up/
        obj_NNNNNN_down/
    onboarding_dynamic/
        obj_NNNNNN/
    test/
    val/
    models/
        obj_000001.ply ... obj_000028.ply
    test_targets_bop24.json
```

| Property | Value |
|----------|-------|
| Objects | 28 |
| Targets year | `bop24` |
| `image_downsample` | 0.5 |
| `depth_scale_to_meter` | 0.001 |
| Inference downsampling | 0.25 |
| Onboarding types | Static (up/down/both) + dynamic |

OMa loading: `run_HOPE.py` -> `set_config_for_bop_onboarding()` -> `run_on_bop_sequences()`.
Same static/dynamic config switching as HANDAL.

---

## 7. Google Scanned Objects

1000+ high-quality 3D scans. Used for **synthetic rendering** only (no real images).

### 7.1 Folder Layout

```
GoogleScannedObjects/
    models/
        <object_name>/                  # e.g., Squirrel, SCHOOL_BUS
            meshes/
                model.obj               # OBJ mesh
                model.mtl
            materials/textures/
                texture.png
            metadata.pbtxt
            model.config
```

### 7.2 OMa Loading

`run_GoogleScannedObjects.py`:
- Synthetic pipeline: loads mesh via Kaolin, renders images from random viewpoints on a sphere
- Camera at `(0, -5.0, 0)`, up vector `(0, 0, 1)` (Z-up convention)
- Rotation generator: `scenarios.random_walk_on_a_sphere`
- Not a real-image dataset -- used for controlled ablation

---

## 8. Mast3r / Dust3r

Located at `repositories/mast3r/` (Dust3r is a git submodule at `repositories/mast3r/dust3r/`).

**Status**: Repository cloned, **no integration** into OMa code. Adapter planned (CLAUDE.md P3.4).

### 8.1 What It Does

Mast3r (Matching And Stereo 3D Reconstruction) is a ViT-based model that predicts dense 3D point
maps and matching descriptors from image pairs. Multiple pairs are combined via global alignment
to produce a full multi-view reconstruction.

### 8.2 Two Reconstruction Modes

**Mode A: Sparse Global Alignment** (primary, `sparse_global_alignment()`):
1. Create image pairs (`make_pairs()` with scene graph: complete/swin/logwin/retrieval)
2. For each pair, run Mast3r forward pass -> per-pixel 3D points + descriptors + confidence
3. Extract keypoints and correspondences
4. Build minimum spanning tree from pairwise scores
5. Coarse optimization (lr=0.07, 300 iters): camera poses + focals via 3D matching loss
6. Fine refinement (lr=0.01, 300 iters): 2D reprojection error
7. Output: `SparseGA` scene object

**Mode B: GLOMAP/COLMAP Integration** (`colmap/mapping.py`):
1. Run Mast3r matching on pairs -> 2D-2D correspondences
2. Export to COLMAP database
3. Run GLOMAP or pycolmap mapper
4. Output: standard COLMAP reconstruction

### 8.3 API

**Model loading:**
```python
from mast3r.model import AsymmetricMASt3R
model = AsymmetricMASt3R.from_pretrained("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
```

**Input:**
- List of image file paths
- Images preprocessed internally to `[B, C, H, W]` normalized to [-1, 1]

**Output (SparseGA):**
```python
scene.cam2w        # (N, 4, 4) camera-to-world transforms
scene.intrinsics   # list of (3, 3) intrinsic matrices
scene.depthmaps    # per-image depth maps
scene.pts3d        # per-image 3D point arrays
scene.imgs         # RGB images as [H, W, 3] numpy in [0, 1]
scene.get_focals() # focal lengths
```

### 8.4 Key Files for Integration

- `mast3r/cloud_opt/sparse_ga.py` -- `sparse_global_alignment()` main entry point
- `mast3r/image_pairs.py` -- `make_pairs()` for generating image pair lists
- `dust3r/inference.py` -- `inference()` for pairwise forward pass
- `dust3r/utils/image.py` -- `load_images()` for preprocessing

---

## 9. VGGT

Located at `repositories/vggt/`.

**Status**: Repository cloned, **no integration** into OMa code. Adapter planned (CLAUDE.md P3.4).

### 9.1 What It Does

VGGT (Visual Geometry Grounded Transformer, 1B params) is a feed-forward model that jointly predicts
camera poses, depth maps, 3D points, and point tracks from a set of input images in a single
forward pass. No iterative optimization.

### 9.2 Pipeline

1. Load images, center-pad to square, resize to 1024x1024 (load resolution)
2. Model internally processes at 518x518
3. DINOv2 backbone -> alternating frame attention + global attention blocks
4. Camera head: `pose_enc [B, S, 9]` = `[translation(3), quaternion(4), FoV(2)]`
5. Depth head: depth maps `[B, S, H, W, 1]` + confidence
6. Convert pose encoding to extrinsic `[S, 3, 4]` and intrinsic `[S, 3, 3]`
7. Unproject depth to world points
8. Optional bundle adjustment via VGGSfM tracker + pycolmap

### 9.3 API

**Model loading:**
```python
from vggt.models.vggt import VGGT
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
```

**Image preprocessing:**
```python
from vggt.utils.load_fn import load_and_preprocess_images_square
images, original_coords = load_and_preprocess_images_square(image_paths, target_size=1024)
# images: [N, 3, H, W] tensor in [0, 1]
```

**Forward pass output (predictions dict):**

| Key | Shape | Description |
|-----|-------|-------------|
| `pose_enc` | `[B, S, 9]` | Camera pose encoding |
| `depth` | `[B, S, H, W, 1]` | Depth maps |
| `depth_conf` | `[B, S, H, W]` | Depth confidence |
| `world_points` | `[B, S, H, W, 3]` | 3D world coordinates per pixel |
| `world_points_conf` | `[B, S, H, W]` | World point confidence |

**Pose conversion:**
```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
# extrinsics: [B, S, 3, 4] camera-from-world (OpenCV convention)
# intrinsics: [B, S, 3, 3]
```

### 9.4 Key Files for Integration

- `vggt/models/vggt.py` -- `VGGT` model class
- `vggt/utils/load_fn.py` -- `load_and_preprocess_images_square()`
- `vggt/utils/pose_enc.py` -- `pose_encoding_to_extri_intri()`
- `vggt/utils/geometry.py` -- `unproject_depth_map_to_point_map()`
- `demo_colmap.py` -- full pipeline reference (images -> COLMAP reconstruction)

---

## 10. MapAnything

**Status**: **Not present** in the repository. Not cloned, no submodule, no code references
outside of CLAUDE.md TODO items. Needs to be added as a submodule or installed before any
integration work.

---

## Quick Reference: Translation Units & Pose Conventions

| Dataset | Translation units | Pose semantics | Rotation format | Model units |
|---------|------------------|----------------|-----------------|-------------|
| BOP (all) | mm | obj-to-cam (`cam_R_m2c`) | 3x3 flat | mm |
| HANDAL | mm | obj-to-cam | 3x3 flat | mm |
| HO3D | **meters** (x1000 in OMa) | obj-to-cam | Rodrigues 3D | **meters** |
| NAVI | mm | world-to-cam | quaternion wxyz | - |
| HOPE | mm | obj-to-cam | 3x3 flat | mm |

## Quick Reference: OMa `depth_scale_to_meter`

| Dataset | Value | Meaning |
|---------|-------|---------|
| HANDAL | 0.001 | raw mm -> meters |
| HOPE | 0.001 | raw mm -> meters |
| T-LESS, LM-O, IC-BIN | 0.001 | raw mm -> meters |
| HO3D | special | 2-channel decode * 0.00012498664727900177 |

## Quick Reference: Image Resolution & Naming

| Dataset | Resolution | Naming | Extension |
|---------|-----------|--------|-----------|
| HANDAL static | 1920x1440 | `{:06d}` | .jpg |
| HANDAL dynamic | 640x480 | `{:06d}` | .jpg |
| HO3D | 640x480 | `{:04d}` | .jpg |
| NAVI | 1080x1920 | `frame_{:05d}` | .jpg |
| BOP classic | varies | `{:06d}` | .png |