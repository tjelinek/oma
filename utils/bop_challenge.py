import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, OrderedDict, Any

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera
from tqdm import tqdm

from configs.glopose_config import GloPoseConfig
from utils.data_utils import get_scale_from_meter, get_scale_to_meter
from utils.math_utils import scale_Se3


def _hot3d_camera_suffix(hot3d_device: str | None) -> str:
    """Return the camera-type suffix for HOT3D file/folder names ('rgb' for Aria, 'gray1' for Quest3)."""
    if hot3d_device == 'quest3':
        return 'gray1'
    return 'rgb'


def _scene_gt_filename(dataset: str, hot3d_device: str | None = None) -> str:
    """Return the scene GT JSON filename for a dataset."""
    if dataset == 'hot3d':
        return f'scene_gt_{_hot3d_camera_suffix(hot3d_device)}.json'
    return 'scene_gt.json'


def _scene_camera_filename(dataset: str, hot3d_device: str | None = None) -> str:
    """Return the scene camera JSON filename for a dataset."""
    if dataset == 'hot3d':
        return f'scene_camera_{_hot3d_camera_suffix(hot3d_device)}.json'
    return 'scene_camera.json'


def _mask_visib_folder_name(dataset: str, hot3d_device: str | None = None) -> str:
    """Return the mask visibility folder name for a dataset."""
    if dataset == 'hot3d':
        return f'mask_visib_{_hot3d_camera_suffix(hot3d_device)}'
    return 'mask_visib'


def _image_folder_name(dataset: str, hot3d_device: str | None = None) -> str:
    """Return the image subfolder name ('rgb' for most datasets, 'gray1' for Quest3)."""
    if dataset == 'hot3d':
        return _hot3d_camera_suffix(hot3d_device)
    return 'rgb'


def get_pinhole_params_from_hot3d(json_file_path: Path, scale: float = 1.0, device='cpu') -> Dict[int, PinholeCamera]:
    """Extract approximate pinhole parameters from HOT3D fisheye camera JSON.

    HOT3D uses FISHEYE624 cameras. This extracts focal length and principal point
    from projection_params as a pinhole approximation (ignoring distortion).
    """
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    pinhole_cameras = {}
    for frame_str, frame_data in json_data.items():
        frame_int = int(frame_str)
        cam_model = frame_data['cam_model']
        params = cam_model['projection_params']
        w = cam_model['image_width']
        h = cam_model['image_height']

        # Approximate pinhole from fisheye: focal_length, cx, cy
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]

        cam_K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

        cam_w2c = Se3.identity(device=device).matrix()
        width = torch.tensor(w, dtype=torch.float32, device=device)
        height = torch.tensor(h, dtype=torch.float32, device=device)

        pinhole_camera = PinholeCamera(cam_K.unsqueeze(0), cam_w2c.unsqueeze(0),
                                       height.unsqueeze(0), width.unsqueeze(0))
        pinhole_camera = pinhole_camera.scale(torch.tensor(scale, device=device).unsqueeze(0))
        pinhole_cameras[frame_int] = pinhole_camera

    return pinhole_cameras


def get_pinhole_params(json_file_path: Path, scale: float = 1.0, device='cpu') -> Dict[int, PinholeCamera]:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    no_image_shape_available = False
    pinhole_cameras = {}
    for frame_str, value in json_data.items():
        frame_int = int(frame_str)
        frame_data = json_data[frame_str]
        cam_K = torch.tensor(frame_data['cam_K'], dtype=torch.float32, device=device).view(3, 3)
        cam_w2c = Se3.identity(device=device).matrix()

        if frame_data.get('width') is not None and frame_data.get('height') is not None:
            width = torch.tensor(frame_data['width'], dtype=torch.float32, device=device)
            height = torch.tensor(frame_data['height'], dtype=torch.float32, device=device)
        else:
            width, height = torch.tensor(0., device=device), torch.tensor(0., device=device)
            no_image_shape_available = True

        pinhole_camera = PinholeCamera(cam_K.unsqueeze(0), cam_w2c.unsqueeze(0),
                                       height.unsqueeze(0), width.unsqueeze(0))
        pinhole_camera = pinhole_camera.scale(torch.tensor(scale, device=device).unsqueeze(0))

        pinhole_cameras[frame_int] = pinhole_camera

    if no_image_shape_available:
        warnings.warn(f"The image shape is not available in the {json_file_path} file. ")

    return pinhole_cameras


def read_obj2cam_Se3_from_gt(pose_json_path, device: str) -> Dict[int, Dict[int, Se3]]:
    dict_gt_Se3_obj2cam = defaultdict(dict)
    with open(pose_json_path, 'r') as file:
        pose_json = json.load(file)
        for frame, data in pose_json.items():
            frame = int(frame)
            for entry in data:
                obj_id = entry['obj_id']
                R_obj_to_cam = entry['cam_R_m2c']
                R_m2c = torch.tensor(np.array(R_obj_to_cam, dtype=np.float64).reshape(3, 3),
                                     dtype=torch.float32, device=device)

                cam_t_m2c = entry['cam_t_m2c']
                t_m2c = torch.tensor(cam_t_m2c, dtype=torch.float32, device=device)

                gt_Se3_obj2cam = Se3(Quaternion.from_matrix(R_m2c), t_m2c)
                dict_gt_Se3_obj2cam[obj_id][frame] = gt_Se3_obj2cam

    return dict_gt_Se3_obj2cam


def load_gt_images(image_folder: Path):
    """Load ground truth images."""
    gt_images = {
        int(file.stem): file
        for file in sorted(image_folder.iterdir())
        if file.is_file()
    }

    return gt_images


def load_gt_segmentations(segmentation_folder: Path, object_id: int = None):
    """Load segmentation files, filtering by GT annotation index.

    When object_id is None, loads all masks (first mask per frame).
    When object_id is given, it is treated as a GT annotation index suffix —
    only masks whose filename ends with that index are loaded.

    Note: For multi-object val/test scenes, use load_gt_segmentations_by_obj_id()
    instead, which resolves the correct GT index per frame from scene_gt.json.
    """
    object_id_str = f"{object_id:06d}" if object_id is not None else None  # Ensure it's a zero-padded 6-digit string

    gt_segs = {
        int(file.stem.split('_')[0]): file
        for file in sorted(segmentation_folder.iterdir())
        if object_id is None or file.stem.endswith(object_id_str)  # Dynamically filter by object ID
    }

    return gt_segs


def load_gt_segmentations_by_obj_id(segmentation_folder: Path, scene_gt_path: Path,
                                     obj_id: int) -> Dict[int, Path]:
    """Load segmentation masks for a specific object ID.

    BOP mask naming is {frame_id}_{gt_index}.png where gt_index is the 0-based
    position of the object in scene_gt.json[frame_id], NOT the obj_id.
    See: https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md

    This function reads scene_gt.json to find the correct gt_index for the given
    obj_id in each frame, then loads the corresponding mask files.
    """
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)

    gt_segs = {}
    for frame_str, annotations in scene_gt.items():
        frame_id = int(frame_str)
        for gt_index, ann in enumerate(annotations):
            if ann['obj_id'] == obj_id:
                mask_path = segmentation_folder / f'{frame_id:06d}_{gt_index:06d}.png'
                if mask_path.exists():
                    gt_segs[frame_id] = mask_path
                break  # Take only the first instance of this obj_id per frame

    return gt_segs


def get_sequence_folder(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, onboarding_type: str = None,
                        direction: str = None, hot3d_device: str = 'aria'):
    """Returns the sequence folder path based on sequence type and onboarding type."""
    if sequence_type == 'onboarding':

        if dataset == 'hot3d':
            # HOT3D uses device-specific folder names
            base = bop_folder / dataset / f'object_ref_{hot3d_device}_{onboarding_type}_scenewise'
            if direction in ['up', 'down']:
                return base / f'{sequence}_{direction}'
            return base / sequence
        elif onboarding_type == 'dynamic':
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / sequence
        elif onboarding_type == 'static' and direction in ['up', 'down']:
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / f'{sequence}_{direction}'
        else:
            raise ValueError(f'Unknown onboarding type {onboarding_type} or direction {direction}')

    elif sequence_type in ['test', 'val', 'train', 'train_primesense', 'train_pbr']:
        return bop_folder / dataset / sequence_type / sequence
    else:
        raise ValueError(f'Unknown sequence type: {sequence_type}')


def extract_gt_Se3_cam2obj(pose_json_path: Path, scale_factor: float, scene_object_object_id: int = None,
                           object_id: int = None, device: str = 'cpu') -> Dict[int, Se3]:
    dict_gt_Se3_obj2cam = read_obj2cam_Se3_from_gt(pose_json_path, device)

    if scene_object_object_id is not None and object_id is not None:
        raise ValueError("Specify either scene_object_object_id or object_id, not both")
    obj_ids = sorted(dict_gt_Se3_obj2cam.keys())

    if scene_object_object_id is None and object_id is None:
        object_id = obj_ids[0]
    elif scene_object_object_id is not None and object_id is None:
        object_id = obj_ids[scene_object_object_id]
    else:
        if object_id not in obj_ids:
            raise ValueError(f"object_id {object_id} not found in GT poses (available: {obj_ids})")

    dict_gt_Se3_obj2cam = dict_gt_Se3_obj2cam[object_id]
    gt_Se3_obj2cam_frames = dict_gt_Se3_obj2cam.keys()
    gt_Se3_cam2obj = {frame: scale_Se3(dict_gt_Se3_obj2cam[frame].inverse(), scale_factor)
                      for frame in gt_Se3_obj2cam_frames}

    return gt_Se3_cam2obj


def extract_object_id(pose_json_path: Path, scene_object_object_id: int = None) -> Dict[int, int]:
    dict_gt_Se3_obj2cam = read_obj2cam_Se3_from_gt(pose_json_path, 'cpu')

    obj_ids = sorted(dict_gt_Se3_obj2cam.keys())

    if scene_object_object_id is None:
        object_id = obj_ids[0]
    else:
        object_id = obj_ids[scene_object_object_id]

    return {1: object_id}


def load_static_onboarding_parts(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str,
        static_onboarding_sequence: Optional[str],
        loader_fn: Callable[[Path], Optional[dict]],
        sequence_starts: Optional[List[int]] = None,
        hot3d_device: str = 'aria',
) -> dict:
    folder_down = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'down',
                                       hot3d_device=hot3d_device)
    folder_up = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'up',
                                     hot3d_device=hot3d_device)

    data_down = loader_fn(folder_down) if folder_down.exists() else {}
    data_up = loader_fn(folder_up) if folder_up.exists() else {}

    if static_onboarding_sequence == 'both':
        assert data_down is not None and data_up is not None and sequence_starts is not None
        merged_data = data_down.copy()
        merged_data.update({k + sequence_starts[1]: v for k, v in data_up.items()})
        return merged_data
    elif static_onboarding_sequence == 'down':
        assert data_down is not None
        return data_down
    elif static_onboarding_sequence == 'up':
        assert data_up is not None
        return data_up
    else:
        raise ValueError(f'Unknown static onboarding sequence type: {static_onboarding_sequence}')


def get_bop_images_and_segmentations(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str = None,
        static_onboarding_sequence: Optional[str] = None,
        scene_obj_id: int = None,
        hot3d_device: str = 'aria',
) -> Tuple[Dict[int, Path], Dict[int, Path], Optional[Dict[int, Path]], Optional[List[int]]]:
    """Loads images and segmentations from BOP dataset based on sequence type."""
    sequence_starts = [0]
    mask_folder = _mask_visib_folder_name(dataset, hot3d_device)
    img_folder = _image_folder_name(dataset, hot3d_device)

    if sequence_type == 'onboarding' and onboarding_type == 'static' and static_onboarding_sequence is not None:
        down_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'down',
                                           hot3d_device=hot3d_device)
        up_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'up',
                                         hot3d_device=hot3d_device)

        images_down = load_gt_images(down_folder / img_folder) if down_folder.exists() else {}
        images_up = load_gt_images(up_folder / img_folder) if up_folder.exists() else {}
        segs_down = load_gt_segmentations(down_folder / mask_folder) if down_folder.exists() else {}
        segs_up = load_gt_segmentations(up_folder / mask_folder) if up_folder.exists() else {}

        depth_down_folder = down_folder / 'depth'
        depth_up_folder = up_folder / 'depth'
        depths_down = None
        depths_up = None
        if depth_down_folder.exists():
            depths_down = load_gt_images(depth_down_folder)
        if depth_up_folder.exists():
            depths_up = load_gt_images(depth_up_folder)

        if static_onboarding_sequence == 'both':
            assert images_down is not None or images_up is not None
            assert segs_down is not None or segs_up is not None
            sequence_starts.append(len(images_down))

            merged_images = images_down
            merged_segmentations = segs_down
            merged_depths = depths_down

            offset = sequence_starts[1]
            for frame, img in images_up.items():
                merged_images[offset + frame] = img
            for frame, seg in segs_up.items():
                merged_segmentations[offset + frame] = seg
            if merged_depths is not None:
                for frame, depth in depths_up.items():
                    merged_depths[offset + frame] = depth

            return merged_images, merged_segmentations, merged_depths, sequence_starts

        elif static_onboarding_sequence == 'down':
            return images_down, segs_down, depths_down, sequence_starts

        elif static_onboarding_sequence == 'up':
            return images_up, segs_up, depths_up, sequence_starts

        else:
            raise ValueError(f'Unknown static onboarding sequence type: {static_onboarding_sequence}')
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_device)
        image_folder = sequence_folder / img_folder
        segmentation_folder = sequence_folder / mask_folder
        depth_folder = sequence_folder / 'depth'
        gt_images = load_gt_images(image_folder)
        if scene_obj_id is not None and segmentation_folder.exists():
            # Multi-object scene: resolve GT index per frame from scene_gt.json
            gt_filename = _scene_gt_filename(dataset, hot3d_device)
            scene_gt_path = sequence_folder / gt_filename
            gt_segs = load_gt_segmentations_by_obj_id(segmentation_folder, scene_gt_path, scene_obj_id)
        elif segmentation_folder.exists():
            gt_segs = load_gt_segmentations(segmentation_folder)
        else:
            gt_segs = {}
        gt_depths = None
        if depth_folder.exists():
            gt_depths = load_gt_images(depth_folder)
        return gt_images, gt_segs, gt_depths, None


def read_gt_Se3_cam2obj_transformations(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, scale_factor,
                                        onboarding_type: str = None, sequence_starts: Optional[List[int]] = None,
                                        static_onboarding_sequence: Optional[str] = None, scene_obj_id: int = None,
                                        device: str = 'cpu', hot3d_device: str = 'aria') -> Dict[int, Se3]:
    gt_filename = _scene_gt_filename(dataset, hot3d_device)
    if sequence_type == 'onboarding' and onboarding_type == 'static' and static_onboarding_sequence is not None:
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: extract_gt_Se3_cam2obj(p / gt_filename, scale_factor, device=device),
            sequence_starts=sequence_starts,
            hot3d_device=hot3d_device,
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_device)
        pose_json_path = sequence_folder / gt_filename
        return extract_gt_Se3_cam2obj(pose_json_path, scale_factor, object_id=scene_obj_id, device=device)


def read_object_id(bop_folder: Path, dataset: str, sequence: str, sequence_type: str,
                   onboarding_type: str = None, static_onboarding_sequence: Optional[str] = None,
                   scene_obj_id: int = None, sequence_starts: List[int] = None,
                   hot3d_device: str = 'aria') -> int:
    gt_filename = _scene_gt_filename(dataset, hot3d_device)
    if sequence_type == 'onboarding' and onboarding_type == 'static' and static_onboarding_sequence is not None:
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: extract_object_id(p / gt_filename),
            sequence_starts=sequence_starts,
            hot3d_device=hot3d_device,
        )[1]
    else:
        # scene_obj_id is the actual obj_id (from get_bop_val_sequences), not an index
        if scene_obj_id is not None:
            return scene_obj_id
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_device)
        pose_json_path = sequence_folder / gt_filename
        return extract_object_id(pose_json_path)[1]


def read_pinhole_params(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, scale,
                        onboarding_type: str = None, static_onboarding_sequence: Optional[str] = None,
                        sequence_starts: Optional[List[int]] = None, device='cpu',
                        hot3d_device: str = 'aria') -> dict[int, PinholeCamera]:
    camera_filename = _scene_camera_filename(dataset, hot3d_device)
    pinhole_loader = get_pinhole_params_from_hot3d if dataset == 'hot3d' else get_pinhole_params
    if sequence_type == 'onboarding' and onboarding_type == 'static' and static_onboarding_sequence is not None:
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: pinhole_loader(p / camera_filename, scale, device=device),
            sequence_starts=sequence_starts,
            hot3d_device=hot3d_device,
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_device)
        pose_json_path = sequence_folder / camera_filename
        return pinhole_loader(pose_json_path, scale, device=device)


def add_extrinsics_to_pinhole_params(pinhole_params: Dict[int, PinholeCamera], gt_Se3_world2cam: dict[int, Se3]) -> (
        Dict)[int, PinholeCamera]:
    for frm_i in pinhole_params.keys():
        pinhole = pinhole_params[frm_i]
        gt_T_world2cam = gt_Se3_world2cam[frm_i].matrix().unsqueeze(0)
        pinhole_params[frm_i] = PinholeCamera(pinhole.intrinsics, gt_T_world2cam, pinhole.width, pinhole.height)

    return pinhole_params



def read_gt_Se3_world2cam(pose_json_path: Path, input_scale='m', output_scale='m', device: str = 'cpu') \
        -> dict[int, Se3]:
    data = json.loads(pose_json_path.read_text())
    scale = get_scale_to_meter(input_scale) * get_scale_from_meter(output_scale)
    result = {}
    for frame_id_str, frame_data in data.items():
        R = torch.tensor(frame_data['cam_R_w2c'], dtype=torch.float32, device=device).reshape(3, 3)
        t = torch.tensor(frame_data['cam_t_w2c'], dtype=torch.float32, device=device).reshape(3) * scale
        result[int(frame_id_str)] = Se3(Quaternion.from_matrix(R), t)
    return result


def read_depth_scales(pose_json_path: Path) -> dict[int, float]:
    data = json.loads(pose_json_path.read_text())
    return {int(k): v['depth_scale'] for k, v in data.items()}


def read_static_onboarding_world2cam(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: Optional[str] = None,
        static_onboarding_sequence: Optional[str] = None,
        sequence_starts: Optional[List[int]] = None,
        device: str = 'cpu',
        hot3d_device: str = 'aria',
) -> dict[int, Se3]:
    camera_filename = _scene_camera_filename(dataset, hot3d_device)
    if sequence_type == 'onboarding' and onboarding_type == 'static' and static_onboarding_sequence is not None:
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: read_gt_Se3_world2cam(p / camera_filename, device=device),
            sequence_starts=sequence_starts,
            hot3d_device=hot3d_device,
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                               hot3d_device=hot3d_device)
        pose_json_path = sequence_folder / camera_filename
        return read_gt_Se3_world2cam(pose_json_path, device=device)


def read_dynamic_onboarding_depth_scales(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str,
        hot3d_device: str = 'aria',
) -> dict[int, float]:
    sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                           hot3d_device=hot3d_device)
    pose_json_path = sequence_folder / 'scene_camera.json'
    return read_depth_scales(pose_json_path)


def set_config_for_bop_onboarding(config: GloPoseConfig, sequence: str):
    sequence_name_split = sequence.split('_')
    if len(sequence_name_split) == 3:
        if sequence_name_split[2] == 'down':
            config.bop.onboarding_type = 'static'
            config.bop.static_onboarding_sequence = 'down'
            config.onboarding.similarity_transformation = 'kabsch'
        elif sequence_name_split[2] == 'up':
            config.bop.onboarding_type = 'static'
            config.bop.static_onboarding_sequence = 'up'
            config.onboarding.similarity_transformation = 'kabsch'
        elif sequence_name_split[2] == 'dynamic':
            config.bop.onboarding_type = 'dynamic'
            config.onboarding.similarity_transformation = 'depths'
            config.input.frame_provider_config.erode_segmentation = True
            config.input.run_only_on_frames_with_known_pose = False
            config.input.skip_indices *= 4
        elif sequence_name_split[2] == 'both':
            config.bop.onboarding_type = 'static'
            config.bop.static_onboarding_sequence = 'both'
            config.onboarding.similarity_transformation = 'kabsch'
        config.run.sequence = '_'.join(sequence_name_split[:2])
    elif len(sequence_name_split) == 2:
        # HOT3D: NNNNNN_static or NNNNNN_dynamic (no up/down distinction)
        if sequence_name_split[1] == 'static':
            config.bop.onboarding_type = 'static'
            config.bop.static_onboarding_sequence = None
            config.onboarding.similarity_transformation = 'kabsch'
        elif sequence_name_split[1] == 'dynamic':
            config.bop.onboarding_type = 'dynamic'
            config.onboarding.similarity_transformation = 'depths'
            config.input.frame_provider_config.erode_segmentation = True
            config.input.run_only_on_frames_with_known_pose = False
            config.input.skip_indices *= 4
        config.run.sequence = sequence_name_split[0]


def group_test_targets_by_image(test_annotations):
    grouped = OrderedDict()

    for item in test_annotations:
        key = (item['im_id'], item['scene_id'])

        # Initialize if first time seeing this key
        if key not in grouped:
            grouped[key] = {'im_id': item['im_id'],
                            'scene_id': item['scene_id'],
                            'objects': [],
                            'objects_counts': []}

        # Add object info if it exists
        if 'obj_id' in item:
            grouped[key]['objects'].append(item['obj_id'])
            grouped[key]['objects_counts'].append(item['inst_count'])

    # Convert to list while preserving order
    test_annotations = list(grouped.values())

    return test_annotations


def get_descriptors_for_templates(path_to_split: Path, path_to_split_cache: Path, descriptor: str, device='cuda') \
        -> Tuple[Dict[int, torch.Tensor], ...]:
    from adapters.cnos_adapter import create_descriptor_extractor
    descriptor = create_descriptor_extractor(model=descriptor)

    images_dict: Dict[int, Any] = defaultdict(list)
    segmentations_dict: Dict[int, Any] = defaultdict(list)
    cls_descriptors_dict: Dict[int, Any] = defaultdict(list)

    obj_dirs = sorted([d for d in path_to_split.iterdir() if d.is_dir()])

    for obj_dir in tqdm(obj_dirs, desc="Loading templates", total=len(obj_dirs)):

        obj_dir_name = obj_dir.name
        obj_id = int(obj_dir.stem.split('_')[1]) if 'obj' in obj_dir.stem else int(obj_dir.stem)

        rgb_dir = obj_dir / 'rgb'
        mask_dir = obj_dir / 'mask_visib'
        descriptor_dir = path_to_split_cache / obj_dir_name

        # Get all image files
        rgb_files = sorted(rgb_dir.glob('*'))
        mask_files = sorted(mask_dir.glob('*'))

        for rgb_file, mask_file in tqdm(zip(rgb_files, mask_files),
                                        desc=f"Templates for {obj_dir.stem}",
                                        total=len(rgb_files),
                                        leave=False, disable=True):

            images_dict[obj_id].append(rgb_file)
            segmentations_dict[obj_id].append(mask_file)

            descriptor_file = descriptor_dir / f'{rgb_file.stem}.pt'
            if descriptor_file.exists():
                payload = torch.load(descriptor_file, weights_only=True)
                if type(payload) is tuple:
                    cls_descriptor, patch_descriptor = payload
                else:
                    cls_descriptor = payload
            else:
                cls_descriptor, patch_descriptor = descriptor.get_detections_from_files(rgb_file, mask_file)
                torch.save(cls_descriptor, descriptor_file)
            cls_descriptors_dict[obj_id].append(cls_descriptor.squeeze(0))

    for obj_id in images_dict.keys():
        cls_descriptors_dict[obj_id] = torch.stack(cls_descriptors_dict[obj_id]).to(device)

    return images_dict, segmentations_dict, cls_descriptors_dict
