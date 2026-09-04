import json
from pathlib import Path
from typing import Dict

import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera

from data_providers.frame_provider import PrecomputedSegmentationProvider
from eval.eval_onboarding import evaluate_onboarding
from onboarding.pipeline import OnboardingPipeline
from utils.dataset_sequences import get_navi_sequences, select_n_sequences, VAL_N_NAVI
from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger
from utils.experiment_runners import resolve_write_folder


def main():
    dataset = 'navi'
    args = parse_args()

    config = load_config(args.config)
    navi_sequences = get_navi_sequences(config.paths.navi_data_folder)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    elif args.val:
        sequences = select_n_sequences(navi_sequences, VAL_N_NAVI)
    else:
        sequences = navi_sequences[4:5]

    for obj_type_sequence in sequences:
        with exception_logger(obj_type_sequence):

            if obj_type_sequence not in navi_sequences:
                raise ValueError(f"Unknown sequence {obj_type_sequence}")

            obj_name, sequence = obj_type_sequence.split('@')
            config = load_config(args.config)

            experiment_name = args.experiment
            output_folder = args.output_folder

            config.run.experiment_name = experiment_name
            config.run.sequence = f'{obj_name}_{sequence}'
            config.run.dataset = dataset
            config.input.image_downsample = 1.0

            config.input.skip_indices *= 1
            config.run.object_id = obj_name

            # Determine output folder
            write_folder = resolve_write_folder(config.paths, experiment_name, dataset,
                                                config.run.sequence, output_folder)

            base_folder = config.paths.navi_data_folder / obj_name / sequence
            image_folder = base_folder / 'images'
            segmentation_folder = base_folder / 'masks'
            depths_folder = base_folder / 'depth'

            gt_path = base_folder / 'annotations.json'

            gt_images = load_images_navi(image_folder)
            gt_segs = load_images_navi(segmentation_folder)
            gt_depths = load_images_navi(depths_folder)

            gt_pinhole_params = extract_cam_data_navi(gt_path, config.input.image_downsample, config.run.device)

            valid_frames = sorted(set(gt_images.keys()) & set(gt_segs.keys()) & set(gt_pinhole_params.keys()))

            gt_images = [gt_images[i] for i in valid_frames]
            gt_segs = [gt_segs[i] for i in valid_frames]
            gt_depths = [gt_depths[i] for i in valid_frames]

            gt_pinhole_params = reindex_frame_dict(gt_pinhole_params, valid_frames)
            gt_Se3_world2cam = {i: Se3.from_matrix(pinhole.extrinsics) for i, pinhole in gt_pinhole_params.items()}

            first_segmentation = \
                PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs, segmentation_channel=0,
                                                                         image_downsample=config.input.image_downsample,
                                                                         device=config.run.device)

            config.input.input_frames = len(gt_images)
            config.input.frame_provider = 'precomputed'
            config.input.segmentation_provider = 'SAM2'

            tracker = OnboardingPipeline(config, write_folder, input_images=gt_images, depth_paths=gt_depths,
                                         gt_Se3_world2cam=gt_Se3_world2cam, gt_pinhole_params=gt_pinhole_params,
                                         input_segmentations=gt_segs, initial_segmentation=first_segmentation)
            view_graph = tracker.run_pipeline()
            evaluate_onboarding(view_graph, gt_Se3_world2cam, config.run, config.bop, write_folder)


def extract_cam_data_navi(gt_path, image_downsample: float = 1.0, device: str = 'cpu') -> Dict[int, PinholeCamera]:
    pinhole_params_per_frame = {}

    image_downsample_tensor = torch.tensor([image_downsample]).to(device)
    with open(gt_path, 'r') as file:
        pose_json = json.load(file)
        for frame_data in pose_json:
            frame_filename = frame_data['filename']
            frame_id = int(Path(frame_filename).stem.split('_')[1])

            camera_data = frame_data['camera']
            gt_q_world2cam = Quaternion(torch.tensor(camera_data['q']).to(device)[None])
            gt_t_world2cam = torch.tensor(camera_data['t']).to(device)[None]
            gt_Se3_world2cam = Se3(gt_q_world2cam, gt_t_world2cam)
            gt_f = camera_data['focal_length']

            img_h, img_w = frame_data['image_size']
            img_h_tensor = torch.tensor([img_h]).to(device).to(torch.float)
            img_w_tensor = torch.tensor([img_w]).to(device).to(torch.float)

            # Kornia PinholeCamera 4x4 intrinsics: principal point lives in COLUMN 2
            # ([0,2]/[1,2]); putting it in column 3 (as before) yields cx=cy=0, which
            # downstream locks a corner principal point into COLMAP — a uniform
            # ~32 deg pose bias across every NAVI sequence.
            intrinsics = torch.tensor([[gt_f, 0, img_w / 2.0, 0.],
                                       [0, gt_f, img_h / 2.0, 0.],
                                       [0, 0, 1., 0.],
                                       [0, 0, 0, 1.]]).to(device)[None]

            pinhole_params = PinholeCamera(intrinsics, gt_Se3_world2cam.matrix().squeeze(), img_h_tensor, img_w_tensor)
            pinhole_params.scale_(image_downsample_tensor)

            pinhole_params_per_frame[frame_id] = pinhole_params

    return pinhole_params_per_frame


def load_images_navi(image_folder: Path):
    """Load ground truth images."""
    gt_images = {
        int(file.stem.replace(f'frame_', '')): file
        for file in sorted(image_folder.iterdir())
        if file.is_file()
    }

    return gt_images


if __name__ == "__main__":
    main()
