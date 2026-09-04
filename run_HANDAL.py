from pathlib import Path

from data_providers.frame_provider import PrecomputedSegmentationProvider
from eval.eval_onboarding import evaluate_onboarding
from onboarding.pipeline import OnboardingPipeline
from utils.bop_challenge import add_extrinsics_to_pinhole_params, load_gt_images, load_gt_segmentations, \
    extract_gt_Se3_cam2obj, extract_object_id, get_pinhole_params
from utils.dataset_sequences import get_handal_sequences
from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger
from utils.experiment_runners import resolve_write_folder


def main():
    dataset = 'handal_native'
    args = parse_args()

    config = load_config(args.config)
    handal_train, handal_test = get_handal_sequences(config.paths.handal_data_folder)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = handal_test[4:5]

    for obj_type_sequence in sequences:
        with exception_logger(obj_type_sequence):

            if obj_type_sequence in handal_train:
                sequence_type = 'train'
            elif obj_type_sequence in handal_test:
                sequence_type = 'test'
            else:
                raise ValueError(f"Unknown sequence {obj_type_sequence}")

            obj_name, sequence = obj_type_sequence.split('@')
            config = load_config(args.config)

            experiment_name = args.experiment
            output_folder = args.output_folder

            config.run.experiment_name = experiment_name
            config.run.sequence = sequence
            config.run.dataset = dataset
            config.input.image_downsample = .5

            config.input.skip_indices *= 1

            config.run.special_hash = obj_name.replace('handal_dataset_', '')

            # Determine output folder
            write_folder = resolve_write_folder(config.paths, experiment_name, dataset,
                                                f'{config.run.special_hash}_{sequence}', output_folder)

            base_folder = config.paths.handal_data_folder / 'HANDAL' / obj_name / sequence_type / sequence
            image_folder = base_folder / 'rgb'
            segmentation_folder = base_folder / 'mask_visib'
            scene_gt_path = base_folder / 'scene_gt.json'
            scene_cam_path = base_folder / 'scene_camera.json'

            gt_images = load_gt_images(image_folder)
            gt_segs = load_gt_segmentations(segmentation_folder)

            cam_scale = 1.0
            dict_gt_Se3_cam2obj = extract_gt_Se3_cam2obj(scene_gt_path, cam_scale, device=config.run.device)
            object_id = extract_object_id(scene_gt_path)[1]
            config.run.object_id = object_id

            valid_frames = sorted(set(gt_images.keys()) & set(gt_segs.keys()) & set(dict_gt_Se3_cam2obj.keys()))

            gt_images = [gt_images[i] for i in valid_frames]
            gt_segs = [gt_segs[i] for i in valid_frames]

            dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)
            gt_Se3_world2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}

            pinhole_params = get_pinhole_params(scene_cam_path, config.input.image_downsample, device=config.run.device)
            pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)
            pinhole_params = add_extrinsics_to_pinhole_params(pinhole_params, gt_Se3_world2cam)

            first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                          segmentation_channel=0)

            config.input.input_frames = len(gt_images)
            config.input.frame_provider = 'precomputed'
            config.input.segmentation_provider = 'SAM2'

            tracker = OnboardingPipeline(config, write_folder, input_images=gt_images,
                                         gt_Se3_cam2obj=dict_gt_Se3_cam2obj,
                                         gt_Se3_world2cam=gt_Se3_world2cam, gt_pinhole_params=pinhole_params,
                                         input_segmentations=gt_segs, initial_segmentation=first_segmentation)
            view_graph = tracker.run_pipeline()
            evaluate_onboarding(view_graph, gt_Se3_world2cam, config.run, config.bop, write_folder)


if __name__ == "__main__":
    main()
