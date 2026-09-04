from pathlib import Path

from data_providers.frame_provider import PrecomputedSegmentationProvider
from eval.eval_onboarding import evaluate_onboarding
from onboarding.pipeline import OnboardingPipeline
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
    read_object_id, read_pinhole_params
from utils.dataset_sequences import get_bop_classic_sequences, select_bop_classic_validation
from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger
from utils.experiment_runners import resolve_write_folder


def main():
    args = parse_args()
    config = load_config(args.config)

    bop_path = config.paths.bop_data_folder
    tless_seqs = get_bop_classic_sequences(bop_path, 'tless', 'train_primesense')
    lmo_seqs = get_bop_classic_sequences(bop_path, 'lmo', 'train')
    icbin_seqs = get_bop_classic_sequences(bop_path, 'icbin', 'train')

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    elif args.val:
        sequences = select_bop_classic_validation(tless_seqs + lmo_seqs + icbin_seqs)
    else:
        sequences = (tless_seqs + lmo_seqs + icbin_seqs)[0:1]

    for sequence_code in sequences:
        sequence_code_split = sequence_code.split('@')
        dataset, onboarding_folder, sequence_name = sequence_code_split

        with exception_logger(sequence_code):
            config = load_config(args.config)

            experiment_name = args.experiment

            config.run.experiment_name = experiment_name
            config.run.dataset = dataset
            config.run.sequence = sequence_name
            config.input.image_downsample = 1.

            config.input.depth_scale_to_meter = 0.001
            config.input.skip_indices *= 4

            # BOP-classic onboarding orbits are long (328 frames after the 4x decimation),
            # so the shared every-8th stride gives 41 keyframes per object, far denser
            # than any other dataset. Double the stride (~21 keyframes) for every
            # passthrough config, ours and baselines alike. Applied 2026-08-26.
            if config.onboarding.frame_filter == 'passthrough':
                config.onboarding.passthrough_skip *= 2

            # Path to BOP dataset
            bop_folder = config.paths.bop_data_folder
            # Determine output folder
            folder = resolve_write_folder(config.paths, experiment_name, dataset, sequence_name,
                                          args.output_folder)

            # Load images and segmentations
            gt_images, gt_segs, gt_depths, sequence_starts = \
                get_bop_images_and_segmentations(bop_folder, dataset, sequence_name, onboarding_folder)
            # Get camera-to-object transformations
            dict_gt_Se3_cam2obj = \
                read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence_name, onboarding_folder, 1.0,
                                                    device=config.run.device)

            object_id = read_object_id(bop_folder, dataset, sequence_name, onboarding_folder)
            config.run.object_id = object_id
            # Apply frame skipping
            if config.input.run_only_on_frames_with_known_pose:
                valid_frames = sorted(dict_gt_Se3_cam2obj.keys())
            else:
                valid_frames = list(range(min(gt_images.keys()), max(gt_images.keys()) + 1))
            gt_images = [gt_images[i] for i in valid_frames]
            gt_segs = [gt_segs[i] for i in valid_frames]
            if gt_depths is not None:
                gt_depths = [gt_depths[i] for i in valid_frames]
            dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)
            # Get initial image and segmentation
            first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                          segmentation_channel=0)
            # Get camera parameters
            pinhole_params = read_pinhole_params(bop_folder, dataset, sequence_name, onboarding_folder,
                                                 config.input.image_downsample, device=config.run.device)

            pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)

            gt_Se3_obj2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}

            # Update config with frame information
            config.input.input_frames = len(gt_images)
            config.input.frame_provider = 'precomputed'
            config.input.segmentation_provider = 'SAM2'
            # Initialize and run the tracker
            tracker = OnboardingPipeline(config, folder, input_images=gt_images, gt_Se3_world2cam=gt_Se3_obj2cam,
                                         gt_pinhole_params=pinhole_params, input_segmentations=gt_segs,
                                         depth_paths=gt_depths,
                                         initial_segmentation=first_segmentation)
            view_graph = tracker.run_pipeline()
            # Use the pipeline's GT copy: skip_indices decimation remaps GT to the
            # decimated frame ids inside OnboardingPipeline.__init__, while the local
            # dict stays full-rate. Passing the local dict compares keyframe d against
            # GT frame d instead of frame skip*d — only frame 0 matches and every pose
            # column is silently wrong (uniform ~97 deg on LM-O).
            evaluate_onboarding(view_graph, tracker.gt_Se3_world2cam, config.run, config.bop, folder)


if __name__ == "__main__":
    main()
