from pathlib import Path

from dataset_generators import scenarios
from utils.dataset_sequences import get_google_scanned_objects_sequences, select_gso_validation
from utils.experiment_runners import run_on_synthetic_data
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'GoogleScannedObjects'
    args = parse_args()

    # Load config first so we can modify dataset-specific parameters
    config = load_config(args.config)

    all_sequences = get_google_scanned_objects_sequences(
        config.paths.google_scanned_objects_data_folder / 'models')
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    elif args.val:
        sequences = select_gso_validation(all_sequences)
    else:
        sequences = all_sequences

    # GSO renders 400 frames per sequence, twice what the real datasets give, so an
    # every-k-th config that lands at ~25 keyframes on HANDAL lands at ~50 here and the
    # complete-graph matching (quadratic in keyframes) pushes a sequence towards two
    # hours. Halve the keyframe count for every passthrough config by doubling the
    # stride, once, outside the loop (the config object is shared across sequences,
    # so doubling inside would compound). Adaptive filters are untouched.
    # Applied 2026-08-25; GSO numbers in the paper predate this and are to be
    # recomputed with it.
    if config.onboarding.frame_filter == 'passthrough':
        config.onboarding.passthrough_skip *= 2

    for sequence in sequences:
        with exception_logger(sequence):
            # Set camera parameters specific to GoogleScannedObjects
            config.renderer.camera_position = (0, -5.0, 0)
            config.renderer.camera_up = (0, 0, 1)

            # Dataset mechanics, not experiment choices: the frames here are rendered
            # from the mesh on the fly, so the frame provider must be the synthetic one
            # and there is no depth to align with, leaving Kabsch. Previously each GSO
            # config had to remember to set these, which meant any config written
            # for the BOP datasets died on the first frame with
            # "FileNotFoundError: The file 0.png does not exist" (the precomputed
            # provider looking for images that are never written to disk). Setting them
            # here lets the shared ablation configs run unmodified on GSO.
            #
            # Segmentation uses SAM2, exactly like the real-image datasets (and as the
            # paper states): the renderer's exact frame-0 mask is used only as the SAM2
            # prompt (see the synthetic-frame branch in FrameProviderAll), and every
            # other frame's mask comes from SAM2 propagation over the rendered frames,
            # not from the renderer. This keeps GSO consistent with the "first-frame
            # mask + SAM2 propagation" pipeline rather than handing it oracle per-frame
            # masks.
            config.input.frame_provider = 'synthetic'
            config.input.segmentation_provider = 'SAM2'
            config.onboarding.similarity_transformation = 'kabsch'

            # Construct paths specific to GoogleScannedObjects
            gt_model_path = config.paths.google_scanned_objects_data_folder / Path('models') / Path(sequence)
            gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
            gt_mesh_path = gt_model_path / Path('meshes/model.obj')

            # Run tracking with z-axis rotations for GoogleScannedObjects
            run_on_synthetic_data(
                config=config,
                dataset=dataset,
                sequence=sequence,
                experiment=args.experiment,
                output_folder=args.output_folder,
                gt_mesh_path=gt_mesh_path,
                gt_texture_path=gt_texture_path,
                rotation_generator=scenarios.random_walk_on_a_sphere
            )


if __name__ == "__main__":
    main()
