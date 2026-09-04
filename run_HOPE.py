from utils.bop_challenge import set_config_for_bop_onboarding
from utils.dataset_sequences import get_bop_onboarding_sequences, select_bop_onboarding_validation
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'hope'
    args = parse_args()

    config = load_config(args.config)
    hope_dynamic, hope_up, hope_down, hope_both = get_bop_onboarding_sequences(
        config.paths.bop_data_folder, dataset)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    elif args.val:
        # '_both' (up+down merged) onboarding is disabled by default — pass it explicitly
        # via --sequences obj_NNNNNN_both to run it.
        val = select_bop_onboarding_validation(hope_dynamic, hope_up, hope_down, hope_both)
        sequences = val['static'] + val['dynamic']
    else:
        sequences = (hope_up + hope_down + hope_dynamic)[4:5]

    for sequence in sequences:
        with exception_logger(sequence):
            config = load_config(args.config)

            experiment_name = args.experiment

            config.run.experiment_name = experiment_name
            config.run.dataset = dataset
            config.input.image_downsample = .5

            config.input.depth_scale_to_meter = 0.001

            config.input.skip_indices *= 1

            # HOPE dynamic sequences are short (64 to 110 frames), so the shared every-8th
            # stride leaves ~10 keyframes; halve it for dynamic sequences (~20 keyframes)
            # for every passthrough config, ours and the baselines alike. Static
            # sequences keep the stride (~23 keyframes; at k=4 they reach 43 to 50 and
            # the complete-graph dense matching takes 7 h per sequence).
            # Applied 2026-08-26; HOPE dynamic numbers before that used the full stride.
            if config.onboarding.frame_filter == 'passthrough' and sequence.endswith('_dynamic'):
                config.onboarding.passthrough_skip = max(1, config.onboarding.passthrough_skip // 2)

            sequence_type = 'onboarding'

            write_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, write_folder,
                                merge_only=args.merge_only)


if __name__ == "__main__":
    main()
