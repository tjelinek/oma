import argparse
import traceback
from contextlib import contextmanager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/base_config.py")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='default')  # Experiment name
    parser.add_argument("--val", action='store_true', default=False,
                        help="When no explicit --sequences are given, default to the fixed "
                             "validation subset instead of the full dataset")
    parser.add_argument("--merge-only", action='store_true', default=False,
                        help="Skip up/down onboarding, only run the merge step (requires cached ViewGraphs)")
    return parser.parse_args()


@contextmanager
def exception_logger(context: str = '', ignore_exceptions=(Exception,)):
    """Catch and log exceptions without crashing the batch run.

    Args:
        context: Identifies what was running when the exception occurred
                 (e.g. sequence name, dataset/object pair). Printed in the log.
        ignore_exceptions: Exception types to catch. Defaults to (Exception,).
    """
    try:
        yield
    except ignore_exceptions as e:
        prefix = f"[{context}] " if context else ""
        print(f"{prefix}Exception caught: {type(e).__name__}: {e}")
        traceback.print_exc()
