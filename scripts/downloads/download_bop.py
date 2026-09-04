"""Download a BOP dataset from the bop-benchmark HuggingFace organisation.

    python scripts/downloads/download_bop.py --dataset hope
    python scripts/downloads/download_bop.py --dataset handal --include 'models/**'

The destination defaults to $OMA_DATA_ROOT/bop/<dataset> (data/bop/<dataset> if the
variable is unset), which is where PathsConfig.bop_data_folder looks.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True, help="BOP dataset name, e.g. hope, handal, lmo")
parser.add_argument("--include", nargs="*", default=None,
                    help="Glob patterns to include, e.g. 'train_aria/*.tar' 'models/**'")
parser.add_argument("--output-dir", type=Path, default=None,
                    help="Destination (default: $OMA_DATA_ROOT/bop/<dataset>)")
args = parser.parse_args()

local_dir = args.output_dir or Path(os.environ.get("OMA_DATA_ROOT", "data")) / "bop" / args.dataset
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=f"bop-benchmark/{args.dataset}",
    allow_patterns=args.include,
    repo_type="dataset",
    local_dir=local_dir,
)
print(f"Downloaded {args.dataset} to {local_dir}")
