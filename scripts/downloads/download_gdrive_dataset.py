"""Download the native (non-BOP) HANDAL or HOPE releases from Google Drive.

    python scripts/downloads/download_gdrive_dataset.py --dataset handal
    python scripts/downloads/download_gdrive_dataset.py --dataset hope --output-dir /data/HOPE

The BOP-format versions of both datasets, which is what the reconstruction entry
points read, come from scripts/downloads/download_bop.py instead. Requires gdown
(``pip install gdown``).
"""
import argparse
import os
import zipfile
from pathlib import Path

import gdown

FOLDERS = {
    'handal': 'https://drive.google.com/drive/folders/1B5r7CO5gEoqFl_K4PAikg7pYYflbyuo_',
    'hope': 'https://drive.google.com/drive/folders/1Hj5K9RIdcNxBFiU8qG0-oL3Ryd9f2gOY',
}


def download_and_unzip(folder_url: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(folder_url, output=str(output_dir), quiet=False)

    for item in sorted(output_dir.iterdir()):
        if item.suffix == '.zip':
            with zipfile.ZipFile(item, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            item.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(FOLDERS), required=True)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Destination (default: $OMA_DATA_ROOT/<DATASET>)')
    args = parser.parse_args()

    out = args.output_dir or Path(os.environ.get('OMA_DATA_ROOT', 'data')) / args.dataset.upper()
    download_and_unzip(FOLDERS[args.dataset], out)
    print(f'Downloaded {args.dataset} to {out}')
