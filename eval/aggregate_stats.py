"""Experiment-level CSV aggregation for the canonical results layout.

Canonical layout (written by ``utils/experiment_runners.py`` and the ``run_*.py`` runners;
see CLAUDE.md, "Results layout"):

    <root>/<experiment>/                                   <- what --output_folder points to
    ├── reconstruction_sequence_stats.csv                  <- AGGREGATE, rebuilt from the per-seq files
    ├── reconstruction_keyframe_stats.csv                  <- AGGREGATE
    ├── reconstruction_dataset_stats.csv                   <- per-dataset summary (one row per dataset)
    └── <dataset>/<sequence>/                              <- one folder per onboarding run
        ├── reconstruction_sequence_stats.csv              <- per-seq SOURCE OF TRUTH (one row)
        ├── reconstruction_keyframe_stats.csv              <- per-seq (one row per registered keyframe)
        ├── rerun_*.rrd, logs/, glomap_*/ ...

Every onboarding run writes only its own per-sequence files, then calls
``rebuild_experiment_aggregates()``. The aggregates are rebuilt from scratch and replaced
atomically (write to a temp file + ``os.replace``), so concurrent jobs of the same
experiment never race on a shared read-modify-write: the worst case is two of them
rebuilding at once, each producing a complete snapshot of what exists at that moment,
and the next one rebuilds again.

Legacy layouts are still read so old experiments keep working:
  * ``<experiment>/<sequence>/<csv>`` (depth 1; older per-sequence directories),
  * rows that exist only in the root aggregate (old runs that appended there directly).
Per-sequence files always win over root-only rows; among per-sequence files the newest wins.

This module deliberately depends only on pandas, so aggregating results needs no GPU
and no torch/kornia/pycolmap installation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCE_STATS = 'reconstruction_sequence_stats.csv'
KEYFRAME_STATS = 'reconstruction_keyframe_stats.csv'
DATASET_STATS = 'reconstruction_dataset_stats.csv'

# dtype=str for the key columns: purely-numeric sequence names (BOP classic '000001')
# otherwise parse as int64 and never match a string key.
KEY_DTYPES = {'dataset': str, 'sequence': str}
KEY_COLUMNS = ['dataset', 'sequence']

RECON_QUALITY_COLS = ['accuracy_mm', 'completeness_mm', 'overall_mm',
                      'fscore_1mm', 'fscore_2mm', 'fscore_5mm']
AUC_COLS = ['pose_auc_at_5', 'pose_auc_at_10', 'pose_auc_at_30']


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    df_copy = df.copy()
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].round(decimals)
    return df_copy


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via a temp file + rename so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def experiment_root_of(write_folder: Path) -> Path:
    """``<root>/<experiment>/<dataset>/<sequence>`` -> ``<root>/<experiment>``."""
    return Path(write_folder).parent.parent


def find_per_sequence_csvs(experiment_root: Path, name: str = SEQUENCE_STATS) -> list[Path]:
    """Per-sequence CSVs under an experiment root: canonical depth 2 plus legacy depth 1.

    Fixed depths on purpose (no rglob): a COLMAP tree holds thousands of files and the
    results mount is sshfs locally.
    """
    experiment_root = Path(experiment_root)
    found = list(experiment_root.glob(f'*/*/{name}')) + list(experiment_root.glob(f'*/{name}'))
    return sorted(set(found))


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=KEY_DTYPES)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None
    if df.empty or not set(KEY_COLUMNS) <= set(df.columns):
        return None
    return df


def collect_rows(experiment_root: Path, name: str = SEQUENCE_STATS,
                 include_root_aggregate: bool = True) -> pd.DataFrame:
    """Concatenate the per-sequence ``name`` CSVs of an experiment into one DataFrame.

    For each ``(dataset, sequence)`` key exactly one source file wins: any per-sequence file
    beats the (legacy) root aggregate, and among per-sequence files the most recently
    modified one wins. All rows of the winning source for that key are kept, so this works
    for the one-row sequence stats and the many-rows keyframe stats alike.
    """
    experiment_root = Path(experiment_root)
    sources: list[tuple[int, float, pd.DataFrame]] = []  # (priority, mtime, df)

    root_csv = experiment_root / name
    if include_root_aggregate and root_csv.exists():
        df = _read_csv(root_csv)
        if df is not None:
            sources.append((0, 0.0, df))

    for path in find_per_sequence_csvs(experiment_root, name):
        df = _read_csv(path)
        if df is not None:
            sources.append((1, path.stat().st_mtime, df))

    if not sources:
        return pd.DataFrame(columns=KEY_COLUMNS)

    frames = []
    for source_id, (priority, mtime, df) in enumerate(sources):
        df = df.copy()
        df['_source'] = source_id
        df['_priority'] = priority
        df['_mtime'] = mtime
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)

    # Winning source per key = max (priority, mtime).
    ranked = all_rows[KEY_COLUMNS + ['_source', '_priority', '_mtime']].drop_duplicates()
    ranked = ranked.sort_values(['_priority', '_mtime']).drop_duplicates(KEY_COLUMNS, keep='last')
    winners = set(zip(ranked['dataset'], ranked['sequence'], ranked['_source']))
    keep = [(d, s, src) in winners for d, s, src in
            zip(all_rows['dataset'], all_rows['sequence'], all_rows['_source'])]
    result = all_rows[keep].drop(columns=['_source', '_priority', '_mtime'])
    return result.sort_values(KEY_COLUMNS, kind='stable').reset_index(drop=True)


def _as_bool(series: pd.Series) -> pd.Series:
    """CSV round-trips turn bool columns with gaps into object/str; normalise."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower() == 'true'


def compute_dataset_stats(sequence_df: pd.DataFrame, dataset_name: str,
                          output_translation_unit: str = 'cm') -> dict | None:
    """Dataset-level summary row from the per-sequence rows of ``dataset_name``.

    Quality metrics are averaged over cells that reconstructed AND aligned (the
    survivorship denominator noted in the paper TODO); the success rates are over all cells.
    """
    dataset_df = sequence_df[sequence_df['dataset'] == dataset_name]
    if len(dataset_df) == 0:
        return None

    recon_ok = _as_bool(dataset_df['reconstruction_success'])
    align_ok = _as_bool(dataset_df['alignment_success'])

    def mean_of(df, col):
        return pd.to_numeric(df[col], errors='coerce').mean() if col in df.columns else None

    stats = {
        'dataset': dataset_name,
        'num_sequences': len(dataset_df),
        'mean_input_frames': mean_of(dataset_df, 'input_frames'),
        'mean_keyframes': mean_of(dataset_df, 'num_keyframes'),
        'mean_colmap_registered_keyframes': mean_of(dataset_df, 'colmap_registered_keyframes'),
        'reconstruction_success_rate': recon_ok.sum() / len(dataset_df),
        'alignment_success_rate': align_ok.sum() / len(dataset_df),
        'mean_frame_filtering_time': mean_of(dataset_df, 'frame_filtering_time'),
        'mean_matching_time': mean_of(dataset_df, 'matching_time'),
        'mean_reconstruction_time': mean_of(dataset_df, 'reconstruction_time'),
    }

    metric_cols = ['mean_rotation_error', f'mean_translation_error_{output_translation_unit}',
                   'rot_accuracy_at_2_deg', 'rot_accuracy_at_5_deg', 'rot_accuracy_at_10_deg',
                   'trans_accuracy_at_1_cm', 'trans_accuracy_at_5_cm', 'trans_accuracy_at_10_cm',
                   ] + AUC_COLS + RECON_QUALITY_COLS
    successful_df = dataset_df[recon_ok & align_ok]
    for col in metric_cols:
        stats[col] = mean_of(successful_df, col) if len(successful_df) > 0 else None
    return stats


def compute_all_dataset_stats(sequence_df: pd.DataFrame,
                              output_translation_unit: str = 'cm') -> pd.DataFrame:
    rows = []
    if not sequence_df.empty:
        for dataset_name in sorted(sequence_df['dataset'].astype(str).unique()):
            row = compute_dataset_stats(sequence_df, dataset_name, output_translation_unit)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def rebuild_experiment_aggregates(experiment_root: Path,
                                  output_translation_unit: str = 'cm',
                                  verbose: bool = True) -> pd.DataFrame:
    """Rebuild ``<experiment>/reconstruction_{sequence,keyframe,dataset}_stats.csv``
    from the per-sequence files and replace them atomically. Returns the sequence table."""
    experiment_root = Path(experiment_root)

    seq_df = collect_rows(experiment_root, SEQUENCE_STATS)
    if seq_df.empty:
        if verbose:
            print(f'No per-sequence stats under {experiment_root}; aggregates not rebuilt.')
        return seq_df
    atomic_write_csv(round_numeric_columns(seq_df), experiment_root / SEQUENCE_STATS)

    kf_df = collect_rows(experiment_root, KEYFRAME_STATS)
    if not kf_df.empty:
        atomic_write_csv(round_numeric_columns(kf_df), experiment_root / KEYFRAME_STATS)

    ds_df = compute_all_dataset_stats(seq_df, output_translation_unit)
    if not ds_df.empty:
        atomic_write_csv(round_numeric_columns(ds_df), experiment_root / DATASET_STATS)

    if verbose:
        print(f'Experiment aggregates rebuilt at {experiment_root} '
              f'({len(seq_df)} sequences, {len(ds_df)} datasets)')
    return seq_df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild experiment-level stats CSVs from per-sequence files.')
    parser.add_argument('experiment_roots', nargs='+', type=Path,
                        help='One or more <root>/<experiment> folders.')
    args = parser.parse_args()
    for root in args.experiment_roots:
        rebuild_experiment_aggregates(root)


if __name__ == '__main__':
    main()
