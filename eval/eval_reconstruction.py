from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from kornia.geometry import Se3, So3, Quaternion
from pycolmap import Reconstruction

from eval.eval_point_cloud import compute_pose_auc
from onboarding.colmap_utils import get_image_Se3_world2cam


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """
    Round all numeric columns in a DataFrame to specified decimal places.

    Args:
        df: Input DataFrame
        decimals: Number of decimal places to round to

    Returns:
        DataFrame with rounded numeric columns
    """
    df_copy = df.copy()
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].round(decimals)
    return df_copy


def get_scale_factor(from_unit: str, to_unit: str) -> float:
    """
    Get the scale factor to convert from one unit to another.

    Args:
        from_unit: Source unit ('m', 'cm', 'mm')
        to_unit: Target unit ('m', 'cm', 'mm')

    Returns:
        Scale factor to multiply values by
    """
    # Convert to meters first, then to target unit
    to_meters = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}
    from_meters = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}

    if from_unit not in to_meters or to_unit not in from_meters:
        raise ValueError(f"Unsupported units. Use 'm', 'cm', or 'mm'. Got {from_unit} -> {to_unit}")

    return to_meters[from_unit] * from_meters[to_unit]


def get_translation_thresholds_in_output_unit(output_unit: str) -> Dict[str, float]:
    """
    Get translation accuracy thresholds converted to the specified output unit.
    Thresholds are always 0.1, 0.5, 1, 5, 10 cm, but converted to output unit for comparison.

    Args:
        output_unit: Output unit ('m', 'cm', 'mm')

    Returns:
        Dictionary mapping threshold names to values in output unit
    """
    # Fixed thresholds in centimeters
    base_thresholds_cm = {
        '0_1': 0.1,
        '0_5': 0.5,
        '1': 1.0,
        '5': 5.0,
        '10': 10.0
    }

    # Convert to target unit for comparison
    scale_factor = get_scale_factor('cm', output_unit)
    return {k: v * scale_factor for k, v in base_thresholds_cm.items()}


def evaluate_reconstruction(
        reconstruction,
        ground_truth_poses: Dict[int, Se3],
        image_name_to_frame_id: Dict[str, int],
        csv_output_path: Path,
        dataset: str,
        sequence: str,
        input_translation_unit: str = 'mm',
        output_translation_unit: str = 'cm'
):
    stats = []

    for image in reconstruction.images.values():
        image_frame_id = image_name_to_frame_id[image.name]

        Se3_obj2cam_pred = get_image_Se3_world2cam(image, 'cpu')

        # Get ground truth pose for this frame
        Se3_world2cam_gt = ground_truth_poses.get(image_frame_id)

        # Ground-truth rotation and translation
        gt_rotation = Se3_world2cam_gt.rotation.matrix().tolist() if Se3_world2cam_gt is not None else None
        gt_translation = Se3_world2cam_gt.translation.tolist() if Se3_world2cam_gt is not None else None

        pred_rotation = Se3_obj2cam_pred.rotation.matrix().tolist()
        pred_translation = Se3_obj2cam_pred.translation.tolist()

        # Add stats for the current image frame
        stats.append({
            'dataset': dataset,
            'sequence': sequence,
            'image_frame_id': image_frame_id,
            'gt_rotation': gt_rotation,
            'gt_translation': gt_translation,
            'pred_rotation': pred_rotation,
            'pred_translation': pred_translation
        })

    # Convert stats to a Pandas DataFrame
    stats_df = pd.DataFrame(stats)

    # If the CSV file exists, append; otherwise, create a new one
    if csv_output_path.exists():
        # dtype=str for the key columns: purely-numeric sequence names (BOP classic
        # '000001') otherwise parse as int64 and never match the string filter.
        existing_df = pd.read_csv(csv_output_path, dtype={'dataset': str, 'sequence': str})
        filtered_df = existing_df[~((existing_df['dataset'] == dataset) &
                                    (existing_df['sequence'] == sequence))]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(csv_output_path, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(csv_output_path, index=False)


def update_sequence_reconstructions_stats(csv_per_frame_stats: Path, csv_per_sequence_stats: Path, num_keyframes: int,
                                          input_frames: int, reconstruction: Reconstruction, dataset: str,
                                          sequence: str, reconstruction_success: bool, pose_alignment_success: bool,
                                          frame_filtering_time: float, reconstruction_time: float,
                                          input_translation_unit: str = 'mm', output_translation_unit: str = 'cm',
                                          reconstruction_quality: dict | None = None,
                                          colmap_num_reconstructions: int = 1,
                                          matching_time: float = 0.0):
    # Read the input CSV file containing reconstruction data.
    # dtype=str for the key columns: BOP-classic sequence names are purely numeric
    # ('000001'), which read_csv otherwise parses as int64 — the string equality
    # filter then matches nothing and every pose column silently aggregates to None.
    if csv_per_frame_stats.exists():
        df = pd.read_csv(csv_per_frame_stats, dtype={'dataset': str, 'sequence': str})
        sequence_df = df[(df['dataset'] == dataset) & (df['sequence'] == sequence)]
    else:
        sequence_df = pd.DataFrame()

    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    # Get scale factor for translation unit conversion
    scale_factor = get_scale_factor(input_translation_unit, output_translation_unit)

    # Get translation thresholds converted to output unit for comparison
    translation_thresholds = get_translation_thresholds_in_output_unit(output_translation_unit)

    stats = {
        'dataset': dataset,
        'sequence': sequence,
        'num_keyframes': num_keyframes,
        'input_frames': input_frames,
        'colmap_registered_keyframes': reconstruction.num_reg_images() if reconstruction is not None else None,
        'colmap_num_reconstructions': colmap_num_reconstructions,
        'mean_rotation_error': None,
        f'mean_translation_error_{output_translation_unit}': None,
        'rot_accuracy_at_2_deg': None,
        'rot_accuracy_at_5_deg': None,
        'rot_accuracy_at_10_deg': None,
        'trans_accuracy_at_1_cm': None,
        'trans_accuracy_at_5_cm': None,
        'trans_accuracy_at_10_cm': None,
        'pose_auc_at_5': None,
        'pose_auc_at_10': None,
        'pose_auc_at_30': None,
        'accuracy_mm': None,
        'completeness_mm': None,
        'overall_mm': None,
        'fscore_1mm': None,
        'fscore_2mm': None,
        'fscore_5mm': None,
        'reconstruction_success': reconstruction_success,
        'alignment_success': pose_alignment_success,
        'frame_filtering_time': frame_filtering_time,
        'matching_time': matching_time,
        'reconstruction_time': reconstruction_time,
        'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
        'note': str()
    }

    if not reconstruction_success:
        # Check reconstruction first: a failed SfM reconstruction also forces
        # alignment to fail, so attributing the failure to alignment would be
        # misleading. The root cause is the missing reconstruction.
        stats['note'] = 'SfM reconstruction failed.'
    elif not pose_alignment_success:
        stats['note'] = 'Pose alignment failed.'
    else:
        stats.update(compute_pose_stats_from_keyframe_df(
            sequence_df, scale_factor, translation_thresholds, output_translation_unit))

    # Merge reconstruction quality metrics if provided
    if reconstruction_quality is not None:
        for key in ('accuracy_mm', 'completeness_mm', 'overall_mm',
                    'fscore_1mm', 'fscore_2mm', 'fscore_5mm'):
            stats[key] = reconstruction_quality.get(key)

    stats_df = pd.DataFrame([stats])

    # Write to CSV. dtype=str keeps numeric sequence names ('000001') comparable
    # with the string keys in stats_df — otherwise the dedup filter misses and
    # re-runs accumulate duplicate rows.
    if csv_per_sequence_stats.exists():
        existing_df = pd.read_csv(csv_per_sequence_stats, dtype={'dataset': str, 'sequence': str})

        filtered_df = existing_df[~existing_df.set_index(['dataset', 'sequence']).index.isin(
            stats_df.set_index(['dataset', 'sequence']).index)]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(csv_per_sequence_stats, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(csv_per_sequence_stats, index=False)

    print(f"Statistics written to {csv_per_sequence_stats}")
    return stats_df


def compute_pose_stats_from_keyframe_df(sequence_df, scale_factor: float,
                                        translation_thresholds: dict,
                                        output_translation_unit: str = 'cm') -> dict:
    """Compute the per-sequence pose-metric columns from per-keyframe GT/pred poses.

    sequence_df: rows of reconstruction_keyframe_stats.csv for one (dataset, sequence).
    Returns {} when no keyframe has both GT and pred poses.
    """
    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    for _, row in sequence_df.iterrows():
        # Skip if ground truth is not available. Missing GT is written as None
        # but read back from CSV as NaN, so an `is None` check misses it — use
        # pd.isna. This is the common case for dynamic onboarding, where only a
        # subset of keyframes (often just the first frame) have a GT pose.
        if pd.isna(row['gt_rotation']) or pd.isna(row['gt_translation']):
            continue

        gt_rot_val = eval(row['gt_rotation'])
        gt_trans_val = eval(row['gt_translation'])
        pred_rot_val = eval(row['pred_rotation'])
        pred_trans_val = eval(row['pred_translation'])

        gt_rot_matrix = torch.Tensor(gt_rot_val) if gt_rot_val is not None else None
        pred_rot_matrix = torch.Tensor(pred_rot_val) if pred_rot_val is not None else None
        gt_trans = torch.Tensor(gt_trans_val) if gt_trans_val is not None else None
        pred_trans = torch.Tensor(pred_trans_val) if pred_trans_val is not None else None

        if gt_rot_matrix is None or pred_rot_matrix is None or gt_trans is None or pred_trans is None:
            continue

        gt_So3 = So3(Quaternion.from_matrix(gt_rot_matrix))
        pred_So3 = So3(Quaternion.from_matrix(pred_rot_matrix))

        rel_rot: So3 = gt_So3.inverse() * pred_So3

        rotation_error_deg = torch.rad2deg(torch.linalg.norm(rel_rot.q.to_axis_angle())).item()
        rotation_errors.append(rotation_error_deg)

        # Convert translation error to output unit
        translation_error = torch.linalg.norm(gt_trans - pred_trans).item() * scale_factor
        translation_errors.append(translation_error)

    if not (rotation_errors and translation_errors):
        return {}

    rotation_errors_np = np.asarray(rotation_errors)
    translation_errors_np = np.asarray(translation_errors)

    pose_stats = {
        'mean_rotation_error': np.mean(rotation_errors_np),
        f'mean_translation_error_{output_translation_unit}': np.mean(translation_errors_np),
        'rot_accuracy_at_2_deg': np.sum(rotation_errors_np <= 2) / len(rotation_errors_np),
        'rot_accuracy_at_5_deg': np.sum(rotation_errors_np <= 5) / len(rotation_errors_np),
        'rot_accuracy_at_10_deg': np.sum(rotation_errors_np <= 10) / len(rotation_errors_np),
        'trans_accuracy_at_1_cm': np.sum(translation_errors_np <= translation_thresholds['1']) / len(
            translation_errors_np),
        'trans_accuracy_at_5_cm': np.sum(translation_errors_np <= translation_thresholds['5']) / len(
            translation_errors_np),
        'trans_accuracy_at_10_cm': np.sum(translation_errors_np <= translation_thresholds['10']) / len(
            translation_errors_np),
    }

    # Pose AUC over rotation errors
    auc = compute_pose_auc(rotation_errors_np)
    pose_stats.update({
        'pose_auc_at_5': auc.get('auc_at_5'),
        'pose_auc_at_10': auc.get('auc_at_10'),
        'pose_auc_at_30': auc.get('auc_at_30'),
    })
    return pose_stats


def update_dataset_reconstruction_statistics(
        per_sequence_stats_file: Path,
        dataset_name: str,
        output_translation_unit: str = 'cm'
):
    """Refresh the one-row-per-dataset summary next to ``per_sequence_stats_file``.

    Thin wrapper over ``eval.aggregate_stats.compute_dataset_stats``; the canonical path
    (``eval_onboarding.evaluate_onboarding``) rebuilds all experiment aggregates via
    ``rebuild_experiment_aggregates`` instead, so this is kept for ad-hoc callers only.
    """
    from eval.aggregate_stats import compute_dataset_stats, atomic_write_csv, DATASET_STATS

    df = pd.read_csv(per_sequence_stats_file, dtype={'dataset': str, 'sequence': str})
    dataset_stats = compute_dataset_stats(df, dataset_name, output_translation_unit)
    if dataset_stats is None:
        print(f"No data found for dataset {dataset_name}")
        return None
    stats_df = pd.DataFrame([dataset_stats])

    per_dataset_stats_file = per_sequence_stats_file.parent / DATASET_STATS
    if per_dataset_stats_file.exists():
        existing_df = pd.read_csv(per_dataset_stats_file)
        filtered_df = existing_df[existing_df['dataset'] != dataset_name]
        stats_df = pd.concat([filtered_df, stats_df], ignore_index=True)
    atomic_write_csv(round_numeric_columns(stats_df), per_dataset_stats_file)
    print(f"Dataset statistics for {dataset_name} written to {per_dataset_stats_file}")
    return stats_df


def update_experiment_statistics(
        experiment_name: str,
        kf_stats_file: Path,
        experiment_stats_file: Path,
        output_translation_unit: str = 'cm'
):
    """
    Aggregate dataset statistics into experiment-level statistics.

    Reads from a single keyframe stats file and creates separate experiment statistics files
    for each distinct dataset, using the same calculation logic as update_dataset_reconstruction_statistics.

    Args:
        experiment_name: Name of the experiment
        kf_stats_file: Path to keyframe statistics CSV file containing all sequences
        experiment_stats_file: Base path for output experiment statistics CSV files
        output_translation_unit: Unit for translation measurements
    """
    if not kf_stats_file.exists():
        print(f"Error: Keyframe stats file {kf_stats_file} does not exist.")
        return

    # Read the keyframe stats file
    df = pd.read_csv(kf_stats_file, dtype={'dataset': str, 'sequence': str})

    # Get all distinct datasets
    distinct_datasets = df['dataset'].unique()

    results = {}

    # Process each dataset separately
    for dataset_name in distinct_datasets:
        dataset_df = df[df['dataset'] == dataset_name]

        if len(dataset_df) == 0:
            print(f"Warning: No data found for dataset {dataset_name}. Skipping.")
            continue

        # Calculate aggregated statistics for this dataset (same as update_dataset_reconstruction_statistics)
        experiment_stats = {
            'experiment': experiment_name,
            'dataset': dataset_name,
            'num_sequences': len(dataset_df),
            'mean_input_frames': dataset_df['input_frames'].mean(),
            'mean_keyframes': dataset_df['num_keyframes'].mean(),
            'mean_colmap_registered_keyframes': dataset_df['colmap_registered_keyframes'].mean(),
            'reconstruction_success_rate': dataset_df['reconstruction_success'].sum() / len(dataset_df),
            'alignment_success_rate': dataset_df['alignment_success'].sum() / len(dataset_df),
            'mean_frame_filtering_time': dataset_df['frame_filtering_time'].mean(),
        'mean_matching_time': dataset_df['matching_time'].mean() if 'matching_time' in dataset_df.columns else None,
            'mean_reconstruction_time': dataset_df['reconstruction_time'].mean(),
        }

        # Only calculate accuracy metrics for successful reconstructions
        successful_df = dataset_df[dataset_df['reconstruction_success'] & dataset_df['alignment_success']]

        recon_quality_cols = ['accuracy_mm', 'completeness_mm', 'overall_mm',
                              'fscore_1mm', 'fscore_2mm', 'fscore_5mm']
        auc_cols = ['pose_auc_at_5', 'pose_auc_at_10', 'pose_auc_at_30']

        if len(successful_df) > 0:
            experiment_stats.update({
                'mean_rotation_error': successful_df['mean_rotation_error'].mean(),
                f'mean_translation_error_{output_translation_unit}': successful_df[
                    f'mean_translation_error_{output_translation_unit}'].mean(),
                'rot_accuracy_at_2_deg': successful_df['rot_accuracy_at_2_deg'].mean(),
                'rot_accuracy_at_5_deg': successful_df['rot_accuracy_at_5_deg'].mean(),
                'rot_accuracy_at_10_deg': successful_df['rot_accuracy_at_10_deg'].mean(),
                'trans_accuracy_at_1_cm': successful_df['trans_accuracy_at_1_cm'].mean(),
                'trans_accuracy_at_5_cm': successful_df['trans_accuracy_at_5_cm'].mean(),
                'trans_accuracy_at_10_cm': successful_df['trans_accuracy_at_10_cm'].mean(),
            })
            for col in auc_cols:
                if col in successful_df.columns:
                    experiment_stats[col] = successful_df[col].mean()
                else:
                    experiment_stats[col] = None
            for col in recon_quality_cols:
                if col in successful_df.columns:
                    experiment_stats[col] = successful_df[col].mean()
                else:
                    experiment_stats[col] = None
        else:
            # No successful reconstructions for this dataset
            for metric in ['mean_rotation_error', f'mean_translation_error_{output_translation_unit}',
                           'rot_accuracy_at_2_deg', 'rot_accuracy_at_5_deg', 'rot_accuracy_at_10_deg',
                           'trans_accuracy_at_1_cm', 'trans_accuracy_at_5_cm',
                           'trans_accuracy_at_10_cm'] + auc_cols + recon_quality_cols:
                experiment_stats[metric] = None

        # Convert to DataFrame
        stats_df = pd.DataFrame([experiment_stats])

        # Create dataset-specific experiment stats file
        dataset_experiment_stats_file = experiment_stats_file.parent / f"{experiment_stats_file.stem}_{dataset_name}.csv"

        # Write to CSV
        if dataset_experiment_stats_file.exists():
            existing_df = pd.read_csv(dataset_experiment_stats_file)
            # Remove existing entry for this experiment
            filtered_df = existing_df[existing_df['experiment'] != experiment_name]
            updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
            # Round numeric columns before saving
            updated_df = round_numeric_columns(updated_df)
            updated_df.to_csv(dataset_experiment_stats_file, index=False)
        else:
            # Round numeric columns before saving
            stats_df = round_numeric_columns(stats_df)
            stats_df.to_csv(dataset_experiment_stats_file, index=False)

        print(f"Experiment statistics for dataset {dataset_name} written to {dataset_experiment_stats_file}")
        results[dataset_name] = stats_df

    print(f"Processed datasets: {', '.join(distinct_datasets)}")
    return results
