import numpy as np
from pathlib import Path

import trimesh
from scipy.spatial import KDTree


def _load_ply_without_edges(mesh_path: Path) -> bytes:
    """Read a PLY file and strip edge elements that crash trimesh.

    Some Blender-exported PLY files contain ``element edge`` sections
    that trigger a bug in trimesh's PLY parser.  This helper reads the
    file, removes the edge element declaration from the header and the
    corresponding data lines (ASCII) or bytes (binary), and returns the
    cleaned content as bytes suitable for ``trimesh.load``.
    """
    import io
    import re

    raw = mesh_path.read_bytes()
    header_end = raw.find(b'end_header')
    header = raw[:header_end].decode('ascii')

    # Check if there's an edge element
    edge_match = re.search(r'^element edge (\d+)$', header, re.MULTILINE)
    has_edges = edge_match is not None
    has_texture = 'comment TextureFile' in header

    if not has_edges and not has_texture:
        return raw  # nothing to strip

    num_edges = int(edge_match.group(1)) if has_edges else 0

    # Remove edge element (and its properties) and TextureFile comments
    # from the header.  Removing the texture comment avoids a resolver
    # error when loading from BytesIO.
    header_lines = header.splitlines(keepends=True)
    cleaned_lines = []
    skip = False
    for line in header_lines:
        stripped = line.strip()
        if stripped.startswith('element edge'):
            skip = True
            continue
        if skip and stripped.startswith('property'):
            continue
        skip = False
        if stripped.startswith('comment TextureFile'):
            continue
        cleaned_lines.append(line)

    new_header = ''.join(cleaned_lines)

    if num_edges > 0:
        is_ascii = 'format ascii' in header
        if is_ascii:
            # For ASCII PLY: drop the last `num_edges` data lines
            after_header = raw[header_end + len('end_header'):].decode('ascii')
            data_lines = after_header.split('\n')
            # Remove trailing empty element if present
            while data_lines and data_lines[-1].strip() == '':
                data_lines.pop()
            data_lines = data_lines[:-num_edges]
            cleaned = new_header.encode('ascii') + b'end_header' + '\n'.join(data_lines).encode('ascii')
        else:
            # For binary PLY: we'd need to compute byte offsets per edge.
            # Fall back to returning the raw bytes and let trimesh try.
            cleaned = raw
    else:
        # No edges to strip — just rebuild with cleaned header
        cleaned = new_header.encode('ascii') + raw[header_end:]

    return cleaned


def sample_points_from_mesh(mesh_path: Path, num_points: int = 100_000) -> np.ndarray:
    """Load a mesh and uniformly sample points from its surface.

    Args:
        mesh_path: Path to a mesh file (PLY, OBJ, etc.).
        num_points: Number of points to sample.

    Returns:
        (N, 3) array of sampled surface points.
    """
    import io

    if mesh_path.suffix.lower() == '.ply':
        cleaned = _load_ply_without_edges(mesh_path)
        mesh = trimesh.load(io.BytesIO(cleaned), file_type='ply', force='mesh')
    else:
        mesh = trimesh.load(mesh_path, force='mesh')
    points, _ = mesh.sample(num_points, return_index=True)
    return np.asarray(points, dtype=np.float64)


def extract_reconstruction_points(reconstruction) -> np.ndarray:
    """Extract 3D point coordinates from a pycolmap Reconstruction.

    Args:
        reconstruction: A pycolmap.Reconstruction object (already aligned).

    Returns:
        (M, 3) array of reconstructed 3D points.
    """
    pts = [p.xyz for p in reconstruction.points3D.values()]
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(pts)


def compute_reconstruction_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    fscore_thresholds_mm: tuple = (1.0, 2.0, 5.0),
    max_dist_mm: float = 20.0,
    gt_mesh_unit: str = 'mm',
    pred_points_unit: str = 'mm',
) -> dict:
    """Compute accuracy, completeness, overall, and F-score metrics.

    Both point clouds are converted to millimetres before evaluation.

    Args:
        pred_points: (M, 3) predicted (reconstructed) points.
        gt_points: (N, 3) ground-truth surface points.
        fscore_thresholds_mm: Thresholds in mm for F-score computation.
        max_dist_mm: Clamping distance in mm for accuracy/completeness.
        gt_mesh_unit: Unit of gt_points ('mm', 'cm', or 'm').
        pred_points_unit: Unit of pred_points ('mm', 'cm', or 'm').

    Returns:
        Dict with accuracy_mm, completeness_mm, overall_mm,
        fscore_{τ}mm for each threshold, and point counts.
    """
    unit_to_mm = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
    pred = pred_points * unit_to_mm[pred_points_unit]
    gt = gt_points * unit_to_mm[gt_mesh_unit]

    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)

    # Accuracy: pred → GT
    dist_pred_to_gt, _ = tree_gt.query(pred)
    dist_pred_to_gt = np.minimum(dist_pred_to_gt, max_dist_mm)
    accuracy = float(np.mean(dist_pred_to_gt))

    # Completeness: GT → pred
    dist_gt_to_pred, _ = tree_pred.query(gt)
    dist_gt_to_pred = np.minimum(dist_gt_to_pred, max_dist_mm)
    completeness = float(np.mean(dist_gt_to_pred))

    overall = (accuracy + completeness) / 2.0

    result = {
        'accuracy_mm': accuracy,
        'completeness_mm': completeness,
        'overall_mm': overall,
        'num_pred_points': len(pred),
        'num_gt_points': len(gt),
    }

    for tau in fscore_thresholds_mm:
        precision = float(np.mean(dist_pred_to_gt <= tau))
        recall = float(np.mean(dist_gt_to_pred <= tau))
        if precision + recall > 0:
            fscore = 2.0 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        result[f'fscore_{tau:.0f}mm'] = fscore

    return result


def compute_pose_auc(
    errors: np.ndarray,
    thresholds: tuple = (5, 10, 30),
) -> dict:
    """Compute AUC of pose error at given thresholds.

    Adapted from VGGT ``calculate_auc_np``: builds a histogram of errors
    in integer-degree bins up to the maximum threshold, computes the
    cumulative sum, and returns the mean (= AUC) at each threshold.

    Args:
        errors: 1-D array of per-frame errors in degrees.
        thresholds: Degree thresholds at which to report AUC.

    Returns:
        Dict mapping ``auc_at_{t}`` to float AUC values.
    """
    max_threshold = max(thresholds)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(errors, bins=bins)
    num_frames = float(len(errors))
    if num_frames == 0:
        return {f'auc_at_{t}': 0.0 for t in thresholds}
    normalized = histogram.astype(float) / num_frames
    cumsum = np.cumsum(normalized)

    result = {}
    for t in thresholds:
        # AUC up to threshold t = mean of cumsum[0:t]
        result[f'auc_at_{t}'] = float(np.mean(cumsum[:t]))
    return result
