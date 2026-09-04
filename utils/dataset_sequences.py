import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


# --- Validation set ---------------------------------------------------------
# A small, fixed-seed subset of onboarding sequences used for development and
# functionality testing (run via the `--val` flag). The selection is
# object-centric and deterministic: the same objects are used for the static and
# dynamic onboarding modes, and reruns are identical regardless of input order.
# See CLAUDE.md "Validation set".
VAL_SEED = 20260609
VAL_N_DEFAULT = 8
VAL_N_NAVI = 12


def select_n_sequences(items: List[str], n: int = VAL_N_DEFAULT, seed: int = VAL_SEED) -> List[str]:
    """Deterministically pick up to `n` items, independent of input ordering.

    Samples from the sorted unique set so the result depends only on the contents
    and the seed. Returns all items (sorted) if there are `n` or fewer.
    """
    unique = sorted(set(items))
    if len(unique) <= n:
        return unique
    return sorted(random.Random(seed).sample(unique, n))


def _object_id(seq: str) -> str:
    """Strip a BOP onboarding suffix (_dynamic/_down/_up/_both) to get the object id."""
    for suffix in ('_dynamic', '_down', '_up', '_both'):
        if seq.endswith(suffix):
            return seq[: -len(suffix)]
    return seq


def select_bop_onboarding_validation(dynamic: List[str], up: List[str], down: List[str],
                                     both: List[str], n: int = VAL_N_DEFAULT,
                                     seed: int = VAL_SEED) -> Dict[str, List[str]]:
    """Pick the validation subset for a BOP onboarding dataset (HANDAL/HOPE/HOT3D shape).

    Selects `n` objects (fewer if unavailable) that have both a static and a dynamic
    capture (falls back to static-only objects when the dataset has no dynamic
    sequences). For each selected object the result includes, when present:
      - one static orientation (`_up` or `_down`), chosen at random with the fixed seed,
      - the merged `_both` static run,
      - the `_dynamic` run.
    The same objects back all three lists.

    Returns a dict with keys 'objects', 'static', 'both', 'dynamic'.
    """
    up_set, down_set, both_set, dyn_set = set(up), set(down), set(both), set(dynamic)
    static_objects = {_object_id(s) for s in up} | {_object_id(s) for s in down}
    dynamic_objects = {_object_id(s) for s in dynamic}

    pool = sorted(static_objects & dynamic_objects) if dynamic_objects else sorted(static_objects)
    objects = select_n_sequences(pool, n, seed)

    val_static: List[str] = []
    val_both: List[str] = []
    val_dynamic: List[str] = []
    for obj in objects:
        variants = [v for v in (f'{obj}_up', f'{obj}_down') if v in up_set or v in down_set]
        if variants:
            # Per-object seed → choice is stable regardless of how many objects are picked.
            val_static.append(random.Random(f'{seed}-{obj}').choice(variants))
        if f'{obj}_both' in both_set:
            val_both.append(f'{obj}_both')
        if f'{obj}_dynamic' in dyn_set:
            val_dynamic.append(f'{obj}_dynamic')

    return {
        'objects': objects,
        'static': sorted(val_static),
        'both': sorted(val_both),
        'dynamic': sorted(val_dynamic),
    }


def select_bop_classic_validation(sequences: List[str], n: int = VAL_N_DEFAULT,
                                  seed: int = VAL_SEED) -> List[str]:
    """Pick the validation subset for BOP classic: `n` scenes per sub-dataset.

    Sequences are formatted '{dataset}@{split}@{scene}'. Selection is grouped by the
    '{dataset}' prefix so each of tless/lmo/icbin is represented (capped at what exists).
    """
    by_dataset: Dict[str, List[str]] = {}
    for seq in sequences:
        prefix = seq.split('@', 1)[0]
        by_dataset.setdefault(prefix, []).append(seq)

    selected: List[str] = []
    for prefix in sorted(by_dataset):
        selected.extend(select_n_sequences(by_dataset[prefix], n, seed))
    return selected


def get_handal_sequences(path: Path) -> Tuple[List[str], List[str]]:
    """Scan HANDAL dataset directory for train/test sequences.

    Args:
        path: Path to the HANDAL dataset root (e.g., data_folder / 'HANDAL')

    Returns:
        Tuple of (train_sequences, test_sequences), each formatted as 'category@sequence'
    """
    train_sequences = []
    test_sequences = []

    if not path.exists():
        return train_sequences, test_sequences

    for category_dir in sorted(path.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        train_dir = category_dir / "train"
        if train_dir.exists():
            for sequence_dir in sorted(train_dir.iterdir()):
                if sequence_dir.is_dir():
                    train_sequences.append(f"{category_name}@{sequence_dir.name}")

        test_dir = category_dir / "test"
        if test_dir.exists():
            for sequence_dir in sorted(test_dir.iterdir()):
                if sequence_dir.is_dir():
                    test_sequences.append(f"{category_name}@{sequence_dir.name}")

    return train_sequences, test_sequences


def get_navi_sequences(path: Path) -> List[str]:
    """Scan NAVI dataset directory for video sequences.

    Args:
        path: Path to the NAVI dataset root (e.g., data_folder / 'NAVI' / 'navi_v1.5')

    Returns:
        Sorted list of sequences formatted as 'object@video-folder'
    """
    video_sequences = []

    if not path.exists():
        return video_sequences

    for object_dir in path.iterdir():
        if object_dir.is_dir():
            for item in object_dir.iterdir():
                if item.is_dir() and item.name.startswith('video-'):
                    video_sequences.append(f"{object_dir.name}@{item.name}")

    return sorted(video_sequences)


def get_ho3d_sequences(path: Path) -> Tuple[List[str], List[str]]:
    """Scan HO3D dataset directory for train/evaluation sequences.

    Args:
        path: Path to the HO3D dataset root (e.g., data_folder / 'HO3D')

    Returns:
        Tuple of (train_sequences, test_sequences)
    """
    train_sequences = []
    test_sequences = []

    train_dir = path / 'train'
    if train_dir.exists():
        train_sequences = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    eval_dir = path / 'evaluation'
    if eval_dir.exists():
        test_sequences = sorted(d.name for d in eval_dir.iterdir() if d.is_dir())

    return train_sequences, test_sequences


# YCBInEOAT videos depict five YCB objects; map the sequence-name prefix to the
# canonical BOP ycbv object integer (used for GT-mesh lookup and object_id).
YCBINEOAT_TO_YCBV_OBJ = {
    'cracker_box': 2,
    'sugar_box': 3,
    'tomato_soup_can': 4,
    'mustard': 5,
    'bleach': 12,
}


def ycbineoat_ycbv_obj_id(sequence: str) -> int:
    """Return the ycbv object integer for a YCBInEOAT video (by name prefix)."""
    for prefix, obj_id in YCBINEOAT_TO_YCBV_OBJ.items():
        if sequence.startswith(prefix):
            return obj_id
    raise ValueError(f"Unknown YCBInEOAT object for sequence '{sequence}'")


def select_ycbineoat_validation(sequences: List[str], seed: int = VAL_SEED) -> List[str]:
    """Pick the YCBInEOAT validation subset: one clip per YCB object.

    YCBInEOAT ships two clips for each of four objects and one for the fifth, so a
    bare `select_n_sequences` would be object-blind (it could take both clips of one
    object and drop another entirely). Mirroring the object-centric BOP-onboarding
    val (one capture per object, fixed seed), we group clips by object (the
    ycbv-name prefix) and pick one clip per object with a per-object seed, so the
    choice is stable regardless of how many clips exist. Covers all five object
    geometries; deterministic and independent of input ordering.
    """
    by_object: Dict[str, List[str]] = {}
    for seq in sequences:
        by_object.setdefault(ycbineoat_ycbv_obj_id(seq), []).append(seq)

    val: List[str] = []
    for obj_id in sorted(by_object):
        variants = sorted(by_object[obj_id])
        val.append(random.Random(f'{seed}-{obj_id}').choice(variants))
    return sorted(val)


def get_ycbineoat_sequences(path: Path) -> List[str]:
    """List YCBInEOAT video sequences (one directory per manipulation clip).

    Args:
        path: YCBInEOAT root holding one folder per video (rgb/, depth/,
              annotated_poses/, cam_K.txt, init_mask.png).

    Returns:
        Sorted list of sequence names.
    """
    if not path.exists():
        return []
    return sorted(d.name for d in path.iterdir()
                  if d.is_dir() and (d / 'annotated_poses').exists())


def get_tum_rgbd_sequences(path: Path) -> List[str]:
    """Scan TUM RGB-D dataset directory for sequences.

    Args:
        path: Path to the TUM RGB-D dataset root (e.g., data_folder / 'SLAM' / 'tum_rgbd')

    Returns:
        Sorted list of sequence directory names
    """
    if not path.exists():
        return []

    return sorted(d.name for d in path.iterdir() if d.is_dir())


def get_behave_sequences(path: Path) -> List[str]:
    """Scan BEHAVE dataset directory for video sequences.

    Args:
        path: Path to the BEHAVE train directory (e.g., data_folder / 'BEHAVE' / 'train')

    Returns:
        Sorted list of sequence names (mp4 stems, excluding *_mask_obj)
    """
    if not path.exists():
        return []

    return sorted(
        f.stem for f in path.glob('*.mp4')
        if not f.stem.endswith('_mask_obj')
    )


def get_google_scanned_objects_sequences(path: Path) -> List[str]:
    """Scan GoogleScannedObjects models directory for sequences.

    Args:
        path: Path to the models directory (e.g., data_folder / 'GoogleScannedObjects' / 'models')

    Returns:
        Sorted list of model directory names
    """
    if not path.exists():
        return []

    return sorted(d.name for d in path.iterdir() if d.is_dir())


def select_gso_validation(sequences: List[str], n: int = VAL_N_DEFAULT,
                          seed: int = VAL_SEED) -> List[str]:
    """Pick the GoogleScannedObjects validation subset.

    GSO is one synthetic sequence per object (a random walk on the sphere rendered
    from the object's mesh), so there is no capture/orientation grouping to respect
    and a plain seeded draw over the object set is the right analogue of the
    object-centric BOP-onboarding val. Deterministic and independent of input
    ordering; returns everything if there are `n` or fewer objects.
    """
    return select_n_sequences(sequences, n, seed)


def get_bop_val_sequences(path: Path) -> List[str]:
    """Scan BOP val directory for scene/object sequences.

    Args:
        path: Path to the BOP val directory (e.g., data_folder / 'bop' / 'handal' / 'val')

    Returns:
        Sorted list of sequences formatted as '{scene_id}_{object_id}'
    """
    if not path.exists():
        return []

    sequences = []
    for scene_dir in sorted(path.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_id = scene_dir.name
        scene_gt_path = scene_dir / 'scene_gt.json'
        if scene_gt_path.exists():
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            obj_ids = set()
            for frame_data in scene_gt.values():
                for obj_entry in frame_data:
                    obj_ids.add(obj_entry['obj_id'])

            for obj_id in sorted(obj_ids):
                sequences.append(f"{scene_id}_{obj_id:06d}")

    return sequences


def get_bop_onboarding_sequences(path: Path, dataset: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Scan BOP onboarding directories for sequences.

    Args:
        path: Path to the BOP root directory (e.g., data_folder / 'bop')
        dataset: Dataset name (e.g., 'handal', 'hope')

    Returns:
        Tuple of (dynamic, static_up, static_down, static_both) sequence lists
    """
    dynamic_sequences = []
    static_up_sequences = []
    static_down_sequences = []

    dynamic_dir = path / dataset / 'onboarding_dynamic'
    if dynamic_dir.exists():
        for obj_dir in sorted(dynamic_dir.iterdir()):
            if obj_dir.is_dir():
                dynamic_sequences.append(f"{obj_dir.name}_dynamic")

    static_dir = path / dataset / 'onboarding_static'
    up_objects = set()
    down_objects = set()
    if static_dir.exists():
        for obj_dir in sorted(static_dir.iterdir()):
            if not obj_dir.is_dir():
                continue

            name = obj_dir.name
            if name.endswith('_up'):
                base = name[:-3]  # Remove '_up'
                up_objects.add(base)
                static_up_sequences.append(f"{base}_up")
            elif name.endswith('_down'):
                base = name[:-5]  # Remove '_down'
                down_objects.add(base)
                static_down_sequences.append(f"{base}_down")

    # Objects that have both up and down get a 'both' entry
    both_objects = sorted(up_objects & down_objects)
    static_both_sequences = [f"{obj}_both" for obj in both_objects]

    static_up_sequences.sort()
    static_down_sequences.sort()

    return dynamic_sequences, static_up_sequences, static_down_sequences, static_both_sequences


def get_hot3d_onboarding_sequences(path: Path, device: str = 'aria') -> Tuple[List[str], List[str]]:
    """Scan HOT3D onboarding directories for sequences.

    HOT3D uses device-specific folders (aria/quest3). Static sequences have
    up/down variants (like HANDAL), dynamic sequences are single captures.

    Static dir names:  obj_NNNNNN_down, obj_NNNNNN_up  → used as-is (3-part,
        handled by set_config_for_bop_onboarding like HANDAL).
    Dynamic dir names: obj_NNNNNN → appended with _dynamic (3-part).

    Args:
        path: Path to the BOP root directory (e.g., data_folder / 'bop')
        device: Camera device ('aria' or 'quest3'). Default 'aria' for RGB images.

    Returns:
        Tuple of (dynamic_sequences, static_sequences)
    """
    dynamic_sequences = []
    static_sequences = []

    dynamic_dir = path / 'hot3d' / f'object_ref_{device}_dynamic_scenewise'
    if dynamic_dir.exists():
        for scene_dir in sorted(dynamic_dir.iterdir()):
            if scene_dir.is_dir():
                dynamic_sequences.append(f"{scene_dir.name}_dynamic")

    static_dir = path / 'hot3d' / f'object_ref_{device}_static_scenewise'
    if static_dir.exists():
        for scene_dir in sorted(static_dir.iterdir()):
            if scene_dir.is_dir():
                # Dirs are already named obj_NNNNNN_down / obj_NNNNNN_up
                static_sequences.append(scene_dir.name)

    return dynamic_sequences, static_sequences


def get_bop_classic_sequences(path: Path, dataset: str, split: str) -> List[str]:
    """Scan BOP classic dataset directory for sequences.

    Args:
        path: Path to the BOP root directory (e.g., data_folder / 'bop')
        dataset: Dataset name (e.g., 'tless', 'lmo', 'icbin')
        split: Split name (e.g., 'train_primesense', 'train')

    Returns:
        Sorted list of sequences formatted as '{dataset}@{split}@{sequence_name}'
    """
    split_dir = path / dataset / split
    if not split_dir.exists():
        return []

    return sorted(
        f"{dataset}@{split}@{d.name}"
        for d in split_dir.iterdir()
        if d.is_dir()
    )
