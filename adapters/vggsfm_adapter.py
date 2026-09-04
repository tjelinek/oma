"""Adapter for the native VGGSfM external repository.

This is the SOLE location in GloPose that touches VGGSfM.

Unlike the other reconstruction adapters, VGGSfM is NOT imported in-process:
its pinned dependency stack (pycolmap==3.10, pyceres==2.3) is incompatible with
the pycolmap 4.x this codebase runs on. Instead the official ``demo.py`` is run
as a subprocess in a dedicated virtualenv (``vggsfm_python_bin``), completely
unmodified — its own tracker, its own camera initializer, its own BA — and the
resulting COLMAP ``sparse/`` folder is loaded back with our pycolmap (the
on-disk binary format is version-stable). This keeps the baseline "native":
no seeding from any external pose head, no code surgery in the repo.

The scene directory fed to demo.py contains symlinks to the pipeline's keyframe
images under their original names, so the exported reconstruction's image names
match DataGraph.image_filename and the standard carve/alignment/eval path works
unchanged.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pycolmap

VGGSFM_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'vggsfm'


def reconstruct_with_vggsfm(
    image_paths: list[Path],
    image_names: list[str],
    device: str = 'cuda',
    camera_K=None,
    segmentation_paths: Optional[list[Path]] = None,
    python_bin: Optional[str] = None,
    repo_path: Optional[Path] = None,
    fine_tracking: bool = True,
    shared_camera: bool = True,
    timeout_s: int = 7200,
) -> Optional[pycolmap.Reconstruction]:
    """Run the official VGGSfM demo on a set of images and load its output.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names — must match DataGraph.image_filename.
        device: Accepted for interface parity; the subprocess inherits CUDA env.
        camera_K: Accepted for interface parity but NOT used — native VGGSfM
            self-calibrates (SIMPLE_PINHOLE); feeding GT K would de-nativize it.
        segmentation_paths: Accepted for parity; the multi-view mask carve runs
            at the pipeline level on the returned reconstruction.
        python_bin: Interpreter of the dedicated VGGSfM venv. Defaults to the
            current interpreter (only correct if VGGSfM deps are importable there).
        repo_path: VGGSfM repo checkout. Defaults to repositories/vggsfm.
        fine_tracking: First-attempt value for VGGSfM's fine tracking. On any
            failure a second attempt runs with fine_tracking=False (their known
            OOM lever). Flat retry ladder across separate processes, so no GPU
            state can leak between attempts.
        shared_camera: Single shared camera across frames (ours is one camera).
        timeout_s: Per-attempt subprocess timeout.

    Returns:
        pycolmap.Reconstruction (in VGGSfM's arbitrary-scale world frame, image
        names matching ``image_names``), or None on failure.
    """
    repo = Path(repo_path) if repo_path is not None else VGGSFM_REPO
    if not (repo / 'demo.py').exists():
        print(f"VGGSfM repo not found at {repo}")
        return None
    python_bin = python_bin or sys.executable

    if len(image_paths) < 2:
        print("VGGSfM requires at least 2 images")
        return None

    # Give the subprocess the whole GPU: the parent pipeline's matcher models keep
    # allocations + a large reserved cache alive; without this the demo sees only
    # ~half of a 40G A100 and OOMs on >~60-image cells.
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    scene_dir = Path(tempfile.mkdtemp(prefix='vggsfm_scene_'))
    images_dir = scene_dir / 'images'
    images_dir.mkdir()
    try:
        for p, name in zip(image_paths, image_names):
            os.symlink(Path(p).resolve(), images_dir / name)

        attempts = [fine_tracking] + ([False] if fine_tracking else [])
        last_tail = ''
        for attempt_fine in attempts:
            sparse_dir = scene_dir / 'sparse'
            if sparse_dir.exists():
                shutil.rmtree(sparse_dir)
            cmd = [
                python_bin, 'demo.py',
                f'SCENE_DIR={scene_dir}',
                f'shared_camera={shared_camera}',
                f'fine_tracking={attempt_fine}',
                'camera_type=SIMPLE_PINHOLE',
                'make_reproj_video=False',
                'viz_visualize=False',
                'gr_visualize=False',
            ]
            print(f"[vggsfm] running native demo (fine_tracking={attempt_fine}) "
                  f"on {len(image_paths)} images")
            try:
                proc = subprocess.run(
                    cmd, cwd=str(repo), capture_output=True, text=True,
                    timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print(f"[vggsfm] attempt timed out after {timeout_s}s")
                continue
            last_tail = (proc.stdout or '')[-3000:] + '\n' + (proc.stderr or '')[-3000:]
            if proc.returncode != 0:
                print(f"[vggsfm] demo exited {proc.returncode} "
                      f"(fine_tracking={attempt_fine}); tail:\n{last_tail}")
                continue
            if not sparse_dir.exists():
                print(f"[vggsfm] demo succeeded but wrote no sparse/; tail:\n{last_tail}")
                continue
            try:
                reconstruction = pycolmap.Reconstruction(str(sparse_dir))
            except Exception as e:
                print(f"[vggsfm] failed to load sparse output: {e}")
                continue
            if reconstruction.num_images() < 2:
                print(f"[vggsfm] empty/degenerate reconstruction "
                      f"({reconstruction.num_images()} images) — treating as failure")
                continue
            print(f"[vggsfm] reconstruction: {reconstruction.num_images()} images, "
                  f"{reconstruction.num_points3D()} 3D points "
                  f"(fine_tracking={attempt_fine})")
            return reconstruction

        print(f"[vggsfm] all attempts failed; last output tail:\n{last_tail}")
        return None
    finally:
        shutil.rmtree(scene_dir, ignore_errors=True)
