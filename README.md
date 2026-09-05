# OMa: Dense Object Matching for Dense Reconstruction

**[Project page](https://tjelinek.github.io/oma/)**

Tomáš Jelínek, Dmytro Mishkin, Jiří Matas — Visual Recognition Group, Czech Technical University in Prague

OMa (Object Matching) reconstructs a previously unseen rigid object from an RGB video of it.
Given the video, the camera intrinsics, and a segmentation mask of the object in the first
frame, it recovers the object's 3D point cloud together with the camera poses relative to the
object (up to a similarity transform). It needs **no depth, no 3D template, no category priors,
and no object-specific training**, and because it reconstructs from foreground correspondences
alone, it also handles objects that move independently of the scene, such as a hand-held object.

The pipeline: SAM2 propagates the first-frame mask through the video; a dense matcher (UFM,
fine-tuned on synthetic render pairs with the background zeroed) estimates foreground-to-foreground
correspondences; a matchability criterion selects keyframes online; the surviving correspondences
are chained into multi-view tracks and passed to correspondence-based Structure-from-Motion
(COLMAP via pycolmap).

![teaser](https://tjelinek.github.io/oma/static/images/teaser.png)

---

## Installation

Tested on Linux with CUDA GPUs. A conda environment covers everything; no cluster modules or
system packages beyond conda itself are assumed.

```bash
git clone https://github.com/tjelinek/oma.git
cd oma
conda env create -f environment.yml    # creates the env "oma" (Python 3.13)
conda activate oma
```

This installs PyTorch (pulled in by kaolin), pycolmap, Kornia, Kaolin, SAM2, UFM, RoMa, the BOP
toolkit and the remaining Python dependencies. UFM and SAM2 are installed as pip packages straight
from their upstream repositories, so the core pipeline needs no submodules and no manual build.

After the install, check that PyTorch actually sees your GPU:

```bash
python -c "import torch, pycolmap; print(torch.__version__, torch.cuda.is_available(), pycolmap.__version__)"
```

If `cuda.is_available()` is `False`, install a PyTorch build matching your CUDA version
([pytorch.org](https://pytorch.org/get-started/locally/)) into the same environment before
continuing.

**Optional, only for the feed-forward baselines** (VGGT, MASt3R, MapAnything, π³). The adapters
look for these under `repositories/`:

```bash
mkdir -p repositories && cd repositories
git clone https://github.com/facebookresearch/vggt.git
git clone https://github.com/naver/mast3r.git
git clone https://github.com/facebookresearch/map-anything.git
git clone https://github.com/yyfz/Pi3.git
```

### Hardware

A CUDA GPU is required. All experiments in the paper ran on a single NVIDIA A100. The dense
matchers (UFM, RoMa) are large models, so plan for a data-center-class or a recent high-memory
consumer GPU.

## Model weights

1. **UFM (base)** — downloaded automatically from Hugging Face (`infinity1096/UFM-Refine`) on
   first use. No manual step.
2. **SAM2.1** — download `sam2.1_hiera_large.pt` from the
   [SAM2 repository](https://github.com/facebookresearch/sam2#model-description) and place it at
   `weights/sam2.1_hiera_large.pt` (or set `SAM2_CHECKPOINT` to wherever you keep it).
3. **OMa fine-tuned UFM checkpoint** — the mask-robust fine-tune used for every "Ours" number.
   Place it at `weights/ufm_ft_last.pth` or set `OMA_UFM_WEIGHTS`.
   Download link: *to be added with the paper release.* Until then you can reproduce the
   checkpoint yourself with `training/ufm/train_ufm.py` (see [Fine-tuning](#fine-tuning-the-matcher-yourself)),
   or run the pipeline on the stock UFM weights by setting
   `cfg.onboarding.ufm.use_custom_weights = False` (this is the `vanilla` ablation arm, and it is
   markedly weaker).

## Configuring paths

Dataset, results and cache locations are fields of `PathsConfig` in
[`configs/glopose_config.py`](configs/glopose_config.py). They default to `data/`, `results/` and
`cache/` inside the repository and can be redirected with environment variables, without editing
any file:

| Variable | Default | Contents |
|---|---|---|
| `OMA_DATA_ROOT` | `data` | datasets (`bop/`, `HANDAL/`, `HO3D/`, `NAVI/`, ...) |
| `OMA_RESULTS_ROOT` | `results` | per-experiment output |
| `OMA_CACHE_ROOT` | `cache` | segmentation / matching / descriptor caches |
| `SAM2_CHECKPOINT` | `weights/sam2.1_hiera_large.pt` | SAM2.1 checkpoint |
| `OMA_UFM_WEIGHTS` | `weights/ufm_ft_last.pth` | fine-tuned UFM checkpoint |

## Data

The paper evaluates on the BOP model-free onboarding sequences of HOPEv2 and HANDAL, on NAVI, on
LM-O, and on rendered Google Scanned Objects clips.

```bash
python scripts/downloads/download_bop.py --dataset hope
python scripts/downloads/download_bop.py --dataset handal
```

- **NAVI**: [official release](https://github.com/google/navi), unpacked into
  `$OMA_DATA_ROOT/NAVI/navi_v1.5`.
- **HANDAL / HOPE native releases** (not needed for the BOP-format onboarding runs):
  `python scripts/downloads/download_gdrive_dataset.py --dataset handal`.

Expected folder layouts, annotation schemas and coordinate conventions for every dataset are
documented in [`docs/dataset_formats.md`](docs/dataset_formats.md).

## Running

Each dataset has its own entry point: `run_HOPE.py`, `run_HANDAL.py`, `run_NAVI.py`, `run_HO3D.py`,
`run_BOP_classic_onboarding.py`, `run_GoogleScannedObjects.py`. Configurations are **Python files**
that return a `GloPoseConfig` from `get_config()`.

Reconstruct one HOPE onboarding sequence with the paper's full method:

```bash
python run_HOPE.py \
    --config configs/onboarding/ours.py \
    --experiment my_first_run \
    --sequences obj_000001_up
```

Arguments shared by every `run_*.py`:

| Argument | Meaning |
|---|---|
| `--config` | Path to a Python config file |
| `--experiment` | Experiment name; results are grouped under it |
| `--sequences` | One or more sequence names (`obj_000001_up`, `obj_000003_dynamic`, ...) |
| `--output_folder` | Optional override of the experiment folder (default `{results_folder}/{experiment}`) |
| `--val` | Run the fixed validation subset instead of a single default sequence |

Results land in `{results_folder}/{experiment}/{dataset}/{sequence}/`: the COLMAP reconstruction
(`glomap_*/output/0/`), per-sequence statistic CSVs, logs, and a Rerun recording (`rerun_*.rrd`)
that you can open with the [Rerun viewer](https://rerun.io) to inspect the point cloud and the
estimated cameras in 3D. Experiment-level aggregates are rebuilt automatically after every
sequence, and by hand with:

```bash
python -m eval.aggregate_stats results/my_first_run
```

### Configurations

| Config | What it is |
|---|---|
| `configs/onboarding/ours.py` | **OMa**, the full method reported as "Ours" |
| `configs/onboarding/ablations/*.py` | single-factor knockouts of `ours.py` (the ablation table) |
| `configs/reconstruction/{vggt,mast3r,map_anything,pi3}_every_8th.py` | the feed-forward baselines |
| `configs/base_config.py` | plain defaults, useful as a starting point of your own |

Each ablation arm changes exactly one field of the shared anchor in
`configs/onboarding/_anchor.py`, so an arm file can be read on its own to see what it varies.

The exact sequence lists behind the `test`, `val` and `sanity` splits are frozen in
[`configs/splits.py`](configs/splits.py):

```bash
python -m configs.splits --split val --datasets hope handal
```

## Repository structure

```
onboarding/        the reconstruction pipeline (keyframe selection, matching, SfM, COLMAP utils)
data_providers/    frame / matching / flow providers (UFM, RoMa, SIFT, tracking backends)
adapters/          wrappers for external models (SAM2, VGGT, MASt3R, MapAnything, π³, ...)
configs/           Python configs; configs/glopose_config.py defines the schema
data_structures/   DataGraph, ViewGraph, keyframe buffer, observation types
eval/              reconstruction and point-cloud evaluation (pose error, F-score, aggregation)
models/            differentiable renderer and encoder used by the evaluation
utils/             dataset I/O, BOP conventions, geometry helpers, experiment runners
training/ufm/      mask-robust UFM fine-tuning on BOP-PBR render pairs
scripts/downloads/ dataset download helpers
run_*.py           per-dataset entry points
```

`adapters/` also carries backends explored during development but not reported in the paper
(point trackers, VGGSfM, SAM3D); they are reachable through config fields and are inert otherwise.

Note on naming: the method is OMa, but the Python package and its main config class are
historically named `glopose`.

## Fine-tuning the matcher yourself

The mask-robust UFM fine-tune is reproduced with BOP-PBR synthetic render pairs whose backgrounds
are zeroed, using ground-truth flow derived from the rendered depth and object poses. HANDAL and
HOPE (and YCB-V, whose objects appear in HO3D) are held out of the training pool, so the evaluation
objects stay unseen.

```bash
python -m training.ufm.train_ufm --datasets tless icbin itodd tudl hb --out-dir weights/ufm_ft
```

See `training/ufm/` for the dataset builder, the losses and the evaluation script.

## License

MIT, see [LICENSE](LICENSE).

Two exceptions worth knowing: the fine-tuned UFM checkpoint is a derivative of the UFM pretrained
weights and inherits their CC BY-NC-SA 4.0 (non-commercial) terms, and the optional baseline
checkouts under `repositories/` carry their own licenses, several of them non-commercial.

## Citation

```bibtex
@misc{jelinek2026oma,
  title  = {OMa: Dense Object Matching for Dense Reconstruction},
  author = {Jel\'inek, Tom\'a\v{s} and Mishkin, Dmytro and Matas, Ji\v{r}\'i},
  year   = {2026},
  note   = {Preprint},
  url    = {https://tjelinek.github.io/oma/}
}
```

## Acknowledgements

OMa builds on [UFM](https://github.com/UniFlowMatch/UFM),
[SAM2](https://github.com/facebookresearch/sam2) and
[COLMAP / pycolmap](https://colmap.github.io/), and is compared against
[VGGT](https://github.com/facebookresearch/vggt),
[MASt3R](https://github.com/naver/mast3r),
[MapAnything](https://github.com/facebookresearch/map-anything) and
[π³](https://github.com/yyfz/Pi3).
