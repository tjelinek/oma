"""
Fine-tune UFM (UniFlowMatchClassificationRefinement) for object-only / masked
matching, supervising endpoint error (flow) + covisibility BCE (occlusion head).

This builds the training infrastructure only. Heavy multi-GPU / long runs are out
of scope here -- defaults are small so a run is a smoke/overfit test, not a real
training. Point `--datasets` at the BOP-PBR training pool (HANDAL/HOPE held out).

Example (single A100, smoke test):
  python -m training.ufm.train_ufm --max-steps 50 --batch-size 2 \
      --datasets tless ycbv --max-groups-per-dataset 20 --out-dir /tmp/ufm_ft
"""

from __future__ import annotations

import os

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from training.ufm.bop_flow_dataset import (
    BopFlowDatasetConfig, BopPbrFlowDataset, collate_drop_meta, DEFAULT_TRAIN_DATASETS,
)
from training.ufm.losses import UFMLoss


def build_model(device: str):
    from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
    model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
    return model.to(device)


def get_normalizer(model, device):
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    norm = IMAGE_NORMALIZATION_DICT[model.encoder.data_norm_type]
    mean = norm.mean.view(1, 3, 1, 1).to(device)
    std = norm.std.view(1, 3, 1, 1).to(device)
    return mean, std


def build_optimizer(model, lr_encoder: float, lr_head: float, weight_decay: float,
                    freeze_encoder: bool):
    groups = model.get_parameter_groups()
    param_groups = []
    for name, params in groups.items():
        params = [p for p in params if p.requires_grad]
        if not params:
            continue
        if name == "encoder":
            if freeze_encoder:
                for p in params:
                    p.requires_grad_(False)
                continue
            param_groups.append({"params": params, "lr": lr_encoder})
        else:
            param_groups.append({"params": params, "lr": lr_head})
    return torch.optim.AdamW(param_groups, lr=lr_head, weight_decay=weight_decay)


def make_views(batch, model, mean, std):
    """Normalize [0,1] RGB to the encoder's norm and pack into UFM view dicts."""
    img0 = (batch["img0"] - mean) / std
    img1 = (batch["img1"] - mean) / std
    dnt = model.encoder.data_norm_type
    view1 = {"img": img0, "symmetrized": False, "data_norm_type": dnt}
    view2 = {"img": img1, "symmetrized": False, "data_norm_type": dnt}
    return view1, view2


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def build_dataset(args) -> ConcatDataset:
    datasets = []
    for ds in args.datasets:
        cfg = BopFlowDatasetConfig(
            bop_root=args.bop_root, datasets=[ds], split=args.split,
            width=args.width, height=args.height,
            min_visib_fract=args.min_visib_fract,
            max_groups_per_dataset=args.max_groups_per_dataset,
            seed=args.seed,
            obj_split=args.obj_split, val_obj_fraction=args.val_obj_fraction,
            obj_split_seed=args.obj_split_seed,
            scene_split=args.scene_split, val_scene_fraction=args.val_scene_fraction,
            scene_split_seed=args.scene_split_seed,
        )
        try:
            datasets.append(BopPbrFlowDataset(cfg))
        except RuntimeError as e:
            print(f"[warn] dataset {ds}: {e}")
    return ConcatDataset(datasets)


def train(args):
    device = args.device
    torch.manual_seed(args.seed)

    print(f"[init] building dataset over {args.datasets} ...")
    dataset = build_dataset(args)
    print(f"[init] total (scene, obj) groups: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_drop_meta,
                        drop_last=True, persistent_workers=args.num_workers > 0)

    print("[init] loading UFM-Refine ...")
    model = build_model(device)
    mean, std = get_normalizer(model, device)
    model.train()

    optimizer = build_optimizer(model, args.lr_encoder, args.lr_head,
                                args.weight_decay, args.freeze_encoder)
    criterion = UFMLoss(flow_weight=args.flow_weight, covis_weight=args.covis_weight,
                        covis_pos_weight=args.covis_pos_weight, epe_clamp=args.epe_clamp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.time()
    fixed_batch = None
    while step < args.max_steps:
        for batch in loader:
            if args.overfit_one_batch:
                if fixed_batch is None:
                    fixed_batch = to_device(batch, device)
                batch = fixed_batch
            else:
                batch = to_device(batch, device)

            view1, view2 = make_views(batch, model, mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(view1, view2)
            # Loss in fp32 outside autocast.
            result.flow.flow_output = result.flow.flow_output.float()
            if result.covisibility is not None and result.covisibility.logits is not None:
                result.covisibility.logits = result.covisibility.logits.float()
            loss, logs = criterion(result, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"[step {step:5d}] total={logs['total']:.4f} epe={logs['epe']:.3f}px "
                      f"covis_bce={logs.get('covis_bce', float('nan')):.4f} "
                      f"covis_acc={logs.get('covis_acc', float('nan')):.3f} "
                      f"covis_iou={logs.get('covis_iou', float('nan')):.3f} "
                      f"n_valid={logs['n_valid']:.0f} ({dt:.1f}s)")
            step += 1

            # Periodic checkpoint. Saving only at the end means any late failure
            # throws away the whole run: a single unreadable mask PNG killed
            # run4 at step 7350 of 8000 and left nothing on disk. Write to a
            # temporary file and rename, so a crash mid-write cannot leave a
            # truncated checkpoint behind.
            if args.save_every > 0 and step % args.save_every == 0:
                tmp = out_dir / "ufm_ft_last.pth.tmp"
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, tmp)
                tmp.replace(out_dir / "ufm_ft_last.pth")
                print(f"[step {step:5d}] checkpoint saved")

            if step >= args.max_steps:
                break

    ckpt = out_dir / "ufm_ft_last.pth"
    tmp = out_dir / "ufm_ft_last.pth.tmp"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, tmp)
    tmp.replace(ckpt)
    print(f"[done] saved {ckpt}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--bop-root", default=os.environ.get("OMA_DATA_ROOT", "data") + "/bop")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_TRAIN_DATASETS))
    p.add_argument("--split", default="train_pbr")
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--min-visib-fract", type=float, default=0.30)
    p.add_argument("--max-groups-per-dataset", type=int, default=None)
    # Object-disjoint split: train on the 'train' objects, hold out 'val' objects for
    # model selection. Keeps HANDAL/HOPE untouched as the final test set.
    p.add_argument("--obj-split", choices=["train", "val", "all"], default="all")
    p.add_argument("--val-obj-fraction", type=float, default=0.2)
    p.add_argument("--obj-split-seed", type=int, default=1234)
    # Scene-disjoint (image-disjoint) split -- the primary val axis for model selection.
    # Train on 'train' scenes, hold out 'val' scenes; no val render is seen in training.
    p.add_argument("--scene-split", choices=["train", "val", "all"], default="train")
    p.add_argument("--val-scene-fraction", type=float, default=0.2)
    p.add_argument("--scene-split-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr-encoder", type=float, default=1e-6)
    p.add_argument("--lr-head", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--covis-weight", type=float, default=1.0)
    p.add_argument("--covis-pos-weight", type=float, default=None)
    # Robustify the flow loss: cap per-pixel EPE before averaging so catastrophic-flow
    # hard pairs (the 80-140px spikes) don't dominate the gradient and push the matcher
    # toward overconfident-but-wrong matches that break downstream SfM.
    p.add_argument("--epe-clamp", type=float, default=None)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--overfit-one-batch", action="store_true",
                   help="Train on a single fixed batch (correctness/overfit test).")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500,
                   help="checkpoint interval in steps; 0 disables and saves only at the end")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="/tmp/ufm_ft")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
