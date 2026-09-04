"""
Evaluate UFM (stock or fine-tuned) on held-out masked object-flow pairs.

Reports, over the held-out datasets (default HANDAL + HOPE -- never trained on):
  - EPE (mean endpoint error, px) on covisible object pixels
  - flow accuracy @{1,3,5}px (fraction of covisible pixels under threshold)
  - covisibility head accuracy / IoU (occlusion prediction)

Use it to compare stock UFM-Refine vs a fine-tuned checkpoint on the SAME pairs:
  python -m training.ufm.eval_ufm --datasets handal hope --num-batches 100
  python -m training.ufm.eval_ufm --datasets handal hope --checkpoint /path/ufm_ft_last.pth
"""

from __future__ import annotations

import os

import argparse

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from training.ufm.bop_flow_dataset import (
    BopFlowDatasetConfig, BopPbrFlowDataset, collate_drop_meta, HELDOUT_EVAL_DATASETS,
)
from training.ufm.losses import endpoint_error, covisibility_metrics
from training.ufm.train_ufm import build_model, get_normalizer, make_views, to_device


def build_eval_dataset(args) -> ConcatDataset:
    datasets = []
    for ds in args.datasets:
        cfg = BopFlowDatasetConfig(
            bop_root=args.bop_root, datasets=[ds], split=args.split,
            width=args.width, height=args.height,
            min_visib_fract=args.min_visib_fract,
            max_groups_per_dataset=args.max_groups_per_dataset, seed=args.seed,
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


@torch.no_grad()
def evaluate(args):
    device = args.device
    ds = build_eval_dataset(args)
    print(f"[eval] {len(ds)} groups over {args.datasets}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_drop_meta,
                        drop_last=True)

    model = build_model(device)
    tag = "stock UFM-Refine"
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        tag = f"checkpoint {args.checkpoint}"
    model.eval()
    mean, std = get_normalizer(model, device)

    epe_sum = w_sum = 0.0
    acc = {1: 0.0, 3: 0.0, 5: 0.0}
    covis_acc_list, covis_iou_list = [], []
    n_batches = 0
    for batch in loader:
        if n_batches >= args.num_batches:
            break
        batch = to_device(batch, device)
        view1, view2 = make_views(batch, model, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(view1, view2)
        pred_flow = result.flow.flow_output.float()
        valid = batch["flow_valid"]
        if valid.sum() == 0:
            continue
        epe = endpoint_error(pred_flow, batch["gt_flow"], valid)      # (B,H,W)
        w = valid.sum().item()
        epe_sum += (epe * valid).sum().item()
        w_sum += w
        for thr in acc:
            acc[thr] += (((epe < thr).float() * valid).sum().item())
        if result.covisibility is not None and result.covisibility.logits is not None:
            m = covisibility_metrics(result.covisibility.logits.float(), batch["covisibility"])
            covis_acc_list.append(m["covis_acc"])
            covis_iou_list.append(m["covis_iou"])
        n_batches += 1

    print(f"\n===== EVAL: {tag} =====")
    print(f"batches={n_batches}  covisible px={int(w_sum)}")
    if w_sum > 0:
        print(f"EPE (covisible)      : {epe_sum / w_sum:.3f} px")
        for thr in (1, 3, 5):
            print(f"flow acc @{thr}px        : {100 * acc[thr] / w_sum:.1f} %")
    if covis_acc_list:
        print(f"covisibility accuracy: {100 * np.mean(covis_acc_list):.1f} %")
        print(f"covisibility IoU     : {100 * np.mean(covis_iou_list):.1f} %")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--bop-root", default=os.environ.get("OMA_DATA_ROOT", "data") + "/bop")
    p.add_argument("--datasets", nargs="+", default=list(HELDOUT_EVAL_DATASETS))
    p.add_argument("--split", default="train_pbr")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--min-visib-fract", type=float, default=0.30)
    p.add_argument("--max-groups-per-dataset", type=int, default=None)
    # 'val' = held-out objects of the TRAIN datasets (object-disjoint dev set for model
    # selection). 'all' = whole dataset (use for the final HANDAL/HOPE test).
    p.add_argument("--obj-split", choices=["train", "val", "all"], default="all")
    p.add_argument("--val-obj-fraction", type=float, default=0.2)
    p.add_argument("--obj-split-seed", type=int, default=1234)
    # Scene-disjoint val: 'val' = held-out RENDERS of the TRAIN datasets (image-disjoint
    # dev set, never trained on). 'all' = whole dataset (for the HANDAL/HOPE test).
    p.add_argument("--scene-split", choices=["train", "val", "all"], default="val")
    p.add_argument("--val-scene-fraction", type=float, default=0.2)
    p.add_argument("--scene-split-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
