#!/usr/bin/env python3
"""
Cross-evaluate downstream models trained on retargeted data against raw test data.

This script loads SGN and MixFormer models that were trained on retargeted
(anonymized) skeletons and evaluates them on the original raw test set.
The purpose is to measure how well action semantics transfer: if a classifier
trained on retargeted data still recognizes actions in raw data, the
retargeting preserves useful motion structure.

Expected usage:
    python scripts/cross_evaluate_downstream.py \
        --checkpoint_root output/downstream_disentangled_tmr_stable \
        --raw_data_path data/ntu/ntu.pkl \
        --dataset ntu --setting cv

Checkpoint layout assumed (produced by train_downstream_models.py):
    <checkpoint_root>/ntu_sgn_ar_paired/model_best.pth.tar
    <checkpoint_root>/ntu_sgn_ri_paired/model_best.pth.tar
    <checkpoint_root>/ntu_mixformer_ar_paired/model_best.pth.tar
    <checkpoint_root>/ntu_mixformer_ri_paired/model_best.pth.tar
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import ActionRecognitionDataset
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-evaluate retargeted-trained downstream models on raw data"
    )
    p.add_argument(
        "--checkpoint_root",
        type=str,
        required=True,
        help="Root dir containing {dataset}_{model}_{task}_paired/ subdirs with model_best.pth.tar",
    )
    p.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help="Path to the raw dataset pickle (e.g. data/ntu/ntu.pkl)",
    )
    p.add_argument("--dataset", choices=["ntu", "ntu120", "etri"], default="ntu")
    p.add_argument("--setting", choices=["cv", "cs"], default="cv")
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save cross_eval_metrics.json (defaults to checkpoint_root)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["sgn_ar", "sgn_ri", "mix_ar", "mix_ri"],
        help="Subset of models to evaluate",
    )
    return p.parse_args()


def prepare_mixformer_input(x: torch.Tensor) -> torch.Tensor:
    """(B, T, V*C) -> (B, C, T, V, 1)"""
    B, T, VC = x.shape
    V = 25
    C = 3
    return x.view(B, T, V, C).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()


def load_checkpoint(path: Path, device: torch.device):
    """Load a downstream model checkpoint and return the state dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    # Strip 'module.' prefix from DataParallel
    clean = {}
    for k, v in state_dict.items():
        clean[k.removeprefix("module.")] = v
    return clean


def infer_num_classes(state_dict: dict) -> int:
    """Infer number of output classes from fc.weight shape."""
    if "fc.weight" in state_dict:
        return state_dict["fc.weight"].shape[0]
    raise ValueError("Cannot infer num_classes: no fc.weight in state dict")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_mixformer: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if is_mixformer:
                x = prepare_mixformer_input(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading raw data from {args.raw_data_path} ...")
    with open(args.raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    print(f"  {len(raw_data)} samples loaded")

    drop_two_person = args.dataset in ("ntu", "ntu120")

    models_requested = set(args.models)

    # Merge into existing metrics so re-running with a new model subset
    # (e.g. adding MixFormer on a dataset that only has SGN entries) does not
    # clobber previously computed numbers.
    metrics_path = output_dir / "cross_eval_metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            metrics = {}

    for model_key in ["sgn_ar", "sgn_ri", "mix_ar", "mix_ri"]:
        if model_key not in models_requested:
            continue

        is_mixformer = model_key.startswith("mix")
        task = model_key.split("_")[1]  # 'ar' or 'ri'

        # Map model key to checkpoint subdir name
        subdir_model = "mixformer" if is_mixformer else "sgn"
        ckpt_dir = Path(args.checkpoint_root) / f"{args.dataset}_{subdir_model}_{task}_paired"
        ckpt_path = ckpt_dir / "model_best.pth.tar"

        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Cross-evaluating {model_key.upper()} on raw {args.dataset} test data")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*70}")

        # Load state dict and infer num_classes
        state_dict = load_checkpoint(ckpt_path, device)
        num_classes = infer_num_classes(state_dict)
        print(f"  Model has {num_classes} output classes")

        # Build the test dataset on raw data
        drop = drop_two_person if task == "ar" else False
        test_ds = ActionRecognitionDataset(
            raw_data,
            args.dataset,
            args.setting,
            split="test",
            task=task,
            seg=args.seg,
            augment=False,
            drop_two_person_actions=drop,
        )
        ds_num_classes = test_ds.num_classes
        print(f"  Raw test set: {len(test_ds)} samples, {ds_num_classes} classes")

        if ds_num_classes != num_classes:
            print(
                f"  WARNING: model has {num_classes} classes but raw test set has "
                f"{ds_num_classes} classes. Label mismatch may affect accuracy."
            )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Create model
        if is_mixformer:
            model = MixFormerModel(
                num_class=num_classes,
                num_point=25,
                num_person=1,
                graph="src.graph.ntu_rgb_d.Graph",
                graph_args={"labeling_mode": "spatial"},
                in_channels=3,
            )
        else:
            model = SGN(
                num_classes=num_classes,
                dataset=args.dataset,
                seg=args.seg,
                bias=True,
            )

        model.load_state_dict(state_dict)
        model.to(device)

        acc = evaluate(model, test_loader, device, is_mixformer)
        metrics[f"cross_{model_key}"] = acc
        print(f"  >> {model_key.upper()} cross-eval accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("CROSS-EVALUATION SUMMARY")
    print(f"  Models trained on: retargeted data ({args.checkpoint_root})")
    print(f"  Evaluated on:      raw {args.dataset} test data ({args.raw_data_path})")
    print(f"{'='*70}")
    for k, v in metrics.items():
        label = k.replace("cross_", "").upper()
        direction = "(utility, higher=better)" if "_ar" in k else "(privacy, lower=better)"
        print(f"  {label}: {v:.4f}  {direction}")

    # Save (merged with any pre-existing entries)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
