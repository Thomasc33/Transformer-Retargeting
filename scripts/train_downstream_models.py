#!/usr/bin/env python3
"""
Train downstream AR/RI models (SGN and MixFormer) on real data only.

This script intentionally avoids any generated data or pretrained weights and
fits the downstream heads to the label space present in the dataset (e.g., 49
actions for NTU after removing two-person actions). It saves checkpoints in the
same locations that the TMR evaluation scripts expect:
  - output/{dataset}_sgn_ar_paired/model_best.pth.tar
  - output/{dataset}_sgn_ri_paired/model_best.pth.tar
  - output/{dataset}_mixformer_ar_paired/model_best.pth.tar
  - output/{dataset}_mixformer_ri_paired/model_best.pth.tar
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root
ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets as DATASETS_CONFIG, ActionRecognitionDataset
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel


def parse_args():
    p = argparse.ArgumentParser(description="Train downstream SGN and MixFormer for AR/RI")
    p.add_argument("--dataset", choices=["ntu", "ntu120", "etri"], default="ntu")
    p.add_argument("--setting", choices=["cv", "cs"], default="cv")
    p.add_argument("--seg", type=int, default=64, help="Frames per sample")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--models", nargs="+", default=["sgn_ar", "sgn_ri", "mix_ar", "mix_ri"],
                   help="Subset of models to train")
    p.add_argument("--output_root", default="output", help="Root directory for checkpoints")
    p.add_argument("--data_path", help="Override dataset pickle path (uses config default if omitted)")
    p.add_argument("--max_batches", type=int, default=-1, help="Limit batches per epoch for quick smoke tests")
    return p.parse_args()


def load_data_dict(dataset: str, data_path: str):
    path = data_path or DATASETS_CONFIG[dataset]["path"]
    with open(path, "rb") as f:
        return pickle.load(f)


def make_loaders(data_dict, dataset: str, setting: str, seg: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, int]:
    drop_two_person = dataset in ("ntu", "ntu120")
    train_ar = ActionRecognitionDataset(
        data_dict, dataset, setting, split="train", task="ar", seg=seg, augment=True, drop_two_person_actions=drop_two_person
    )
    val_ar = ActionRecognitionDataset(
        data_dict, dataset, setting, split="test", task="ar", seg=seg, augment=False, drop_two_person_actions=drop_two_person
    )
    train_ri = ActionRecognitionDataset(data_dict, dataset, setting, split="train", task="ri", seg=seg, augment=True)
    val_ri = ActionRecognitionDataset(data_dict, dataset, setting, split="test", task="ri", seg=seg, augment=False)

    num_classes_ar = getattr(train_ar, "num_classes", int(max(lbl for _, lbl in train_ar.samples) + 1))
    num_classes_ri = getattr(train_ri, "num_classes", int(max(lbl for _, lbl in train_ri.samples) + 1))

    loaders = (
        DataLoader(train_ar, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ar, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(train_ri, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ri, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
    return (*loaders, num_classes_ar, num_classes_ri)


def prepare_mixformer_input(x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, V*C) -> (B, C, T, V, 1)
    B, T, VC = x.shape
    V = 25
    C = 3
    x = x.view(B, T, V, C).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
    return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_interval: int,
    max_batches: int,
    name: str,
    is_mixformer: bool = False,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if 0 < max_batches <= batch_idx:
                break
            x = x.to(device)  # (B, T, V*C)
            y = y.to(device)
            optimizer.zero_grad()

            if is_mixformer:
                x_in = prepare_mixformer_input(x)
            else:
                x_in = x

            logits = model(x_in)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            seen += y.size(0)

            if (batch_idx + 1) % log_interval == 0:
                print(f"[{name}] Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} "
                      f"Loss={loss.item():.4f} Acc={correct/seen:.4f}")

        train_acc = correct / max(1, seen)
        avg_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if 0 < max_batches <= batch_idx:
                    break
                x = x.to(device)
                y = y.to(device)
                if is_mixformer:
                    x_in = prepare_mixformer_input(x)
                else:
                    x_in = x
                logits = model(x_in)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_seen += y.size(0)
        val_acc = val_correct / max(1, val_seen)

        print(f"[{name}] Epoch {epoch+1}/{epochs} TrainLoss={avg_loss:.4f} TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"[{name}] ✓ New best val acc: {best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dict = load_data_dict(args.dataset, args.data_path)

    (
        train_ar_loader,
        val_ar_loader,
        train_ri_loader,
        val_ri_loader,
        num_classes_ar,
        num_classes_ri,
    ) = make_loaders(data_dict, args.dataset, args.setting, args.seg, args.batch_size, args.num_workers)

    print(f"Dataset {args.dataset} setting={args.setting} seg={args.seg}")
    print(f"  AR classes: {num_classes_ar}")
    print(f"  RI classes: {num_classes_ri}")

    models_requested = set(args.models)

    metrics = {}

    if "sgn_ar" in models_requested:
        print("\n=== Training SGN (AR) ===")
        sgn_ar = SGN(num_classes=num_classes_ar, dataset=args.dataset, seg=args.seg, bias=True)
        sgn_ar, acc = train_model(
            sgn_ar,
            train_ar_loader,
            val_ar_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.log_interval,
            args.max_batches,
            name="sgn_ar",
            is_mixformer=False,
        )
        metrics["sgn_ar"] = acc
        save_checkpoint(sgn_ar, Path(args.output_root) / f"{args.dataset}_sgn_ar_paired" / "model_best.pth.tar")

    if "sgn_ri" in models_requested:
        print("\n=== Training SGN (RI) ===")
        sgn_ri = SGN(num_classes=num_classes_ri, dataset=args.dataset, seg=args.seg, bias=True)
        sgn_ri, acc = train_model(
            sgn_ri,
            train_ri_loader,
            val_ri_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.log_interval,
            args.max_batches,
            name="sgn_ri",
            is_mixformer=False,
        )
        metrics["sgn_ri"] = acc
        save_checkpoint(sgn_ri, Path(args.output_root) / f"{args.dataset}_sgn_ri_paired" / "model_best.pth.tar")

    if "mix_ar" in models_requested:
        print("\n=== Training MixFormer (AR) ===")
        mix_ar = MixFormerModel(
            num_class=num_classes_ar,
            num_point=25,
            num_person=1,
            graph="src.graph.ntu_rgb_d.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
        )
        mix_ar, acc = train_model(
            mix_ar,
            train_ar_loader,
            val_ar_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.log_interval,
            args.max_batches,
            name="mix_ar",
            is_mixformer=True,
        )
        metrics["mix_ar"] = acc
        save_checkpoint(mix_ar, Path(args.output_root) / f"{args.dataset}_mixformer_ar_paired" / "model_best.pth.tar")

    if "mix_ri" in models_requested:
        print("\n=== Training MixFormer (RI) ===")
        mix_ri = MixFormerModel(
            num_class=num_classes_ri,
            num_point=25,
            num_person=1,
            graph="src.graph.ntu_rgb_d.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
        )
        mix_ri, acc = train_model(
            mix_ri,
            train_ri_loader,
            val_ri_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.log_interval,
            args.max_batches,
            name="mix_ri",
            is_mixformer=True,
        )
        metrics["mix_ri"] = acc
        save_checkpoint(mix_ri, Path(args.output_root) / f"{args.dataset}_mixformer_ri_paired" / "model_best.pth.tar")

    # Save metrics
    import json
    metrics_path = Path(args.output_root) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
