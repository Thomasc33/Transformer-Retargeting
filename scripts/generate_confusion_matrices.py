#!/usr/bin/env python3
"""
Generate AR and RI confusion matrices for DisentangledTMR downstream models.

Evaluates SGN AR (49x49 action) and SGN RI (Nx40 identity) classifiers on the
raw NTU60 test set. Produces row-normalized heatmaps and top-confused-pairs lists.

Outputs:
  - confusion_ar.pdf: 49x49 action confusion matrix
  - confusion_ri.pdf: identity confusion matrix
  - confusion_top_pairs.json: top confused pairs
  - confusion_results.json: raw confusion data

Must be submitted via SLURM (loads models + data).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets as DATASETS_CONFIG, ActionRecognitionDataset
from src.model.sgn import SGN

# NTU60 single-person action names (1-indexed: actions 1-49)
NTU60_ACTION_NAMES = {
    1: "drink water", 2: "eat meal", 3: "brush teeth", 4: "brush hair",
    5: "drop", 6: "pick up", 7: "throw", 8: "sit down", 9: "stand up",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "put on jacket", 15: "take off jacket", 16: "put on shoe",
    17: "take off shoe", 18: "put on glasses", 19: "take off glasses",
    20: "put on hat/cap", 21: "take off hat/cap", 22: "cheer up",
    23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jump up", 28: "phone call", 29: "play with phone",
    30: "type on keyboard", 31: "point to something", 32: "taking a selfie",
    33: "check time", 34: "rub two hands", 35: "nod head/bow",
    36: "shake head", 37: "wipe face", 38: "salute",
    39: "put palms together", 40: "cross hands in front",
    41: "sneeze/cough", 42: "staggering", 43: "falling down",
    44: "headache", 45: "chest pain", 46: "back pain",
    47: "neck pain", 48: "nausea/vomiting", 49: "fan self",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ntu")
    p.add_argument("--setting", default="cv")
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="output/analysis/confusion")
    p.add_argument("--checkpoint_dir", default="output/downstream_disentangled_tmr_stable")
    p.add_argument("--data_path", default=None)
    return p.parse_args()


def load_sgn(checkpoint_path, num_classes, dataset, seg, device):
    """Load an SGN model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: {checkpoint_path} not found")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    if isinstance(state_dict, dict) and "fc.weight" in state_dict:
        ckpt_classes = state_dict["fc.weight"].shape[0]
    else:
        ckpt_classes = num_classes

    model = SGN(num_classes=ckpt_classes, dataset=dataset, seg=seg, bias=True)
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    print(f"Loaded SGN from {checkpoint_path} ({ckpt_classes} classes)")
    return model, ckpt_classes


def collect_predictions(model, dataloader, num_classes, device):
    """Run inference and collect all (true_label, pred_label) pairs."""
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            for i in range(y.size(0)):
                lbl = y[i].item()
                if 0 <= lbl < num_classes:
                    all_true.append(lbl)
                    all_pred.append(preds[i].item())

    return np.array(all_true), np.array(all_pred)


def build_confusion_matrix(true_labels, pred_labels, num_classes):
    """Build a confusion matrix (num_classes x num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def normalize_rows(cm):
    """Row-normalize confusion matrix (each row sums to 1)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # avoid division by zero
    return cm.astype(np.float64) / row_sums


def top_confused_pairs(cm_norm, n=10, class_names=None):
    """Find top-N most confused (off-diagonal) pairs."""
    pairs = []
    num_classes = cm_norm.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_norm[i, j] > 0:
                name_i = class_names.get(i, f"cls_{i}") if class_names else f"cls_{i}"
                name_j = class_names.get(j, f"cls_{j}") if class_names else f"cls_{j}"
                pairs.append({
                    "true_class": i,
                    "pred_class": j,
                    "true_name": name_i,
                    "pred_name": name_j,
                    "confusion_rate": float(cm_norm[i, j]),
                })
    pairs.sort(key=lambda x: x["confusion_rate"], reverse=True)
    return pairs[:n]


def plot_confusion_matrix(cm_norm, title, save_path, class_labels=None, figsize=None):
    """Plot a row-normalized confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = cm_norm.shape[0]
    if figsize is None:
        figsize = (max(10, n * 0.3), max(8, n * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13)

    if class_labels and n <= 50:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_labels, rotation=90, fontsize=5)
        ax.set_yticklabels(class_labels, fontsize=5)
    elif n <= 50:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    data_path = args.data_path or DATASETS_CONFIG[args.dataset]["path"]
    print(f"Loading data from {data_path}...")
    import pickle
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    results = {}

    # ========== AR Confusion Matrix ==========
    print("\n" + "=" * 60)
    print("AR Confusion Matrix")
    print("=" * 60)

    test_ar = ActionRecognitionDataset(
        data_dict, args.dataset, args.setting,
        split="test", task="ar", seg=args.seg,
        augment=False, drop_two_person_actions=True,
    )
    num_classes_ar = test_ar.num_classes
    ar_loader = DataLoader(test_ar, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    ar_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_sgn_ar_paired", "model_best.pth.tar")
    ar_result = load_sgn(ar_path, num_classes_ar, args.dataset, args.seg, device)
    if ar_result is not None:
        ar_model, ar_nc = ar_result
        true_ar, pred_ar = collect_predictions(ar_model, ar_loader, ar_nc, device)
        cm_ar = build_confusion_matrix(true_ar, pred_ar, ar_nc)
        cm_ar_norm = normalize_rows(cm_ar)

        # Build class name mapping (contiguous idx -> name)
        if hasattr(test_ar, "action_label_map") and test_ar.action_label_map is not None:
            inv_map = {v: k for k, v in test_ar.action_label_map.items()}
        else:
            inv_map = {i: i for i in range(ar_nc)}

        ar_class_names = {}
        ar_class_labels = []
        for i in range(ar_nc):
            orig_0idx = inv_map.get(i, i)
            orig_1idx = orig_0idx + 1
            name = NTU60_ACTION_NAMES.get(orig_1idx, f"A{orig_1idx}")
            ar_class_names[i] = name
            ar_class_labels.append(f"A{orig_1idx}")

        plot_confusion_matrix(
            cm_ar_norm,
            "AR Confusion Matrix (DisentangledTMR, SGN)",
            os.path.join(args.output_dir, "confusion_ar.pdf"),
            class_labels=ar_class_labels,
        )

        ar_top_pairs = top_confused_pairs(cm_ar_norm, n=20, class_names=ar_class_names)
        overall_ar = (true_ar == pred_ar).mean()
        results["ar"] = {
            "overall_accuracy": float(overall_ar),
            "num_classes": ar_nc,
            "num_samples": len(true_ar),
            "top_confused_pairs": ar_top_pairs,
        }
        print(f"AR accuracy: {overall_ar:.4f}")
        print("\nTop 10 confused action pairs:")
        for p in ar_top_pairs[:10]:
            print(f"  {p['true_name']:25s} -> {p['pred_name']:25s}: {p['confusion_rate']:.3f}")

    # ========== RI Confusion Matrix ==========
    print("\n" + "=" * 60)
    print("RI Confusion Matrix")
    print("=" * 60)

    test_ri = ActionRecognitionDataset(
        data_dict, args.dataset, args.setting,
        split="test", task="ri", seg=args.seg,
        augment=False,
    )
    num_classes_ri = test_ri.num_classes
    ri_loader = DataLoader(test_ri, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    ri_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_sgn_ri_paired", "model_best.pth.tar")
    ri_result = load_sgn(ri_path, num_classes_ri, args.dataset, args.seg, device)
    if ri_result is not None:
        ri_model, ri_nc = ri_result
        true_ri, pred_ri = collect_predictions(ri_model, ri_loader, ri_nc, device)
        cm_ri = build_confusion_matrix(true_ri, pred_ri, ri_nc)
        cm_ri_norm = normalize_rows(cm_ri)

        # RI labels are identity IDs
        ri_class_labels = [f"P{i+1}" for i in range(ri_nc)]

        plot_confusion_matrix(
            cm_ri_norm,
            "RI Confusion Matrix (DisentangledTMR, SGN)",
            os.path.join(args.output_dir, "confusion_ri.pdf"),
            class_labels=ri_class_labels,
        )

        ri_top_pairs = top_confused_pairs(cm_ri_norm, n=20)
        overall_ri = (true_ri == pred_ri).mean()
        results["ri"] = {
            "overall_accuracy": float(overall_ri),
            "num_classes": ri_nc,
            "num_samples": len(true_ri),
            "top_confused_pairs": ri_top_pairs,
        }
        print(f"RI accuracy: {overall_ri:.4f}")
        print("\nTop 10 confused identity pairs:")
        for p in ri_top_pairs[:10]:
            print(f"  P{p['true_class']+1} -> P{p['pred_class']+1}: {p['confusion_rate']:.3f}")

    # Save results
    json_path = os.path.join(args.output_dir, "confusion_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
