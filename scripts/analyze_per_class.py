#!/usr/bin/env python3
"""
Per-action-class analysis of AR performance across all methods.

Loads SGN AR models trained on each method's data (Raw, Noise, DMR, PMR, DisentangledTMR)
and evaluates them on the raw NTU60 test set, reporting per-class accuracy.

Outputs:
  - per_class_results.json: Full per-class breakdown
  - per_class_bar_chart.pdf: Grouped bar chart of top/bottom actions
  - per_class_summary.txt: Human-readable summary

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
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets as DATASETS_CONFIG, ActionRecognitionDataset
from src.model.sgn import SGN

# NTU60 single-person action names (1-indexed: actions 1-49)
NTU60_ACTION_NAMES = {
    1: "drink water",
    2: "eat meal",
    3: "brush teeth",
    4: "brush hair",
    5: "drop",
    6: "pick up",
    7: "throw",
    8: "sit down",
    9: "stand up",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "put on jacket",
    15: "take off jacket",
    16: "put on shoe",
    17: "take off shoe",
    18: "put on glasses",
    19: "take off glasses",
    20: "put on hat/cap",
    21: "take off hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping",
    27: "jump up",
    28: "phone call",
    29: "play with phone",
    30: "type on keyboard",
    31: "point to something",
    32: "taking a selfie",
    33: "check time (watch)",
    34: "rub two hands",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put palms together",
    40: "cross hands in front",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling down",
    44: "headache",
    45: "chest pain",
    46: "back pain",
    47: "neck pain",
    48: "nausea/vomiting",
    49: "fan self",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ntu", choices=["ntu"])
    p.add_argument("--setting", default="cv")
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="output/analysis/per_class")
    p.add_argument("--data_path", default=None,
                   help="Override raw data pickle path")
    return p.parse_args()


# Method configs: name -> checkpoint directory
METHODS = {
    "Raw": "output/downstream_ntu60_raw",
    "Noise": "output/downstream_noise_baseline",
    "DMR": "output/downstream_dmr",
    "PMR": "output/downstream_pmr",
    "Ours": "output/downstream_disentangled_tmr_stable",
}


def load_sgn_ar(checkpoint_dir, num_classes, dataset, seg, device):
    """Load an SGN AR model from a checkpoint directory."""
    path = os.path.join(checkpoint_dir, f"{dataset}_sgn_ar_paired", "model_best.pth.tar")
    if not os.path.exists(path):
        print(f"WARNING: checkpoint not found at {path}")
        return None

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # Detect num_classes from checkpoint
    if isinstance(state_dict, dict) and "fc.weight" in state_dict:
        ckpt_classes = state_dict["fc.weight"].shape[0]
    else:
        ckpt_classes = num_classes

    model = SGN(num_classes=ckpt_classes, dataset=dataset, seg=seg, bias=True)

    # Strip 'module.' prefix if present
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace("module.", "")] = v
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    print(f"Loaded SGN AR from {path} ({ckpt_classes} classes)")
    return model


def evaluate_per_class(model, dataloader, num_classes, device):
    """Evaluate model and return per-class accuracy dict.

    Returns:
        dict mapping class_idx -> {"correct": int, "total": int, "accuracy": float}
    """
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)  # (B, T, V*C)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            for i in range(y.size(0)):
                cls = y[i].item()
                if 0 <= cls < num_classes:
                    total_per_class[cls] += 1
                    if preds[i].item() == cls:
                        correct_per_class[cls] += 1

    results = {}
    for cls in sorted(total_per_class.keys()):
        total = total_per_class[cls]
        correct = correct_per_class[cls]
        acc = correct / total if total > 0 else 0.0
        results[cls] = {"correct": correct, "total": total, "accuracy": acc}
    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data for evaluation
    data_path = args.data_path or DATASETS_CONFIG[args.dataset]["path"]
    print(f"Loading data from {data_path}...")
    import pickle
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    # Build test set (AR task, cross-view, drop two-person actions for NTU60)
    test_ds = ActionRecognitionDataset(
        data_dict, args.dataset, args.setting,
        split="test", task="ar", seg=args.seg,
        augment=False, drop_two_person_actions=True,
    )
    num_classes = test_ds.num_classes
    print(f"Test set: {len(test_ds)} samples, {num_classes} classes")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Evaluate each method
    all_results = {}
    for method_name, ckpt_dir in METHODS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        model = load_sgn_ar(ckpt_dir, num_classes, args.dataset, args.seg, device)
        if model is None:
            print(f"Skipping {method_name} (no checkpoint)")
            continue

        per_class = evaluate_per_class(model, test_loader, num_classes, device)
        overall_acc = sum(v["correct"] for v in per_class.values()) / max(1, sum(v["total"] for v in per_class.values()))
        all_results[method_name] = {
            "per_class": per_class,
            "overall_accuracy": overall_acc,
        }
        print(f"{method_name} overall accuracy: {overall_acc:.4f}")

    # Build a per-class table with action names
    # ActionRecognitionDataset uses action_label_map to remap labels.
    # The action_label_map maps original 0-indexed action -> contiguous index.
    # We need the inverse: contiguous index -> original 1-indexed action.
    if hasattr(test_ds, "action_label_map") and test_ds.action_label_map is not None:
        inv_map = {v: k for k, v in test_ds.action_label_map.items()}
    else:
        inv_map = {i: i for i in range(num_classes)}

    # Build table
    class_table = []
    for cls_idx in range(num_classes):
        orig_action_0idx = inv_map.get(cls_idx, cls_idx)
        orig_action_1idx = orig_action_0idx + 1
        action_name = NTU60_ACTION_NAMES.get(orig_action_1idx, f"action_{orig_action_1idx}")

        row = {
            "class_idx": cls_idx,
            "action_id": orig_action_1idx,
            "action_name": action_name,
        }
        for method_name in METHODS:
            if method_name in all_results:
                pc = all_results[method_name]["per_class"]
                if cls_idx in pc:
                    row[f"{method_name}_acc"] = pc[cls_idx]["accuracy"]
                    row[f"{method_name}_total"] = pc[cls_idx]["total"]
                else:
                    row[f"{method_name}_acc"] = 0.0
                    row[f"{method_name}_total"] = 0
        class_table.append(row)

    # Compute Spearman rank correlation: Raw accuracy vs each method's accuracy
    from scipy.stats import spearmanr

    correlations = {}
    if "Raw" in all_results:
        raw_accs = [row.get("Raw_acc", 0.0) for row in class_table]
        for method_name in METHODS:
            if method_name == "Raw" or method_name not in all_results:
                continue
            method_accs = [row.get(f"{method_name}_acc", 0.0) for row in class_table]
            rho, pval = spearmanr(raw_accs, method_accs)
            correlations[method_name] = {"spearman_rho": rho, "p_value": pval}
            print(f"Spearman(Raw, {method_name}): rho={rho:.4f}, p={pval:.6f}")

    # Save JSON results
    output = {
        "class_table": class_table,
        "overall": {m: all_results[m]["overall_accuracy"] for m in all_results},
        "correlations": correlations,
        "num_classes": num_classes,
        "num_test_samples": len(test_ds),
    }
    json_path = os.path.join(args.output_dir, "per_class_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Identify top-5 best and worst preserved actions for Ours
    if "Ours" in all_results:
        sorted_by_ours = sorted(class_table, key=lambda r: r.get("Ours_acc", 0.0), reverse=True)
        print("\n--- Top 5 best-preserved actions (Ours) ---")
        for row in sorted_by_ours[:5]:
            print(f"  A{row['action_id']:2d} {row['action_name']:25s} "
                  f"Raw={row.get('Raw_acc', 0):.3f}  Ours={row.get('Ours_acc', 0):.3f}")
        print("\n--- Top 5 worst-preserved actions (Ours) ---")
        for row in sorted_by_ours[-5:]:
            print(f"  A{row['action_id']:2d} {row['action_name']:25s} "
                  f"Raw={row.get('Raw_acc', 0):.3f}  Ours={row.get('Ours_acc', 0):.3f}")

    # Save human-readable summary
    summary_path = os.path.join(args.output_dir, "per_class_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Per-Action-Class AR Accuracy Analysis\n")
        f.write("=" * 100 + "\n\n")

        # Overall
        f.write("Overall Accuracy:\n")
        for m in METHODS:
            if m in all_results:
                f.write(f"  {m:20s}: {all_results[m]['overall_accuracy']:.4f}\n")
        f.write("\n")

        # Correlations
        f.write("Spearman Rank Correlations (vs Raw):\n")
        for m, c in correlations.items():
            f.write(f"  {m:20s}: rho={c['spearman_rho']:.4f}, p={c['p_value']:.6f}\n")
        f.write("\n")

        # Full table
        header = f"{'AID':>4s} {'Action Name':25s}"
        for m in METHODS:
            if m in all_results:
                header += f" {m:>8s}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in class_table:
            line = f"{row['action_id']:4d} {row['action_name']:25s}"
            for m in METHODS:
                if m in all_results:
                    line += f" {row.get(f'{m}_acc', 0.0):8.3f}"
            f.write(line + "\n")

    print(f"Saved summary to {summary_path}")

    # Generate grouped bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Sort by Ours accuracy to find top/bottom 10
        if "Ours" in all_results:
            sorted_table = sorted(class_table, key=lambda r: r.get("Ours_acc", 0.0), reverse=True)
            selected = sorted_table[:10] + sorted_table[-10:]
            # Remove duplicates if < 20 classes
            seen = set()
            unique_selected = []
            for r in selected:
                if r["class_idx"] not in seen:
                    seen.add(r["class_idx"])
                    unique_selected.append(r)
            selected = unique_selected
        else:
            selected = class_table[:20]

        # Sort selected by Ours accuracy descending
        selected = sorted(selected, key=lambda r: r.get("Ours_acc", 0.0), reverse=True)

        fig, ax = plt.subplots(figsize=(16, 7))
        x = np.arange(len(selected))
        width = 0.15
        method_colors = {
            "Raw": "#2196F3",
            "Noise": "#9E9E9E",
            "DMR": "#FF9800",
            "PMR": "#F44336",
            "Ours": "#4CAF50",
        }
        offsets = {"Raw": -2, "Noise": -1, "DMR": 0, "PMR": 1, "Ours": 2}

        for method_name in METHODS:
            if method_name not in all_results:
                continue
            vals = [r.get(f"{method_name}_acc", 0.0) for r in selected]
            offset = offsets[method_name]
            ax.bar(x + offset * width, vals, width,
                   label=method_name, color=method_colors[method_name], alpha=0.85)

        ax.set_xlabel("Action Class", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Per-Action AR Accuracy: Top 10 & Bottom 10 (by DisentangledTMR)", fontsize=13)
        labels = [f"A{r['action_id']}\n{r['action_name'][:12]}" for r in selected]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(args.output_dir, "per_class_bar_chart.pdf")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {fig_path}")

        # Also make a scatter plot: Raw accuracy vs Ours accuracy
        if "Raw" in all_results and "Ours" in all_results:
            fig2, ax2 = plt.subplots(figsize=(7, 7))
            raw_vals = [r.get("Raw_acc", 0.0) for r in class_table]
            ours_vals = [r.get("Ours_acc", 0.0) for r in class_table]
            ax2.scatter(raw_vals, ours_vals, c="#4CAF50", s=40, alpha=0.7)
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
            ax2.set_xlabel("Raw Accuracy", fontsize=12)
            ax2.set_ylabel("DisentangledTMR Accuracy", fontsize=12)
            ax2.set_title("Per-Action: Raw vs DisentangledTMR", fontsize=13)
            if "Ours" in correlations:
                rho = correlations["Ours"]["spearman_rho"]
                ax2.annotate(f"Spearman rho = {rho:.3f}", xy=(0.05, 0.92),
                             xycoords="axes fraction", fontsize=11)
            ax2.set_xlim(0, 1.05)
            ax2.set_ylim(0, 1.05)
            ax2.set_aspect("equal")
            ax2.grid(alpha=0.3)
            plt.tight_layout()

            scatter_path = os.path.join(args.output_dir, "raw_vs_ours_scatter.pdf")
            fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved scatter to {scatter_path}")

    except ImportError as e:
        print(f"Matplotlib not available, skipping figures: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
