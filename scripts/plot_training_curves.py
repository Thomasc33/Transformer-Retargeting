#!/usr/bin/env python3
"""
Plot training curves from training log JSON files.

Reads JSONL training logs and generates multi-panel figures showing:
- Loss curves across all 3 stages
- AR/RI accuracy evolution
- Disentanglement metric evolution (contrastive, MI, orthogonality, adversarial)
- Stage boundaries marked with vertical lines

Can run on the login node (just JSON parsing + matplotlib, no GPU/models).

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --log_path output/disentangled_tmr_stable/disentangled_tmr_ntu_log.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_path",
                   default="output/disentangled_tmr_stable/disentangled_tmr_ntu_log.json")
    p.add_argument("--output_dir", default="output/analysis/training_curves")
    p.add_argument("--additional_logs", nargs="*", default=[],
                   help="Additional log files to overlay (e.g., ablation runs)")
    return p.parse_args()


def load_log(path):
    """Load JSONL log file into a list of event dicts."""
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def extract_epoch_data(events):
    """Extract epoch-end events organized by stage.

    Returns:
        list of dicts with keys: stage, epoch, global_epoch, train_metrics, validation_metrics
    """
    epoch_data = []
    stage_starts = {}
    global_epoch = 0

    for ev in events:
        if ev.get("event_type") == "stage_start":
            stage_starts[ev["stage"]] = global_epoch

        if ev.get("event_type") == "epoch_end":
            entry = {
                "stage": ev["stage"],
                "epoch": ev["epoch"],
                "global_epoch": global_epoch,
                "train_metrics": ev.get("train_metrics", {}),
                "validation_metrics": ev.get("validation_metrics", {}),
                "is_best": ev.get("is_best", False),
            }
            epoch_data.append(entry)
            global_epoch += 1

    return epoch_data, stage_starts


def extract_batch_data(events):
    """Extract batch-level loss data for smoother curves."""
    batch_data = []
    global_step = 0

    for ev in events:
        if ev.get("event_type") == "batch_log":
            entry = {
                "stage": ev["stage"],
                "epoch": ev["epoch"],
                "batch": ev["batch"],
                "global_step": global_step,
                "losses": ev.get("losses", {}),
                "metrics": ev.get("metrics", {}),
            }
            batch_data.append(entry)
            global_step += 1

    return batch_data


def plot_curves(epoch_data, stage_starts, output_dir, label="stable"):
    """Generate multi-panel training curve figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not epoch_data:
        print("No epoch data found.")
        return

    # Organize data
    global_epochs = [d["global_epoch"] for d in epoch_data]
    stages = [d["stage"] for d in epoch_data]

    # Find stage boundaries
    stage_boundaries = []
    prev_stage = epoch_data[0]["stage"]
    for d in epoch_data:
        if d["stage"] != prev_stage:
            stage_boundaries.append(d["global_epoch"] - 0.5)
            prev_stage = d["stage"]

    # Colors per stage
    stage_colors = {1: "#2196F3", 2: "#FF9800", 3: "#4CAF50"}

    # ==========================================
    # Figure 1: 2x2 main training curves
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0,0): Total loss
    ax = axes[0, 0]
    losses_keys = ["ar_loss", "ri_loss", "contrastive_loss"]
    for key in losses_keys:
        vals = [d["train_metrics"].get(key, None) for d in epoch_data]
        valid = [(ge, v) for ge, v in zip(global_epochs, vals) if v is not None]
        if valid:
            geps, vs = zip(*valid)
            ax.plot(geps, vs, label=key.replace("_", " ").title(), alpha=0.8)
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for b in stage_boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5)

    # Panel (0,1): AR and RI accuracy
    ax = axes[0, 1]
    for key, color, ls in [("ar_accuracy", "#4CAF50", "-"), ("ri_accuracy", "#F44336", "-")]:
        train_vals = [d["train_metrics"].get(key, None) for d in epoch_data]
        val_vals = [d["validation_metrics"].get(key, None) for d in epoch_data]

        valid_train = [(ge, v) for ge, v in zip(global_epochs, train_vals) if v is not None]
        valid_val = [(ge, v) for ge, v in zip(global_epochs, val_vals) if v is not None]

        lbl = key.replace("_", " ").upper().replace("ACCURACY", "Acc")
        if valid_train:
            geps, vs = zip(*valid_train)
            ax.plot(geps, vs, color=color, linestyle=ls, alpha=0.5, label=f"Train {lbl}")
        if valid_val:
            geps, vs = zip(*valid_val)
            ax.plot(geps, vs, color=color, linestyle="--", linewidth=2, label=f"Val {lbl}")

    ax.set_ylabel("Accuracy")
    ax.set_title("AR & RI Accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for b in stage_boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5)

    # Panel (1,0): Disentanglement losses
    ax = axes[1, 0]
    disent_keys = ["contrastive_loss", "adversarial_loss", "orthogonality_loss", "mutual_info_loss"]
    for key in disent_keys:
        vals = [d["train_metrics"].get(key, None) for d in epoch_data]
        valid = [(ge, v) for ge, v in zip(global_epochs, vals) if v is not None]
        if valid:
            geps, vs = zip(*valid)
            ax.plot(geps, vs, label=key.replace("_", " ").title(), alpha=0.8)
    ax.set_xlabel("Epoch (global)")
    ax.set_ylabel("Loss")
    ax.set_title("Disentanglement Losses")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for b in stage_boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5)

    # Panel (1,1): Reconstruction losses (Stage 2+3 only)
    ax = axes[1, 1]
    recon_keys = ["total_loss"]
    for key in recon_keys:
        vals = [d["train_metrics"].get(key, None) for d in epoch_data]
        valid = [(ge, v) for ge, v in zip(global_epochs, vals) if v is not None]
        if valid:
            geps, vs = zip(*valid)
            ax.plot(geps, vs, label=key.replace("_", " ").title(), alpha=0.8, color="#9C27B0")
    ax.set_xlabel("Epoch (global)")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for b in stage_boundaries:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.5)

    # Add stage labels
    for ax_row in axes:
        for ax in ax_row:
            # Label stages at top
            y_top = ax.get_ylim()[1]
            if len(stage_boundaries) >= 1:
                ax.text(stage_boundaries[0] / 2, y_top * 0.95, "Stage 1",
                        ha="center", fontsize=9, color="#2196F3", fontweight="bold")
            if len(stage_boundaries) >= 2:
                mid = (stage_boundaries[0] + stage_boundaries[1]) / 2
                ax.text(mid, y_top * 0.95, "Stage 2",
                        ha="center", fontsize=9, color="#FF9800", fontweight="bold")
                ax.text((stage_boundaries[1] + max(global_epochs)) / 2, y_top * 0.95,
                        "Stage 3", ha="center", fontsize=9, color="#4CAF50", fontweight="bold")
            elif len(stage_boundaries) == 1:
                ax.text((stage_boundaries[0] + max(global_epochs)) / 2, y_top * 0.95,
                        "Stage 2+", ha="center", fontsize=9, color="#FF9800", fontweight="bold")

    fig.suptitle(f"DisentangledTMR Training Curves ({label})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves_main.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training_curves_main.pdf")

    # ==========================================
    # Figure 2: Per-stage detailed view
    # ==========================================
    unique_stages = sorted(set(stages))
    n_stages = len(unique_stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 5))
    if n_stages == 1:
        axes = [axes]

    for idx, stage in enumerate(unique_stages):
        ax = axes[idx]
        stage_epochs = [d for d in epoch_data if d["stage"] == stage]
        local_epochs = [d["epoch"] for d in stage_epochs]

        # Plot key metrics for this stage
        for key, color in [("ar_accuracy", "#4CAF50"), ("ri_accuracy", "#F44336")]:
            val_vals = [d["validation_metrics"].get(key, None) for d in stage_epochs]
            valid = [(e, v) for e, v in zip(local_epochs, val_vals) if v is not None]
            if valid:
                eps, vs = zip(*valid)
                lbl = "AR" if "ar" in key else "RI"
                ax.plot(eps, vs, color=color, marker="o", markersize=3, label=f"Val {lbl}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Stage {stage}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Mark best epoch
        best_epochs = [d["epoch"] for d in stage_epochs if d.get("is_best")]
        for be in best_epochs:
            ax.axvline(be, color="gold", linestyle=":", alpha=0.5)

    fig.suptitle("Per-Stage Validation Accuracy", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves_per_stage.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training_curves_per_stage.pdf")

    # ==========================================
    # Summary stats
    # ==========================================
    summary = {"stages": {}}
    for stage in unique_stages:
        stage_epochs = [d for d in epoch_data if d["stage"] == stage]
        best = [d for d in stage_epochs if d.get("is_best")]

        val_ar = [d["validation_metrics"].get("ar_accuracy", 0) for d in stage_epochs]
        val_ri = [d["validation_metrics"].get("ri_accuracy", 0) for d in stage_epochs]

        summary["stages"][str(stage)] = {
            "num_epochs": len(stage_epochs),
            "best_val_ar": max(val_ar) if val_ar else None,
            "best_val_ri": max(val_ri) if val_ri else None,
            "final_val_ar": val_ar[-1] if val_ar else None,
            "final_val_ri": val_ri[-1] if val_ri else None,
        }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training_summary.json")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading log from {args.log_path}...")
    events = load_log(args.log_path)
    print(f"Loaded {len(events)} events")

    epoch_data, stage_starts = extract_epoch_data(events)
    print(f"Found {len(epoch_data)} epoch-end events across stages {set(d['stage'] for d in epoch_data)}")

    plot_curves(epoch_data, stage_starts, args.output_dir, label="stable")

    # Overlay additional logs if provided
    if args.additional_logs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
        all_log_paths = [args.log_path] + args.additional_logs

        for i, log_path in enumerate(all_log_paths):
            if not os.path.exists(log_path):
                print(f"Skipping {log_path} (not found)")
                continue

            evts = load_log(log_path)
            ed, _ = extract_epoch_data(evts)
            if not ed:
                continue

            label = Path(log_path).parent.name
            geps = [d["global_epoch"] for d in ed]
            val_ar = [d["validation_metrics"].get("ar_accuracy", 0) for d in ed]
            color = colors[i % len(colors)]
            ax.plot(geps, val_ar, color=color, label=label, alpha=0.8)

        ax.set_xlabel("Epoch (global)")
        ax.set_ylabel("Val AR Accuracy")
        ax.set_title("Comparison: Validation AR Across Runs")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "training_curves_comparison.pdf"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved training_curves_comparison.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
