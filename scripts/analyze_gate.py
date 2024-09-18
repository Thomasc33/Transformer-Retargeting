#!/usr/bin/env python3
"""
Analyze the Factorized Decoder's Adaptive Fusion Gate Values

Loads the Stage 3 checkpoint, runs inference on validation data, captures gate
values from every AdaptiveFusionLayer in the FactorizedDecoder, and produces
publication-quality analysis plots:

  1. Histogram of gate values across all decoder layers
  2. Heatmap of gate values per joint (which joints rely on action vs identity)
  3. Gate values over time (how the gate evolves across 64 frames)
  4. Gate values by action class (do different actions use different patterns)

Gate semantics:
  gate=1 -> decoder relies fully on action stream
  gate=0 -> decoder relies fully on identity stream

Outputs saved to paper/fig/gate_*.{pdf,png} at 300 DPI.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.data.datasets import Cross_Data

# ---------------------------------------------------------------------------
# NTU RGB+D 25-joint names (0-indexed)
# ---------------------------------------------------------------------------
NTU_JOINT_NAMES = [
    "Spine Base",       # 0
    "Mid Spine",        # 1
    "Neck",             # 2
    "Head",             # 3
    "R Shoulder",       # 4
    "R Elbow",          # 5
    "R Wrist",          # 6
    "R Hand",           # 7
    "L Shoulder",       # 8
    "L Elbow",          # 9
    "L Wrist",          # 10
    "L Hand",           # 11
    "R Hip",            # 12
    "R Knee",           # 13
    "R Ankle",          # 14
    "R Foot",           # 15
    "L Hip",            # 16
    "L Knee",           # 17
    "L Ankle",          # 18
    "L Foot",           # 19
    "Spine",            # 20
    "R Hand Tip",       # 21
    "R Thumb",          # 22
    "L Hand Tip",       # 23
    "L Thumb",          # 24
]

# Body-part groups for colouring
JOINT_GROUPS = {
    "Torso":     [0, 1, 2, 3, 20],
    "R Arm":     [4, 5, 6, 7, 21, 22],
    "L Arm":     [8, 9, 10, 11, 23, 24],
    "R Leg":     [12, 13, 14, 15],
    "L Leg":     [16, 17, 18, 19],
}

GROUP_COLORS = {
    "Torso":  "#4C72B0",
    "R Arm":  "#DD8452",
    "L Arm":  "#55A868",
    "R Leg":  "#C44E52",
    "L Leg":  "#8172B3",
}

# NTU RGB+D 60 action names (single-person subset: 49 actions after removing
# two-person actions 50-60). 0-indexed.
NTU_ACTION_NAMES = {
    0: "Drink water", 1: "Eat meal", 2: "Brush teeth", 3: "Brush hair",
    4: "Drop", 5: "Pick up", 6: "Throw", 7: "Sit down", 8: "Stand up",
    9: "Clapping", 10: "Reading", 11: "Writing", 12: "Tear up paper",
    13: "Wear jacket", 14: "Take off jacket", 15: "Wear shoe",
    16: "Take off shoe", 17: "Wear glasses", 18: "Take off glasses",
    19: "Put on hat/cap", 20: "Take off hat/cap", 21: "Cheer up",
    22: "Hand waving", 23: "Kicking", 24: "Reach into pocket",
    25: "Hopping", 26: "Jump up", 27: "Phone call", 28: "Play w/ phone",
    29: "Type on keyboard", 30: "Point to something", 31: "Take selfie",
    32: "Check time (watch)", 33: "Rub hands", 34: "Nod head/bow",
    35: "Shake head", 36: "Wipe face", 37: "Salute", 38: "Put palms together",
    39: "Cross hands in front", 40: "Sneeze/cough", 41: "Staggering",
    42: "Falling down", 43: "Headache", 44: "Chest pain", 45: "Back pain",
    46: "Neck pain", 47: "Nausea/vomiting", 48: "Fan self",
}


# ============================================================================
# Model loading (mirrors generate_retargeted_dataset.py)
# ============================================================================

def load_model(checkpoint_path, device, dataset_name):
    """Load the trained DisentangledTMR model from a Stage 3 checkpoint."""
    print(f"Loading model from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None and isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    d_action  = getattr(ckpt_args, "d_action",  768)  if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model    = getattr(ckpt_args, "d_model",    320) if ckpt_args else 320

    from src.data.datasets import datasets as ds_cfg
    num_class = ds_cfg[dataset_name]["num_class"]

    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    model = create_disentangled_tmr(
        dataset=dataset_name,
        num_class=num_class,
        device=device,
        d_action=d_action,
        d_identity=d_identity,
        d_model=d_model,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args else True,
        identity_use_full_sequence=(getattr(ckpt_args, "identity_mode", "static") == "full_seq") if ckpt_args else False,
        tokenizer_type=tokenizer,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  Model loaded.  Decoder has {len(model.decoder.layers)} layers.")
    return model


# ============================================================================
# Gate capture via forward hooks
# ============================================================================

class GateCapture:
    """
    Registers forward hooks on every AdaptiveFusionLayer inside the decoder to
    capture the sigmoid gate tensor at each call.

    Stored gate tensors have shape (T, B, D_model) where gate[i,j,k] in [0,1].
    """

    def __init__(self, model):
        self.gates = []       # list of dicts per forward call
        self._hooks = []
        self._current = {}    # layer_idx -> gate tensor (filled during one fwd)

        for layer_idx, layer in enumerate(model.decoder.layers):
            fusion = layer.fusion  # AdaptiveFusionLayer
            hook = fusion.gate.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        """Return a hook function that captures the output of nn.Sequential gate."""
        def hook_fn(module, inp, out):
            # `out` is the sigmoid gate tensor: (T, B, D) or (B, D)
            self._current[layer_idx] = out.detach().cpu()
        return hook_fn

    def reset_batch(self):
        self._current = {}

    def save_batch(self):
        if self._current:
            self.gates.append(dict(self._current))
        self._current = {}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ============================================================================
# Data loading
# ============================================================================

def load_val_data(data_path, batch_size, num_workers):
    """Load the validation split from the paired .pt file."""
    print(f"Loading data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)
    val_dataset = data["test"]

    # Disable augmentation for deterministic analysis
    if hasattr(val_dataset, "augment"):
        val_dataset.augment = False

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Validation samples: {len(val_dataset)}")
    return val_loader, val_dataset


# ============================================================================
# Inference loop
# ============================================================================

@torch.no_grad()
def collect_gates(model, val_loader, device, max_batches=None):
    """
    Run inference on the validation set, collecting gate values and metadata.

    Returns:
        gate_data : dict[int, list[np.ndarray]]
            Mapping layer_idx -> list of gate arrays, each (T-1, B, D)
        action_labels : np.ndarray  (N,)   0-indexed action class per sample
        actor_labels  : np.ndarray  (N,)   actor id per sample
    """
    capturer = GateCapture(model)
    gate_data = defaultdict(list)
    all_actions = []
    all_actors  = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Collecting gates")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Unpack: x1 (P1,A1), x2 (P2,A2), y1 (P1,A2), y2 (P2,A1), actors, actions
        x1, x2, y1, y2, actors, actions = batch

        # x1: source motion (B, C, T, V) -- no M dim from Cross_Data
        x1 = x1.to(device)
        x2 = x2.to(device)

        # Add person dim if missing: (B,C,T,V) -> (B,C,T,V,1)
        if x1.dim() == 4:
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)

        capturer.reset_batch()

        # Forward pass: source_motion=x1 (P1,A1), target_skeleton=x2 (P2,A2)
        # Retarget action A1 onto identity P2
        model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

        capturer.save_batch()

        # Store gate data per layer
        for layer_idx, gate_tensor in capturer.gates[-1].items():
            gate_data[layer_idx].append(gate_tensor.numpy())

        # actions[:, 0] = a1 (source action class)
        all_actions.append(actions[:, 0].numpy())
        all_actors.append(actors[:, 0].numpy())

    capturer.remove_hooks()

    action_labels = np.concatenate(all_actions, axis=0)
    actor_labels  = np.concatenate(all_actors, axis=0)

    return dict(gate_data), action_labels, actor_labels


# ============================================================================
# Analysis helpers
# ============================================================================

def gates_to_per_joint(gate_arrays, d_model, num_joints=25, in_channels=3):
    """
    The decoder d_model dimension does not directly correspond 1:1 to joints --
    the output projection maps d_model -> C*V.  However the gate operates in
    d_model space before the output projection, so we can still analyse the
    *average* gate value per joint by grouping d_model dimensions evenly.

    We partition the d_model dimension into num_joints groups and average within
    each group.  This gives a proxy for per-joint reliance on action vs identity.

    Returns: (N_total, T, V) array
    """
    # Concatenate all batches: each element is (T, B, D)
    all_gates = np.concatenate(gate_arrays, axis=1)  # (T, N_total, D)
    T, N, D = all_gates.shape

    # Group d_model dims into V groups
    dims_per_joint = D // num_joints
    remainder = D % num_joints

    per_joint = np.zeros((T, N, num_joints), dtype=np.float32)
    start = 0
    for j in range(num_joints):
        end = start + dims_per_joint + (1 if j < remainder else 0)
        per_joint[:, :, j] = all_gates[:, :, start:end].mean(axis=-1)
        start = end

    # Transpose to (N, T, V)
    return per_joint.transpose(1, 0, 2)


def gates_flat(gate_arrays):
    """
    Return all gate values as a flat 1-D array (for histogram).
    Each element of gate_arrays is (T, B, D).
    """
    all_gates = np.concatenate(gate_arrays, axis=1)  # (T, N, D)
    return all_gates.ravel()


def gates_over_time(gate_arrays):
    """
    Return mean gate value per timestep, averaged over samples and d_model.
    Returns: (T,) array
    """
    all_gates = np.concatenate(gate_arrays, axis=1)  # (T, N, D)
    return all_gates.mean(axis=(1, 2))  # (T,)


def gates_over_time_std(gate_arrays):
    """Return std of gate value per timestep."""
    all_gates = np.concatenate(gate_arrays, axis=1)
    return all_gates.std(axis=(1, 2))


# ============================================================================
# Plotting
# ============================================================================

def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def save_fig(fig, name, out_dir):
    """Save figure in both PDF and PNG."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    print(f"  Saved {name}.pdf / .png")


def plot_histogram(gate_data, num_layers, out_dir):
    """Plot 1: Histogram of gate values per decoder layer."""
    plt = setup_matplotlib()
    import seaborn as sns

    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(3.5 * ((num_layers + 1) // 2), 6))
    axes = axes.flatten()

    colors = sns.color_palette("viridis", num_layers)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        vals = gates_flat(gate_data[layer_idx])
        ax.hist(vals, bins=100, density=True, alpha=0.8, color=colors[layer_idx],
                edgecolor="none")
        mean_val = vals.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.3f}")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", frameon=False)

    # Hide extra axes
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Adaptive Fusion Gate Distribution by Decoder Layer\n"
                 "(1 = action, 0 = identity)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "gate_histogram", out_dir)
    plt.close(fig)


def plot_per_joint_heatmap(gate_data, num_layers, out_dir):
    """Plot 2: Heatmap of mean gate value per joint, per layer."""
    plt = setup_matplotlib()
    import seaborn as sns

    # Build matrix: (num_layers, 25)
    mat = np.zeros((num_layers, 25))
    for layer_idx in range(num_layers):
        per_joint = gates_to_per_joint(gate_data[layer_idx], d_model=None, num_joints=25)
        # per_joint: (N, T, V) -> average over samples and time
        mat[layer_idx, :] = per_joint.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(12, 0.9 * num_layers + 1.5))
    im = sns.heatmap(
        mat,
        xticklabels=NTU_JOINT_NAMES,
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
        cmap="RdYlBu_r",
        vmin=0.0, vmax=1.0,
        annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Gate value  (1=action, 0=identity)"},
        ax=ax,
    )
    ax.set_xlabel("Joint")
    ax.set_ylabel("Decoder Layer")
    ax.set_title("Mean Fusion Gate Value per Joint and Layer")
    plt.xticks(rotation=55, ha="right")
    fig.tight_layout()
    save_fig(fig, "gate_per_joint", out_dir)
    plt.close(fig)


def plot_over_time(gate_data, num_layers, out_dir):
    """Plot 3: Gate value over the 64 (T-1) frames, per layer."""
    plt = setup_matplotlib()
    import seaborn as sns

    colors = sns.color_palette("viridis", num_layers)

    fig, ax = plt.subplots(figsize=(8, 4))
    for layer_idx in range(num_layers):
        mean_t = gates_over_time(gate_data[layer_idx])
        std_t  = gates_over_time_std(gate_data[layer_idx])
        frames = np.arange(len(mean_t))
        ax.plot(frames, mean_t, label=f"Layer {layer_idx}", color=colors[layer_idx], linewidth=1.5)
        ax.fill_between(frames, mean_t - std_t, mean_t + std_t,
                        alpha=0.15, color=colors[layer_idx])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean gate value")
    ax.set_title("Adaptive Fusion Gate Over Time\n(1 = action, 0 = identity)")
    ax.set_xlim(0, len(frames) - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "gate_over_time", out_dir)
    plt.close(fig)


def plot_by_action(gate_data, action_labels, num_layers, out_dir, top_k=15):
    """Plot 4: Mean gate value by action class (top-K most frequent actions)."""
    plt = setup_matplotlib()
    import seaborn as sns

    # Use a single representative layer (last layer, which is closest to output)
    last_layer = num_layers - 1

    all_gates = np.concatenate(gate_data[last_layer], axis=1)  # (T, N, D)
    per_sample_mean = all_gates.mean(axis=(0, 2))  # (N,)

    # Also compute per-layer averages for all layers
    per_layer_per_action = {}
    for layer_idx in range(num_layers):
        g = np.concatenate(gate_data[layer_idx], axis=1)
        per_layer_per_action[layer_idx] = g.mean(axis=(0, 2))  # (N,)

    # Get per-action stats
    unique_actions = np.unique(action_labels)
    action_means = {}
    action_stds  = {}
    action_counts = {}
    for a in unique_actions:
        mask = action_labels == a
        action_means[a] = per_sample_mean[mask].mean()
        action_stds[a]  = per_sample_mean[mask].std()
        action_counts[a] = mask.sum()

    # Select top-K by frequency
    sorted_actions = sorted(action_counts.keys(), key=lambda a: action_counts[a], reverse=True)
    top_actions = sorted_actions[:top_k]
    top_actions_sorted = sorted(top_actions, key=lambda a: action_means[a], reverse=True)

    labels = []
    means  = []
    stds   = []
    for a in top_actions_sorted:
        a_int = int(a)
        name = NTU_ACTION_NAMES.get(a_int, f"Action {a_int}")
        labels.append(f"{name} (n={action_counts[a]})")
        means.append(action_means[a])
        stds.append(action_stds[a])

    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 0.45 * top_k + 1.5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, align="center", height=0.6,
                   color=sns.color_palette("coolwarm_r", len(labels)),
                   edgecolor="none", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Mean gate value (Layer {last_layer})")
    ax.set_title(f"Fusion Gate by Action Class (Top {top_k})\n(1 = action-driven, 0 = identity-driven)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "gate_by_action", out_dir)
    plt.close(fig)

    # --- Supplementary: multi-layer action heatmap ---
    # Rows = actions, Columns = layers
    mat = np.zeros((len(top_actions_sorted), num_layers))
    for i, a in enumerate(top_actions_sorted):
        mask = action_labels == a
        for layer_idx in range(num_layers):
            mat[i, layer_idx] = per_layer_per_action[layer_idx][mask].mean()

    row_labels = []
    for a in top_actions_sorted:
        a_int = int(a)
        row_labels.append(NTU_ACTION_NAMES.get(a_int, f"Action {a_int}"))

    fig2, ax2 = plt.subplots(figsize=(max(4, 1.3 * num_layers), 0.45 * top_k + 1.5))
    sns.heatmap(
        mat,
        xticklabels=[f"L{i}" for i in range(num_layers)],
        yticklabels=row_labels,
        cmap="RdYlBu_r",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Gate value"},
        ax=ax2,
    )
    ax2.set_xlabel("Decoder Layer")
    ax2.set_ylabel("Action Class")
    ax2.set_title(f"Fusion Gate by Action & Layer (Top {top_k})")
    fig2.tight_layout()
    save_fig(fig2, "gate_by_action_layer", out_dir)
    plt.close(fig2)


def print_summary(gate_data, num_layers, action_labels):
    """Print a textual summary of gate statistics."""
    print("\n" + "=" * 60)
    print("Gate Analysis Summary")
    print("=" * 60)
    for layer_idx in range(num_layers):
        vals = gates_flat(gate_data[layer_idx])
        print(f"  Layer {layer_idx}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={np.median(vals):.4f}  "
              f"<0.3: {(vals < 0.3).mean()*100:.1f}%  "
              f">0.7: {(vals > 0.7).mean()*100:.1f}%")
    print()

    # Overall
    all_vals = np.concatenate([gates_flat(gate_data[l]) for l in range(num_layers)])
    print(f"  Overall: mean={all_vals.mean():.4f}  std={all_vals.std():.4f}")
    if all_vals.mean() > 0.6:
        print("  => Decoder is biased toward ACTION stream.")
    elif all_vals.mean() < 0.4:
        print("  => Decoder is biased toward IDENTITY stream.")
    else:
        print("  => Decoder is roughly balanced between action and identity.")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive fusion gate values")
    parser.add_argument("--checkpoint", type=str,
                        default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth",
                        help="Path to Stage 3 checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="data/ntu/ntu_cv_paired_10k.pt",
                        help="Path to paired data .pt file")
    parser.add_argument("--dataset", type=str, default="ntu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches for quick testing (None = all)")
    parser.add_argument("--output_dir", type=str, default="paper/fig",
                        help="Directory for output figures")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("Adaptive Fusion Gate Analysis")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Device     : {args.device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Output dir : {args.output_dir}")
    print()

    # 1. Load model
    model = load_model(args.checkpoint, args.device, args.dataset)
    num_layers = len(model.decoder.layers)

    # 2. Load data
    val_loader, val_dataset = load_val_data(args.data_path, args.batch_size, args.num_workers)

    # 3. Collect gate values
    gate_data, action_labels, actor_labels = collect_gates(
        model, val_loader, args.device, max_batches=args.max_batches
    )
    print(f"\nCollected gates for {len(action_labels)} samples across {num_layers} layers.")

    # 4. Print summary
    print_summary(gate_data, num_layers, action_labels)

    # 5. Generate plots
    print("\nGenerating plots ...")
    plot_histogram(gate_data, num_layers, args.output_dir)
    plot_per_joint_heatmap(gate_data, num_layers, args.output_dir)
    plot_over_time(gate_data, num_layers, args.output_dir)
    plot_by_action(gate_data, action_labels, num_layers, args.output_dir)

    print("\nDone. All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
