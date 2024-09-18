#!/usr/bin/env python3
"""
Generate t-SNE visualizations of action and identity embeddings from a trained
DisentangledTMR model.

Produces 4 publication-quality figures:
  1. Action embeddings colored by action class
  2. Action embeddings colored by identity (should show NO clustering = good privacy)
  3. Identity embeddings colored by identity
  4. Identity embeddings colored by action class (should show NO clustering = good disentanglement)
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.data.datasets import parse_file_name

# ---------------------------------------------------------------------------
# Matplotlib setup -- use Agg backend (no display) and publication defaults
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PUBLICATION_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
}
plt.rcParams.update(PUBLICATION_RC)

# ---------------------------------------------------------------------------
# Model loading (mirrors generate_retargeted_dataset.py / evaluate_disentangled_tmr.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device, dataset_name):
    """Load trained DisentangledTMR and return in eval mode."""
    print(f"Loading model from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None and isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    # Dataset -> num_class
    ds_num_class = {"ntu": 49, "ntu_smoke": 49, "ntu_small": 49, "ntu120": 94, "etri": 55}
    num_class = ds_num_class.get(dataset_name, 49)

    d_action  = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model   = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320

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
        tokenizer_dim=getattr(ckpt_args, "tokenizer_dim", 256) if ckpt_args else 256,
        token_fusion=getattr(ckpt_args, "token_fusion", "add") if ckpt_args else "add",
        use_codebook=getattr(ckpt_args, "use_codebook", False) if ckpt_args else False,
        codebook_size=getattr(ckpt_args, "codebook_size", 256) if ckpt_args else 256,
        codebook_dim=getattr(ckpt_args, "codebook_dim", 256) if ckpt_args else 256,
        codebook_distance=getattr(ckpt_args, "codebook_distance", "euclidean") if ckpt_args else "euclidean",
        vq_commitment_weight=getattr(ckpt_args, "vq_commitment_weight", 0.25) if ckpt_args else 0.25,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model, dataloader, device, max_samples=5000):
    """
    Run the model on the dataloader and return pooled action / identity
    embeddings together with their action-class and identity labels.

    Returns:
        action_embs  : np.ndarray  (N, D_action)
        identity_embs: np.ndarray  (N, D_identity)
        action_labels: np.ndarray  (N,)  -- 1-indexed action class
        identity_labels: np.ndarray (N,) -- 1-indexed person ID
    """
    all_action = []
    all_identity = []
    all_action_labels = []
    all_identity_labels = []

    n_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            x1, x2, y1, y2, actors, actions = batch

            # x1 is (B, C, T, V) -- add person dim
            x1 = x1.to(device).unsqueeze(-1)   # (B, C, T, V, 1)

            # Encode action from source motion
            act_feat = model.encode_action(x1)  # (T, B, D_action)

            # Pool over time -> (B, D_action)
            act_feat_pooled = act_feat.mean(dim=0)

            # Encode identity from source skeleton (same sample)
            id_feat = model.encode_identity(x1)  # (B, D_identity)

            all_action.append(act_feat_pooled.cpu().numpy())
            all_identity.append(id_feat.cpu().numpy())

            # Labels: actors[:, 0] = P1, actions[:, 0] = A1 (1-indexed)
            all_action_labels.append(actions[:, 0].numpy())
            all_identity_labels.append(actors[:, 0].numpy())

            n_collected += x1.size(0)
            if n_collected >= max_samples:
                break

    action_embs   = np.concatenate(all_action, axis=0)[:max_samples]
    identity_embs = np.concatenate(all_identity, axis=0)[:max_samples]
    action_labels   = np.concatenate(all_action_labels, axis=0)[:max_samples]
    identity_labels = np.concatenate(all_identity_labels, axis=0)[:max_samples]

    print(f"Collected {len(action_embs)} samples.")
    print(f"  Action embedding shape:   {action_embs.shape}")
    print(f"  Identity embedding shape: {identity_embs.shape}")
    print(f"  Unique actions:    {len(np.unique(action_labels))}")
    print(f"  Unique identities: {len(np.unique(identity_labels))}")

    return action_embs, identity_embs, action_labels, identity_labels


# ---------------------------------------------------------------------------
# t-SNE and plotting
# ---------------------------------------------------------------------------

def preprocess_embeddings(embeddings, n_components=50):
    """PCA + standardize embeddings that may have near-zero variance.

    The identity encoder produces embeddings with extremely low total variance
    (O(1e-8)) concentrated in a few dimensions.  Raw t-SNE cannot resolve this.
    PCA extracts informative directions; standardizing makes them visible.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    total_var = np.var(embeddings, axis=0).sum()
    if total_var > 1e-3:
        # Action embeddings have healthy variance — skip preprocessing
        return embeddings

    print(f"  Low-variance embeddings detected (total var={total_var:.2e}). "
          f"Applying PCA({n_components}) + standardization.")
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA: {n_components} components explain {explained[-1]*100:.1f}% variance.")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_pca)
    return emb_scaled


def run_tsne(embeddings, perplexity=30, random_state=42):
    """Run t-SNE on embeddings (with automatic preprocessing for low-variance data)."""
    from sklearn.manifold import TSNE

    embeddings = preprocess_embeddings(embeddings)

    print(f"Running t-SNE on {embeddings.shape} with perplexity={perplexity} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    print("  t-SNE complete.")
    return coords


def _build_colormap(labels):
    """Return a color array and legend handles for a set of integer labels."""
    unique = np.sort(np.unique(labels))
    n = len(unique)

    # Choose a qualitative colormap; fall back to a continuous one for many classes
    if n <= 10:
        cmap = plt.cm.get_cmap("tab10", 10)
    elif n <= 20:
        cmap = plt.cm.get_cmap("tab20", 20)
    else:
        cmap = plt.cm.get_cmap("gist_ncar", n)

    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    colors = np.array([cmap(label_to_idx[l] / max(n - 1, 1)) for l in labels])

    # Build legend handles (show a subset if too many classes)
    show_labels = unique if n <= 20 else unique[:: max(1, n // 15)]
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(label_to_idx[lbl] / max(n - 1, 1)),
               markersize=5, label=str(int(lbl)))
        for lbl in show_labels
    ]
    return colors, handles, n


def plot_tsne(coords, labels, title, xlabel, ylabel, save_prefix, label_name="Class"):
    """Create and save a single t-SNE scatter plot in both PDF and PNG."""
    colors, handles, n_classes = _build_colormap(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.6,
               edgecolors="none", rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Legend
    if n_classes <= 40:
        ncol = max(1, n_classes // 10)
        ax.legend(handles=handles, title=label_name, loc="upper right",
                  ncol=ncol, framealpha=0.7, markerscale=1.5,
                  handletextpad=0.3, columnspacing=0.8)
    else:
        # Too many classes -- add a text annotation instead
        ax.annotate(f"{n_classes} {label_name.lower()}s",
                    xy=(0.98, 0.98), xycoords="axes fraction",
                    ha="right", va="top",
                    fontsize=10, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])

    for ext in ("pdf", "png"):
        path = f"{save_prefix}.{ext}"
        fig.savefig(path, format=ext)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of DisentangledTMR embeddings")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 3 checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to paired .pt data file")
    parser.add_argument("--dataset", type=str, default="ntu",
                        choices=["ntu", "ntu120", "etri", "ntu_smoke", "ntu_small"])
    parser.add_argument("--output_dir", type=str, default="paper/fig",
                        help="Directory to save figures")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Max samples to embed (more = slower but denser plot)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data --------------------------------------------------------
    print(f"Loading paired data from {args.data_path} ...")
    data = torch.load(args.data_path, weights_only=False)
    test_dataset = data["test"]
    print(f"  Test dataset size: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ---- Load model -------------------------------------------------------
    model = load_model(args.checkpoint, device, args.dataset)

    # ---- Extract embeddings -----------------------------------------------
    action_embs, identity_embs, action_labels, identity_labels = extract_embeddings(
        model, test_loader, device, max_samples=args.max_samples
    )

    # ---- t-SNE ------------------------------------------------------------
    action_tsne   = run_tsne(action_embs, perplexity=args.perplexity, random_state=args.seed)
    identity_tsne = run_tsne(identity_embs, perplexity=args.perplexity, random_state=args.seed)

    # ---- Create output directory ------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Plot 1: Action embeddings x Action class -------------------------
    plot_tsne(
        action_tsne, action_labels,
        title="Action Embeddings Colored by Action Class",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        save_prefix=os.path.join(args.output_dir, "tsne_action_by_action"),
        label_name="Action",
    )

    # ---- Plot 2: Action embeddings x Identity (should show NO clustering) -
    plot_tsne(
        action_tsne, identity_labels,
        title="Action Embeddings Colored by Identity (Privacy Check)",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        save_prefix=os.path.join(args.output_dir, "tsne_action_by_identity"),
        label_name="Identity",
    )

    # ---- Plot 3: Identity embeddings x Identity ---------------------------
    plot_tsne(
        identity_tsne, identity_labels,
        title="Identity Embeddings Colored by Identity",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        save_prefix=os.path.join(args.output_dir, "tsne_identity_by_identity"),
        label_name="Identity",
    )

    # ---- Plot 4: Identity embeddings x Action class (should show NO clustering)
    plot_tsne(
        identity_tsne, action_labels,
        title="Identity Embeddings Colored by Action Class (Disentanglement Check)",
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        save_prefix=os.path.join(args.output_dir, "tsne_identity_by_action"),
        label_name="Action",
    )

    # ---- Save raw data for potential reuse --------------------------------
    npz_path = os.path.join(args.output_dir, "tsne_embeddings.npz")
    np.savez_compressed(
        npz_path,
        action_tsne=action_tsne,
        identity_tsne=identity_tsne,
        action_embs=action_embs,
        identity_embs=identity_embs,
        action_labels=action_labels,
        identity_labels=identity_labels,
    )
    print(f"\nRaw embeddings + t-SNE coords saved to {npz_path}")
    print("All done.")


if __name__ == "__main__":
    main()
