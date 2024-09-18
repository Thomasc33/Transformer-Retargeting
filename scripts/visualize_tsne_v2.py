#!/usr/bin/env python3
"""
Generate publication-quality t-SNE visualizations of action and identity embeddings
from a trained DisentangledTMR model.

Improved version of visualize_tsne.py with:
  - Curated subset of 12 action classes for clarity
  - Stratified sampling for balanced representation
  - Higher t-SNE quality (perplexity=50, n_iter=3000)
  - Clean formatting for paper figures (no titles, no axis labels)
  - Distinct qualitative colormaps (tab10/tab20)
  - Configurable subsets via CLI

Produces 4 figures:
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
from collections import Counter

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
# NTU RGB+D 60 action names (1-indexed, matching dataset labels)
# ---------------------------------------------------------------------------
NTU_ACTION_NAMES = {
    1: "Drink water", 2: "Eat meal", 3: "Brush teeth", 4: "Brush hair",
    5: "Drop", 6: "Pick up", 7: "Throw", 8: "Sit down", 9: "Stand up",
    10: "Clapping", 11: "Reading", 12: "Writing", 13: "Tear up paper",
    14: "Wear jacket", 15: "Take off jacket", 16: "Wear shoe",
    17: "Take off shoe", 18: "Wear glasses", 19: "Take off glasses",
    20: "Put on hat", 21: "Take off hat", 22: "Cheer up", 23: "Hand waving",
    24: "Kicking something", 25: "Reach into pocket", 26: "Hopping",
    27: "Jump up", 28: "Phone call", 29: "Play with phone",
    30: "Type on keyboard", 31: "Point to something", 32: "Take selfie",
    33: "Check time", 34: "Rub two hands", 35: "Nod head/bow",
    36: "Shake head", 37: "Wipe face", 38: "Salute", 39: "Put palms together",
    40: "Cross hands in front", 41: "Sneeze/cough", 42: "Staggering",
    43: "Falling down", 44: "Headache", 45: "Chest pain",
    46: "Back pain", 47: "Neck pain", 48: "Nausea/vomiting",
    49: "Fan self",
}

# Curated action subset: visually distinct actions spanning different motion types.
# These are 1-indexed NTU action IDs.
# 6 classes chosen for maximum motion diversity without outlier "Falling down" (43)
# which pulls UMAP/t-SNE layout to extreme, compressing everything else.
# Upper body: drink water(1), clapping(10)
# Whole body: sit down(8), jump up(27), kicking(24)
# Arm gesture: hand waving(23)
DEFAULT_ACTION_SUBSET = [1, 8, 10, 23, 24, 27]


# ---------------------------------------------------------------------------
# Model loading (IDENTICAL to visualize_tsne.py)
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
# Embedding extraction - multiple strategies
# ---------------------------------------------------------------------------

def load_ar_classifier(checkpoint_path, d_action, num_classes, device):
    """Load the AR classifier head from checkpoint if available."""
    from src.model.simple_classifiers import ActionClassifier
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'ar_classifier_state_dict' not in checkpoint:
        print("  WARNING: No AR classifier found in checkpoint. Using raw encoder features.")
        return None
    ar_classifier = ActionClassifier(d_action, num_classes).to(device)
    ar_classifier.load_state_dict(checkpoint['ar_classifier_state_dict'])
    ar_classifier.eval()
    print(f"  AR classifier loaded (d_action={d_action}, num_classes={num_classes}).")
    return ar_classifier


def extract_embeddings(model, dataloader, device, max_samples=5000,
                       pooling='mean', ar_classifier=None, feature_source='encoder'):
    """
    Run the model on the dataloader and return pooled action / identity
    embeddings together with their action-class and identity labels.

    Args:
        model: DisentangledTMR model
        dataloader: test data loader
        device: torch device
        max_samples: max samples to extract
        pooling: temporal pooling method for action features:
            'mean' - average over time (default, matches training)
            'max' - max-pool over time (may capture peak discriminative frames)
            'meanmax' - concatenation of mean and max pooling
        ar_classifier: optional trained ActionClassifier; if provided and
            feature_source='classifier', extract intermediate features from
            the classifier's hidden layers rather than raw encoder output
        feature_source: 'encoder' (raw action encoder output) or
            'classifier' (intermediate features from AR classifier head)

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

    # If using classifier features, define a forward hook to capture hidden layer
    classifier_hidden = {}
    hook_handle = None
    if feature_source == 'classifier' and ar_classifier is not None:
        # Hook into the penultimate layer (after 768->512->ReLU->Dropout->256->ReLU)
        # The classifier is Sequential: Linear(768,512), ReLU, Dropout, Linear(512,256), ReLU, Dropout, Linear(256,num_classes)
        # We want the output after the second ReLU (index 4), which gives 256-d features
        def hook_fn(module, input, output):
            classifier_hidden['feat'] = output
        hook_handle = ar_classifier.classifier[4].register_forward_hook(hook_fn)
        print(f"  Extracting classifier hidden features (256-d penultimate layer).")

    n_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            x1, x2, y1, y2, actors, actions = batch

            # x1 is (B, C, T, V) -- add person dim
            x1 = x1.to(device).unsqueeze(-1)   # (B, C, T, V, 1)

            # Encode action from source motion
            act_feat = model.encode_action(x1)  # (T, B, D_action)

            # Temporal pooling
            if pooling == 'mean':
                act_feat_pooled = act_feat.mean(dim=0)    # (B, D_action)
            elif pooling == 'max':
                act_feat_pooled = act_feat.max(dim=0)[0]  # (B, D_action)
            elif pooling == 'meanmax':
                act_mean = act_feat.mean(dim=0)
                act_max = act_feat.max(dim=0)[0]
                act_feat_pooled = torch.cat([act_mean, act_max], dim=-1)  # (B, 2*D_action)
            else:
                act_feat_pooled = act_feat.mean(dim=0)

            # If using classifier features, pass through classifier to trigger hook
            if feature_source == 'classifier' and ar_classifier is not None:
                _ = ar_classifier(act_feat)  # triggers hook; handles temporal avg internally
                act_feat_pooled = classifier_hidden['feat']  # (B, 256)

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

    if hook_handle is not None:
        hook_handle.remove()

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
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(embeddings_list, action_labels, identity_labels,
                      action_subset=None, identity_subset=None,
                      samples_per_class=None, rng=None):
    """
    Subsample data for cleaner visualization.

    For action plots: keep only samples whose action label is in action_subset,
    then take equal numbers per action class.

    For identity plots: keep only samples whose identity label is in identity_subset,
    then take equal numbers per identity.

    Args:
        embeddings_list: list of np.ndarray to filter in parallel (e.g. [action_embs, identity_embs])
        action_labels: (N,) action labels
        identity_labels: (N,) identity labels
        action_subset: list of action IDs to keep, or None for all
        identity_subset: list of identity IDs to keep, or None for all
        samples_per_class: max samples per class (auto-computed if None)
        rng: numpy random generator

    Returns:
        filtered_embeddings: list of filtered arrays
        filtered_action_labels: filtered action labels
        filtered_identity_labels: filtered identity labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = np.arange(len(action_labels))

    # Filter by action subset
    if action_subset is not None:
        action_mask = np.isin(action_labels, action_subset)
        indices = indices[action_mask]

    # Filter by identity subset
    if identity_subset is not None:
        identity_mask = np.isin(identity_labels[indices], identity_subset)
        indices = indices[identity_mask]

    # Determine which label to stratify on
    if action_subset is not None:
        stratify_labels = action_labels[indices]
    elif identity_subset is not None:
        stratify_labels = identity_labels[indices]
    else:
        # No subset filtering; return as-is
        return (
            [emb[indices] for emb in embeddings_list],
            action_labels[indices],
            identity_labels[indices],
        )

    # Stratified sampling: equal samples per class
    unique_labels = np.unique(stratify_labels)
    if samples_per_class is None:
        # Use minimum class count so all classes are equally represented
        counts = Counter(stratify_labels)
        samples_per_class = min(counts.values())

    selected = []
    for lbl in unique_labels:
        lbl_indices = indices[stratify_labels == lbl]
        if len(lbl_indices) > samples_per_class:
            lbl_indices = rng.choice(lbl_indices, size=samples_per_class, replace=False)
        selected.append(lbl_indices)

    selected = np.concatenate(selected)
    rng.shuffle(selected)  # Shuffle so plot layers are mixed

    return (
        [emb[selected] for emb in embeddings_list],
        action_labels[selected],
        identity_labels[selected],
    )


def select_action_subset(action_labels, desired_subset=None, n_classes=12):
    """
    Select a subset of action classes for visualization.

    If desired_subset IDs exist in the data, use them. Otherwise fall back
    to the n_classes most frequent classes.
    """
    unique_actions = np.unique(action_labels)

    if desired_subset is not None:
        # Keep only those that actually exist in the data
        valid = [a for a in desired_subset if a in unique_actions]
        if len(valid) >= n_classes:
            return sorted(valid[:n_classes])
        elif len(valid) >= n_classes // 2:
            print(f"  Warning: only {len(valid)}/{len(desired_subset)} desired actions found. "
                  f"Using those {len(valid)}.")
            return sorted(valid)

    # Fallback: most frequent classes
    print(f"  Falling back to {n_classes} most frequent action classes.")
    counts = Counter(action_labels)
    most_common = counts.most_common(n_classes)
    return sorted([lbl for lbl, _ in most_common])


def select_identity_subset(identity_labels, n_identities=15):
    """Select top-N identities by sample count."""
    counts = Counter(identity_labels)
    most_common = counts.most_common(n_identities)
    return sorted([lbl for lbl, _ in most_common])


# ---------------------------------------------------------------------------
# Dimensionality reduction: t-SNE, UMAP, and preprocessing
# ---------------------------------------------------------------------------

def preprocess_embeddings(embeddings, n_components=50, force_pca=False):
    """PCA + standardize embeddings that may have near-zero variance.

    The identity encoder produces embeddings with extremely low total variance
    (O(1e-8)) concentrated in a few dimensions.  Raw t-SNE cannot resolve this.
    PCA extracts informative directions; standardizing makes them visible.

    For high-dimensional action embeddings (768-d), PCA pre-reduction to 50-d
    can also help t-SNE/UMAP by removing noise dimensions. Set force_pca=True.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    total_var = np.var(embeddings, axis=0).sum()
    need_pca = (total_var <= 1e-3) or force_pca

    if not need_pca:
        return embeddings

    if total_var <= 1e-3:
        print(f"  Low-variance embeddings detected (total var={total_var:.2e}).")
    else:
        print(f"  Force PCA pre-reduction ({embeddings.shape[1]}-d -> {n_components}-d).")

    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA: {n_components} components explain {explained[-1]*100:.1f}% variance.")

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_pca)
    return emb_scaled


def run_tsne(embeddings, perplexity=50, n_iter=3000, random_state=42, pca_first=False):
    """Run t-SNE on embeddings (with automatic preprocessing for low-variance data)."""
    from sklearn.manifold import TSNE

    embeddings = preprocess_embeddings(embeddings, force_pca=pca_first)

    print(f"Running t-SNE on {embeddings.shape} with perplexity={perplexity}, n_iter={n_iter} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    print("  t-SNE complete.")
    return coords


def run_umap(embeddings, n_neighbors=30, min_dist=0.3, spread=1.0,
             metric='cosine', random_state=42, pca_first=False):
    """Run UMAP on embeddings. Requires umap-learn package.

    UMAP often produces better global structure than t-SNE for many classes,
    and is much faster for large datasets.

    Args:
        spread: controls how spread out the clusters are. Higher values push
            clusters further apart. Must be >= min_dist. Default 1.0.
    """
    try:
        import umap
    except ImportError:
        print("  WARNING: umap-learn not installed. Install with: pip install umap-learn")
        print("  Falling back to t-SNE.")
        return run_tsne(embeddings, perplexity=min(50, len(embeddings) // 5),
                        random_state=random_state, pca_first=pca_first)

    embeddings = preprocess_embeddings(embeddings, force_pca=pca_first)

    print(f"Running UMAP on {embeddings.shape} with n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}, spread={spread}, metric={metric} ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    print("  UMAP complete.")
    return coords


def run_dr(embeddings, method='tsne', perplexity=50, n_iter=3000,
           n_neighbors=30, min_dist=0.3, spread=1.0, metric='cosine',
           random_state=42, pca_first=False):
    """Dispatch to t-SNE or UMAP based on method string."""
    if method == 'umap':
        return run_umap(embeddings, n_neighbors=n_neighbors, min_dist=min_dist,
                        spread=spread, metric=metric, random_state=random_state,
                        pca_first=pca_first)
    else:
        return run_tsne(embeddings, perplexity=perplexity, n_iter=n_iter,
                        random_state=random_state, pca_first=pca_first)


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def _build_qualitative_colors(labels, cmap_name=None):
    """
    Build distinct colors for a set of integer labels using qualitative colormaps.

    Uses tab10 for <=10 classes, tab20 for <=20, and a combination of
    tab10 + tab20 for up to 30 classes.

    Returns:
        colors: np.ndarray (N, 4) -- RGBA colors per sample
        label_to_color: dict -- {label: RGBA tuple}
    """
    unique = np.sort(np.unique(labels))
    n = len(unique)

    if cmap_name is not None:
        cmap = plt.cm.get_cmap(cmap_name, max(n, 10))
        label_to_color = {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(unique)}
    elif n <= 10:
        cmap = plt.cm.get_cmap("tab10", 10)
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique)}
    elif n <= 20:
        cmap = plt.cm.get_cmap("tab20", 20)
        label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique)}
    else:
        # Combine tab10 + tab20 for maximum distinction
        tab10 = plt.cm.get_cmap("tab10", 10)
        tab20 = plt.cm.get_cmap("tab20", 20)
        combined_colors = []
        for i in range(10):
            combined_colors.append(tab10(i))
        for i in range(20):
            combined_colors.append(tab20(i))
        # Deduplicate and take first n
        seen = set()
        deduped = []
        for c in combined_colors:
            key = (round(c[0], 3), round(c[1], 3), round(c[2], 3))
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        label_to_color = {lbl: deduped[i % len(deduped)] for i, lbl in enumerate(unique)}

    colors = np.array([label_to_color[l] for l in labels])
    return colors, label_to_color


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tsne_v2(coords, labels, save_prefix, label_names=None,
                 show_title=False, title="", point_size=15, alpha=0.7,
                 label_type="Class", cmap_name=None, figsize=(5, 5)):
    """
    Create and save a single publication-quality t-SNE scatter plot.

    Args:
        coords: (N, 2) t-SNE coordinates
        labels: (N,) integer labels
        save_prefix: path prefix for output files (no extension)
        label_names: dict mapping label -> human-readable name, or None
        show_title: if True, add title to the plot
        title: title string (only used if show_title=True)
        point_size: scatter point size
        alpha: scatter point alpha
        label_type: "Action" or "Identity" (for legend title)
        cmap_name: override colormap name, or None for auto
        figsize: figure size tuple
    """
    colors, label_to_color = _build_qualitative_colors(labels, cmap_name=cmap_name)
    unique_labels = np.sort(np.unique(labels))
    n_classes = len(unique_labels)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=point_size, alpha=alpha,
               edgecolors="none", rasterized=True)

    if show_title and title:
        ax.set_title(title)

    # No axis labels (t-SNE dimensions are meaningless)
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend -- compact, inside the plot for paper figures
    def _get_label_str(lbl, label_names, label_type):
        """Get human-readable label, handling int/float keys."""
        if label_names:
            # Try both int and float keys
            for key in [lbl, int(lbl)]:
                if key in label_names:
                    return label_names[key]
        if label_type == "Identity":
            return f"P{int(lbl)}"
        return str(int(lbl))

    if n_classes <= 10:
        handles = []
        for lbl in unique_labels:
            lbl_str = _get_label_str(lbl, label_names, label_type)
            handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=label_to_color[lbl],
                       markersize=5, label=lbl_str)
            )
        ncol = 1
        legend = ax.legend(
            handles=handles, title=label_type,
            loc="upper right",
            ncol=ncol, framealpha=0.85, markerscale=1.0,
            handletextpad=0.3, columnspacing=0.6,
            borderaxespad=0.3, fontsize=7,
        )
        legend.get_title().set_fontsize(8)
    elif n_classes <= 20:
        handles = []
        for lbl in unique_labels:
            lbl_str = _get_label_str(lbl, label_names, label_type)
            handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=label_to_color[lbl],
                       markersize=5, label=lbl_str)
            )
        ncol = 2
        legend = ax.legend(
            handles=handles, title=label_type,
            loc="upper right",
            ncol=ncol, framealpha=0.85, markerscale=1.0,
            handletextpad=0.3, columnspacing=0.6,
            borderaxespad=0.3, fontsize=7,
        )
        legend.get_title().set_fontsize(8)
    else:
        # Too many classes -- no legend, just text annotation
        # Pluralize correctly
        type_plural = "identities" if label_type.lower() == "identity" else f"{label_type.lower()}s"
        ax.annotate(f"{n_classes} {type_plural}",
                    xy=(0.98, 0.98), xycoords="axes fraction",
                    ha="right", va="top",
                    fontsize=9, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    for ext in ("pdf", "png"):
        path = f"{save_prefix}.{ext}"
        fig.savefig(path, format=ext)
        print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality t-SNE/UMAP visualization of DisentangledTMR embeddings (v2)"
    )
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Dimensionality reduction method
    parser.add_argument("--dr_method", type=str, default="tsne",
                        choices=["tsne", "umap"],
                        help="Dimensionality reduction method (default: tsne)")

    # t-SNE parameters
    parser.add_argument("--perplexity", type=float, default=50,
                        help="t-SNE perplexity for action plots (default: 50)")
    parser.add_argument("--perplexity_identity", type=float, default=None,
                        help="t-SNE perplexity for identity plots (default: perplexity - 10)")
    parser.add_argument("--n_iter", type=int, default=3000,
                        help="t-SNE iterations (default: 3000)")

    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=30,
                        help="UMAP n_neighbors (default: 30)")
    parser.add_argument("--min_dist", type=float, default=0.3,
                        help="UMAP min_dist (default: 0.3)")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="UMAP spread: controls inter-cluster distance. "
                             "Higher values push clusters further apart. Must be >= min_dist. (default: 1.0)")
    parser.add_argument("--umap_metric", type=str, default="cosine",
                        help="UMAP distance metric (default: cosine)")

    # Feature extraction options
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "meanmax"],
                        help="Temporal pooling for action features (default: mean)")
    parser.add_argument("--feature_source", type=str, default="encoder",
                        choices=["encoder", "classifier"],
                        help="Use raw encoder output or AR classifier hidden features (default: encoder)")
    parser.add_argument("--pca_first", action="store_true", default=False,
                        help="Apply PCA pre-reduction even for high-variance embeddings (recommended for 768-d)")

    # Subset control
    parser.add_argument("--action_subset", type=int, nargs="+", default=None,
                        help="Specific action IDs to include (1-indexed). "
                             "Default: curated list of 12 distinct actions.")
    parser.add_argument("--n_action_classes", type=int, default=6,
                        help="Number of action classes to show (default: 6)")
    parser.add_argument("--identity_subset", type=int, nargs="+", default=None,
                        help="Specific identity IDs to include (1-indexed). "
                             "Default: top 6 identities by sample count.")
    parser.add_argument("--n_identities", type=int, default=6,
                        help="Number of identities to show (default: 6)")

    # Display control
    parser.add_argument("--no_titles", action="store_true", default=False,
                        help="Omit plot titles (recommended for paper figures)")
    parser.add_argument("--all_classes", action="store_true", default=False,
                        help="Show all classes (no subsetting). Backward-compatible mode.")

    args = parser.parse_args()

    # Derived defaults
    if args.perplexity_identity is None:
        args.perplexity_identity = max(5, args.perplexity - 10)

    # Seed
    rng = np.random.default_rng(args.seed)
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

    # ---- Optionally load AR classifier ------------------------------------
    ar_classifier = None
    if args.feature_source == 'classifier':
        ds_num_class = {"ntu": 49, "ntu_smoke": 49, "ntu_small": 49, "ntu120": 94, "etri": 55}
        num_class = ds_num_class.get(args.dataset, 49)
        ar_classifier = load_ar_classifier(args.checkpoint, 768, num_class, device)
        if ar_classifier is None:
            print("  Falling back to encoder features.")
            args.feature_source = 'encoder'

    # ---- Extract embeddings -----------------------------------------------
    print(f"\nExtraction config: pooling={args.pooling}, feature_source={args.feature_source}")
    action_embs, identity_embs, action_labels, identity_labels = extract_embeddings(
        model, test_loader, device, max_samples=args.max_samples,
        pooling=args.pooling, ar_classifier=ar_classifier,
        feature_source=args.feature_source,
    )

    # ---- Create output directory ------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Determine subsets ------------------------------------------------
    if args.all_classes:
        # Backward-compatible: use all classes, original behavior
        action_subset_ids = None
        identity_subset_ids = None
        print("\n--all_classes mode: showing all classes (no subsetting).")
    else:
        # Select action subset
        desired_actions = args.action_subset if args.action_subset else DEFAULT_ACTION_SUBSET
        action_subset_ids = select_action_subset(
            action_labels, desired_subset=desired_actions, n_classes=args.n_action_classes
        )
        print(f"\nSelected {len(action_subset_ids)} action classes: {action_subset_ids}")
        for aid in action_subset_ids:
            name = NTU_ACTION_NAMES.get(int(aid), f"Action {int(aid)}")
            count = np.sum(action_labels == aid)
            print(f"  A{int(aid):02d}: {name} ({count} samples)")

        # Select identity subset
        if args.identity_subset:
            identity_subset_ids = args.identity_subset
        else:
            identity_subset_ids = select_identity_subset(
                identity_labels, n_identities=args.n_identities
            )
        print(f"\nSelected {len(identity_subset_ids)} identities: {identity_subset_ids}")

    # ---- Prepare SEPARATE subsets for action and identity plots -------------
    # Top row (a,b): filter by action classes only, keep all identities → more data
    # Bottom row (c,d): filter by identities only, keep all actions → more data
    # Each row uses identical data points with different coloring.
    dr_label = args.dr_method.upper()

    if action_subset_ids is not None:
        [act_embs_act, id_embs_act], act_labels_act, id_labels_act = stratified_sample(
            [action_embs, identity_embs], action_labels, identity_labels,
            action_subset=action_subset_ids, rng=rng,
        )
        print(f"\nAction plot subset: {len(act_embs_act)} samples, "
              f"{len(np.unique(act_labels_act))} actions, "
              f"{len(np.unique(id_labels_act))} identities")
    else:
        act_embs_act, id_embs_act = action_embs, identity_embs
        act_labels_act, id_labels_act = action_labels, identity_labels

    if identity_subset_ids is not None:
        [id_embs_id, act_embs_id], act_labels_id, id_labels_id = stratified_sample(
            [identity_embs, action_embs], action_labels, identity_labels,
            identity_subset=identity_subset_ids, rng=rng,
        )
        print(f"Identity plot subset: {len(id_embs_id)} samples, "
              f"{len(np.unique(act_labels_id))} actions, "
              f"{len(np.unique(id_labels_id))} identities")
    else:
        id_embs_id, act_embs_id = identity_embs, action_embs
        act_labels_id, id_labels_id = action_labels, identity_labels

    # ---- DR on action embeddings (action-subset data) ---------------------
    print(f"\n--- {dr_label}: Action embeddings ---")
    action_coords = run_dr(
        act_embs_act, method=args.dr_method,
        perplexity=args.perplexity, n_iter=args.n_iter,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, spread=args.spread,
        metric=args.umap_metric,
        random_state=args.seed, pca_first=args.pca_first,
    )

    # ---- DR on identity embeddings (identity-subset data) -----------------
    print(f"\n--- {dr_label}: Identity embeddings ---")
    identity_coords = run_dr(
        id_embs_id, method=args.dr_method,
        perplexity=args.perplexity_identity, n_iter=args.n_iter,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist, spread=args.spread,
        metric=args.umap_metric,
        random_state=args.seed, pca_first=args.pca_first,
    )

    # ---- Prepare label name dicts -----------------------------------------
    action_name_dict = NTU_ACTION_NAMES if args.dataset in ("ntu", "ntu_smoke", "ntu_small") else None
    show_title = not args.no_titles

    # Build suffix for filenames to distinguish different runs
    suffix = ""
    if args.feature_source == 'classifier':
        suffix += "_clf"
    if args.pooling != 'mean':
        suffix += f"_{args.pooling}"
    if args.dr_method == 'umap':
        suffix += "_umap"

    # Top row uses action-subset data, bottom row uses identity-subset data
    ps = 20  # point size

    # ---- Plot 1: Action embeddings x Action class -------------------------
    print(f"\n--- Plot 1: Action embeddings colored by action class ---")
    plot_tsne_v2(
        action_coords, act_labels_act,
        save_prefix=os.path.join(args.output_dir, f"tsne_action_by_action{suffix}"),
        label_names=action_name_dict,
        show_title=show_title,
        title=f"Action Embeddings by Action Class ({dr_label})",
        point_size=ps, alpha=0.7,
        label_type="Action",
        figsize=(5, 5),
    )

    # ---- Plot 2: Action embeddings x Identity (privacy check) -------------
    # Same data as plot 1, colored by identity instead. Many identities → text annotation.
    print(f"\n--- Plot 2: Action embeddings colored by identity ---")
    plot_tsne_v2(
        action_coords, id_labels_act,
        save_prefix=os.path.join(args.output_dir, f"tsne_action_by_identity{suffix}"),
        label_names=None,
        show_title=show_title,
        title=f"Action Embeddings by Identity ({dr_label})",
        point_size=ps, alpha=0.7,
        label_type="Identity",
        figsize=(5, 5),
    )

    # ---- Plot 3: Identity embeddings x Identity ---------------------------
    print(f"\n--- Plot 3: Identity embeddings colored by identity ---")
    plot_tsne_v2(
        identity_coords, id_labels_id,
        save_prefix=os.path.join(args.output_dir, f"tsne_identity_by_identity{suffix}"),
        label_names=None,
        show_title=show_title,
        title=f"Identity Embeddings by Identity ({dr_label})",
        point_size=ps, alpha=0.7,
        label_type="Identity",
        figsize=(5, 5),
    )

    # ---- Plot 4: Identity embeddings x Action class (disentanglement check)
    # Same data as plot 3, colored by action instead.
    print(f"\n--- Plot 4: Identity embeddings colored by action class ---")
    plot_tsne_v2(
        identity_coords, act_labels_id,
        save_prefix=os.path.join(args.output_dir, f"tsne_identity_by_action{suffix}"),
        label_names=action_name_dict,
        show_title=show_title,
        title=f"Identity Embeddings by Action Class ({dr_label})",
        point_size=ps, alpha=0.7,
        label_type="Action",
        figsize=(5, 5),
    )

    # ---- Save raw embeddings + coords for reuse --------------------------
    npz_path = os.path.join(args.output_dir, f"tsne_embeddings{suffix}.npz")
    np.savez_compressed(
        npz_path,
        # DR coordinates
        action_coords=action_coords,
        identity_coords=identity_coords,
        # Action-plot subset
        act_embs_act=act_embs_act,
        act_labels_act=act_labels_act,
        id_labels_act=id_labels_act,
        # Identity-plot subset
        id_embs_id=id_embs_id,
        act_labels_id=act_labels_id,
        id_labels_id=id_labels_id,
        # Full raw embeddings
        action_embs=action_embs,
        identity_embs=identity_embs,
        action_labels=action_labels,
        identity_labels=identity_labels,
        # Subset IDs used
        action_subset_ids=np.array(action_subset_ids) if action_subset_ids else np.array([]),
        identity_subset_ids=np.array(identity_subset_ids) if identity_subset_ids else np.array([]),
        # Config
        dr_method=np.array(args.dr_method),
        feature_source=np.array(args.feature_source),
        pooling=np.array(args.pooling),
    )
    print(f"\nRaw embeddings + {dr_label} coords saved to {npz_path}")
    print("All done.")


if __name__ == "__main__":
    main()
