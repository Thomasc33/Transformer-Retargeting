#!/usr/bin/env python3
"""
Replot action/identity embedding visualizations from saved embeddings.

Generates multiple projection methods:
  - LDA: Linear Discriminant Analysis (supervised by action labels)
  - LDA+t-SNE: LDA dimensionality reduction then t-SNE for layout
  - LDA+UMAP: LDA dimensionality reduction then UMAP for layout
  - Supervised UMAP: UMAP with label guidance for clean clusters
  - t-SNE / UMAP: Unsupervised nonlinear projections

All methods use the highlight approach: gray background of all data,
bold colors for selected action classes. Dense 2000-point plots.

CPU-only: loads pre-extracted embeddings from NPZ, no model needed.
"""

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

HIGHLIGHT_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#ff7f00',  # orange
    '#984ea3',  # purple
]


# ---------------------------------------------------------------------------
# Dimensionality reduction methods
# ---------------------------------------------------------------------------

def run_lda(embeddings, labels, n_components=2):
    """LDA: find the 2D projection that maximally separates action classes."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print(f"Running LDA on {embeddings.shape} ({len(np.unique(labels))} classes)...")
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    coords = lda.fit_transform(embeddings, labels)
    explained = lda.explained_variance_ratio_
    print(f"  LDA complete. Explained variance: {explained[0]:.3f}, {explained[1]:.3f}")
    # Normalize to equal scale for aspect ratio
    from sklearn.preprocessing import StandardScaler
    coords = StandardScaler().fit_transform(coords)
    return coords


def run_lda_tsne(embeddings, labels, n_lda=10, perplexity=40, n_iter=5000):
    """LDA to n_lda dims, then t-SNE to 2D."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n_classes = len(np.unique(labels))
    n_lda = min(n_lda, n_classes - 1)

    print(f"Running LDA({n_lda}) + t-SNE(2) on {embeddings.shape}...")
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda_coords = lda.fit_transform(embeddings, labels)
    total_var = sum(lda.explained_variance_ratio_[:n_lda])
    print(f"  LDA: {n_lda} components capture {total_var:.1%} of discriminant variance")

    lda_scaled = StandardScaler().fit_transform(lda_coords)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords = tsne.fit_transform(lda_scaled)
    print(f"  t-SNE complete.")
    return coords


def run_lda_umap(embeddings, labels, n_lda=10, n_neighbors=30, min_dist=0.3, spread=1.5):
    """LDA to n_lda dims, then UMAP to 2D."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    import umap

    n_classes = len(np.unique(labels))
    n_lda = min(n_lda, n_classes - 1)

    print(f"Running LDA({n_lda}) + UMAP(2) on {embeddings.shape}...")
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda_coords = lda.fit_transform(embeddings, labels)
    total_var = sum(lda.explained_variance_ratio_[:n_lda])
    print(f"  LDA: {n_lda} components capture {total_var:.1%} of discriminant variance")

    lda_scaled = StandardScaler().fit_transform(lda_coords)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric='euclidean',
        random_state=42,
    )
    coords = reducer.fit_transform(lda_scaled)
    print(f"  UMAP complete.")
    return coords


def run_supervised_umap(embeddings, labels, n_neighbors=15, min_dist=0.2, spread=1.5):
    """Supervised UMAP: uses action labels to guide cluster formation."""
    import umap
    print(f"Running supervised UMAP on {embeddings.shape} "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric='cosine',
        target_metric='categorical',
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings, y=labels.astype(int))
    print("  Supervised UMAP complete.")
    return coords


def run_tsne(embeddings, perplexity=50, n_iter=5000):
    from sklearn.manifold import TSNE
    print(f"Running t-SNE on {embeddings.shape} (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords = tsne.fit_transform(embeddings)
    print("  t-SNE complete.")
    return coords


def run_umap(embeddings, n_neighbors=30, min_dist=0.1, spread=1.0):
    import umap
    print(f"Running UMAP on {embeddings.shape} (n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}, spread={spread})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric='cosine',
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    print("  UMAP complete.")
    return coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_highlighted(coords, labels, highlight_ids, highlight_names,
                     save_path, label_type="Action", figsize=(4.5, 4.5)):
    """Gray background of all data, bold colors for highlighted classes."""
    fig, ax = plt.subplots(figsize=figsize)

    bg_mask = ~np.isin(labels, highlight_ids)
    ax.scatter(coords[bg_mask, 0], coords[bg_mask, 1],
               c='#e0e0e0', s=4, alpha=0.15, edgecolors='none',
               rasterized=True, zorder=1)

    handles = []
    for i, aid in enumerate(highlight_ids):
        mask = labels == aid
        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        name = highlight_names.get(aid, str(aid))
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=35, alpha=0.9, edgecolors='white',
                   linewidths=0.4, rasterized=True, zorder=2 + i)
        handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=7, label=name)
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')

    legend = ax.legend(
        handles=handles, title=label_type,
        loc="upper right", framealpha=0.92,
        handletextpad=0.3, borderaxespad=0.3,
        fontsize=8, markerscale=1.0,
        edgecolor='#cccccc',
    )
    legend.get_title().set_fontsize(9)

    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path}.{ext}", format=ext)
        print(f"  Saved {save_path}.{ext}")
    plt.close(fig)


def plot_all_colored(coords, labels, class_ids, class_names,
                     save_path, label_type="Action", figsize=(4.5, 4.5)):
    """All points colored by class. Used when subset_only mode has no background."""
    fig, ax = plt.subplots(figsize=figsize)

    handles = []
    for i, aid in enumerate(class_ids):
        mask = labels == aid
        color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        name = class_names.get(aid, str(aid))
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=40, alpha=0.85, edgecolors='white',
                   linewidths=0.5, rasterized=True, zorder=2 + i)
        handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=7, label=name)
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')

    legend = ax.legend(
        handles=handles, title=label_type,
        loc="upper right", framealpha=0.92,
        handletextpad=0.3, borderaxespad=0.3,
        fontsize=8, markerscale=1.0,
        edgecolor='#cccccc',
    )
    legend.get_title().set_fontsize(9)

    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path}.{ext}", format=ext)
        print(f"  Saved {save_path}.{ext}")
    plt.close(fig)


def plot_identity_scatter(coords, identity_labels, save_path,
                          figsize=(4.5, 4.5)):
    """All points colored by identity. Key visual: NO clustering."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_ids = np.unique(identity_labels)
    n_ids = len(unique_ids)

    rng = np.random.default_rng(42)
    cmap = plt.colormaps.get_cmap("hsv").resampled(n_ids)
    color_indices = np.arange(n_ids)
    rng.shuffle(color_indices)
    id_to_color = {uid: cmap(color_indices[i] / n_ids) for i, uid in enumerate(unique_ids)}

    colors = np.array([id_to_color[l] for l in identity_labels])
    order = rng.permutation(len(identity_labels))

    ax.scatter(coords[order, 0], coords[order, 1],
               c=colors[order], s=15, alpha=0.55,
               edgecolors='none', rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')

    ax.annotate(f"{n_ids} identities",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path}.{ext}", format=ext)
        print(f"  Saved {save_path}.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replot embeddings with LDA/t-SNE/UMAP")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="paper/fig")
    parser.add_argument("--method", type=str, default="all",
                        choices=["lda", "lda_tsne", "lda_umap", "supervised_umap",
                                 "tsne", "umap", "all"])
    # t-SNE params
    parser.add_argument("--perplexity", type=float, default=50)
    parser.add_argument("--n_iter", type=int, default=5000)
    # UMAP params
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--spread", type=float, default=1.5)
    # Highlight actions
    parser.add_argument("--highlight_actions", type=int, nargs="+",
                        default=[27, 8, 23, 10, 24],
                        help="Action IDs to highlight (1-indexed)")
    parser.add_argument("--subset_only", action="store_true",
                        help="Run DR only on highlighted action subset (cleaner separation)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"Loading embeddings from {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    action_embs = data['action_embs']
    action_labels = data['action_labels']
    identity_labels = data['identity_labels']
    print(f"  {len(action_embs)} samples, {action_embs.shape[1]}-d, "
          f"{len(np.unique(action_labels))} actions, {len(np.unique(identity_labels))} identities")

    available = set(np.unique(action_labels).astype(int))
    highlight_ids = [a for a in args.highlight_actions if a in available]
    highlight_names = {a: NTU_ACTION_NAMES.get(a, f"A{a}") for a in highlight_ids}
    print(f"\nHighlighting: {', '.join(f'{a}={highlight_names[a]}' for a in highlight_ids)}")
    for a in highlight_ids:
        print(f"  A{a:02d} {highlight_names[a]}: {np.sum(action_labels == a)} samples")

    # Keep full data for identity plot (separate DR)
    from sklearn.preprocessing import StandardScaler
    full_embs_scaled = StandardScaler().fit_transform(action_embs)
    full_identity_labels = identity_labels.copy()

    # Optionally subset to only highlighted actions for action plot DR
    if args.subset_only:
        mask = np.isin(action_labels.astype(int), highlight_ids)
        sub_embs = action_embs[mask]
        sub_action_labels = action_labels[mask]
        sub_identity_labels = identity_labels[mask]
        print(f"\n  Subset to highlighted actions: {len(sub_embs)} samples, "
              f"{len(np.unique(sub_action_labels))} actions, {len(np.unique(sub_identity_labels))} identities")
        sub_embs_scaled = StandardScaler().fit_transform(sub_embs)
    else:
        sub_embs_scaled = full_embs_scaled
        sub_action_labels = action_labels
        sub_identity_labels = identity_labels

    methods = []
    if args.method == "all":
        methods = ["lda", "lda_tsne", "lda_umap", "supervised_umap", "tsne", "umap"]
    else:
        methods = [args.method]

    sub_tag = "_sub" if args.subset_only else ""

    for method in methods:
        print(f"\n{'='*60}\nMethod: {method.upper()}\n{'='*60}")

        # --- Action plot: use subset data (clear action clusters) ---
        if method == "lda":
            coords = run_lda(sub_embs_scaled, sub_action_labels.astype(int))
            suffix = f"_lda{sub_tag}"
        elif method == "lda_tsne":
            n_lda = min(10, len(highlight_ids) - 1) if args.subset_only else 10
            coords = run_lda_tsne(sub_embs_scaled, sub_action_labels.astype(int),
                                  n_lda=n_lda, perplexity=min(40, len(sub_embs_scaled)//4),
                                  n_iter=args.n_iter)
            suffix = f"_lda_tsne{sub_tag}"
        elif method == "lda_umap":
            n_lda = min(10, len(highlight_ids) - 1) if args.subset_only else 10
            coords = run_lda_umap(sub_embs_scaled, sub_action_labels.astype(int),
                                  n_lda=n_lda,
                                  n_neighbors=min(args.n_neighbors, len(sub_embs_scaled)//5),
                                  min_dist=0.3, spread=args.spread)
            suffix = f"_lda_umap{sub_tag}"
        elif method == "supervised_umap":
            coords = run_supervised_umap(sub_embs_scaled, sub_action_labels,
                                         n_neighbors=min(15, len(sub_embs_scaled)//5),
                                         min_dist=0.2,
                                         spread=args.spread)
            suffix = f"_sup_umap{sub_tag}"
        elif method == "tsne":
            coords = run_tsne(sub_embs_scaled,
                              min(args.perplexity, len(sub_embs_scaled)//4),
                              args.n_iter)
            suffix = f"_tsne_v2{sub_tag}"
        else:
            coords = run_umap(sub_embs_scaled,
                              min(args.n_neighbors, len(sub_embs_scaled)//5),
                              args.min_dist, args.spread)
            suffix = f"_umap_v2{sub_tag}"

        print(f"\n--- Action embeddings by action class ({method}) ---")
        if args.subset_only:
            plot_all_colored(
                coords, sub_action_labels, highlight_ids, highlight_names,
                save_path=os.path.join(args.output_dir, f"tsne_action_by_action{suffix}"),
                label_type="Action",
            )
        else:
            plot_highlighted(
                coords, sub_action_labels, highlight_ids, highlight_names,
                save_path=os.path.join(args.output_dir, f"tsne_action_by_action{suffix}"),
                label_type="Action",
            )

        # --- Identity plot: use FULL data with unsupervised DR (no action clustering) ---
        print(f"\n--- Action embeddings by identity ({method}, full data unsupervised) ---")
        if args.subset_only:
            # Run separate unsupervised UMAP on full data for identity plot
            id_coords = run_umap(full_embs_scaled, n_neighbors=30, min_dist=0.3, spread=1.5)
            plot_identity_scatter(
                id_coords, full_identity_labels,
                save_path=os.path.join(args.output_dir, f"tsne_action_by_identity{suffix}"),
            )
        else:
            plot_identity_scatter(
                coords, sub_identity_labels,
                save_path=os.path.join(args.output_dir, f"tsne_action_by_identity{suffix}"),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
