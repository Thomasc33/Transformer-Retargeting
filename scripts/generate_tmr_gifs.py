#!/usr/bin/env python3
"""Generate GIFs for TMR dissertation presentation.

Creates:
1. Side-by-side comparison GIFs: Raw | β=0.2 | Full TMR
2. Overlay GIFs: Raw (gray ghost) + β=0.2 (orange)
3. Full alpha spectrum GIFs showing the smooth transition
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v2 as imageio
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.datasets import load_data, parse_file_name

NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

ACTION_NAMES = {
    6: 'Pickup', 8: 'Sit down', 10: 'Clapping',
    24: 'Kicking', 27: 'Jump up', 43: 'Falling',
}


def center_at_hip(j):
    c = j.copy()
    c -= c[0]
    return c


def rotate_to_view(j, az=15, el=10):
    """Subtle 3/4 view rotation. NTU data is front-facing so we use a small
    azimuth (15 deg) to add slight depth without looking like a side view."""
    az, el = np.radians(az), np.radians(el)
    Ry = np.array([[np.cos(az), 0, np.sin(az)],
                   [0, 1, 0],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    return (Rx @ Ry @ j.T).T


def draw(ax, joints, color, lw=2.5, ms=10, alpha=1.0):
    j = rotate_to_view(center_at_hip(joints))
    x, y = j[:, 0], j[:, 1]
    for i, k in NTU_BONES:
        ax.plot([x[i], x[k]], [y[i], y[k]], '-', color=color,
                linewidth=lw, alpha=alpha, solid_capstyle='round')
    ax.scatter(x, y, c=color, s=ms, zorder=5, alpha=alpha,
              edgecolors='white', linewidths=0.3)


def fig_to_array(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    return img


def compute_limits(seqs):
    all_pts = []
    for seq in seqs:
        for t in range(seq.shape[0]):
            j = rotate_to_view(center_at_hip(seq[t].reshape(25, 3)))
            all_pts.append(j)
    all_pts = np.concatenate(all_pts)
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    half = max(np.abs(all_pts[:, 0] - cx).max(),
               np.abs(all_pts[:, 1] - cy).max()) * 1.15
    return cx, cy, half


def main():
    output_dir = 'paper/gifs'
    os.makedirs(output_dir, exist_ok=True)

    print('Loading data...')
    raw = load_data('ntu', 64)

    # Load soft retarget and full TMR data
    data_paths = {}
    for a in ['0_1', '0_2', '0_3', '0_5']:
        p = f'output/frozen_ar_experiments/soft_a{a}/retargeted_ntu.pkl'
        if os.path.exists(p):
            label = 'a=' + a.replace('_', '.')
            data_paths[label] = p

    full_tmr_path = 'output/retargeted_data/disentangled_tmr_stable_retargeted.pkl'
    if os.path.exists(full_tmr_path):
        data_paths['Full TMR'] = full_tmr_path

    methods = {}
    for name, path in data_paths.items():
        with open(path, 'rb') as f:
            methods[name] = pickle.load(f)
        print(f'  Loaded {name}')

    common = set(raw.keys())
    for d in methods.values():
        common &= set(d.keys())

    # Pick samples
    actions = [10, 24, 8, 27]  # Clapping, Kicking, Sit down, Jump up
    samples = []
    for a in actions:
        for name in sorted(common):
            info = parse_file_name(name, 'ntu')
            if info['A'] == a:
                samples.append(name)
                break

    # =============================================
    # GIF Type 1: Side-by-side Raw | α=0.2 | Full TMR
    # =============================================
    print('\n=== Generating side-by-side GIFs (Raw | α=0.2 | Full TMR) ===')
    side_methods = {'Raw': raw}
    if 'a=0.2' in methods:
        side_methods['β=0.2'] = methods['a=0.2']
    if 'Full TMR' in methods:
        side_methods['Full TMR'] = methods['Full TMR']

    colors = {'Raw': '#333333', 'β=0.2': '#FF9800', 'Full TMR': '#2196F3'}

    for sample in samples:
        action = int(sample[17:20])
        aname = ACTION_NAMES.get(action, f'A{action}')

        seqs = {n: d[sample][:, :75] for n, d in side_methods.items()}
        T = min(s.shape[0] for s in seqs.values())
        cx, cy, half = compute_limits(list(seqs.values()))

        frames = []
        for t in range(1, T):  # Skip t=0 (dummy/init frame)
            n_panels = len(seqs)
            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3, 4))
            if n_panels == 1:
                axes = [axes]
            for mi, (mname, seq) in enumerate(seqs.items()):
                ax = axes[mi]
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                draw(ax, seq[t].reshape(25, 3), colors[mname])
                ax.set_title(mname, fontsize=12, fontweight='bold', color=colors[mname])
            fig.suptitle(f'{aname}  |  frame {t}/{T-1}', fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            frames.append(fig_to_array(fig))
            plt.close(fig)

        path = os.path.join(output_dir, f'sidebyside_A{action:03d}_{aname.lower().replace(" ", "_")}.gif')
        imageio.mimsave(path, frames, fps=12, loop=0)
        print(f'  Saved: {path} ({T} frames)')

    # =============================================
    # GIF Type 2: Overlay Raw (ghost) + α=0.2
    # =============================================
    print('\n=== Generating overlay GIFs (Raw ghost + α=0.2) ===')
    if 'a=0.2' in methods:
        soft02 = methods['a=0.2']
        for sample in samples:
            action = int(sample[17:20])
            aname = ACTION_NAMES.get(action, f'A{action}')
            raw_seq = raw[sample][:, :75]
            anon_seq = soft02[sample][:, :75]
            T = min(raw_seq.shape[0], anon_seq.shape[0])
            cx, cy, half = compute_limits([raw_seq, anon_seq])

            frames = []
            for t in range(1, T):  # Skip t=0 (dummy/init frame)
                fig, ax = plt.subplots(figsize=(4.5, 5))
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                draw(ax, raw_seq[t].reshape(25, 3), '#999999', lw=1.5, ms=5, alpha=0.3)
                draw(ax, anon_seq[t].reshape(25, 3), '#FF9800', lw=2.5, ms=10, alpha=0.9)
                legend = [
                    Line2D([0], [0], color='#999999', lw=2, alpha=0.4, label='Raw'),
                    Line2D([0], [0], color='#FF9800', lw=2.5, label='β=0.2'),
                ]
                ax.legend(handles=legend, loc='upper right', fontsize=9, framealpha=0.8)
                ax.set_title(f'{aname}  |  frame {t}/{T-1}', fontsize=11)
                plt.tight_layout()
                frames.append(fig_to_array(fig))
                plt.close(fig)

            path = os.path.join(output_dir, f'overlay_A{action:03d}_{aname.lower().replace(" ", "_")}.gif')
            imageio.mimsave(path, frames, fps=12, loop=0)
            print(f'  Saved: {path} ({T} frames)')

    # =============================================
    # GIF Type 3: Alpha spectrum (all alphas for one action)
    # =============================================
    print('\n=== Generating alpha spectrum GIFs ===')
    spectrum = {'Raw': (raw, '#333333')}
    alpha_colors = {'a=0.1': '#4CAF50', 'a=0.2': '#FF9800', 'a=0.3': '#FF5722',
                    'a=0.5': '#9C27B0', 'Full TMR': '#2196F3'}
    for name in ['a=0.1', 'a=0.2', 'a=0.3', 'a=0.5', 'Full TMR']:
        if name in methods:
            spectrum[name] = (methods[name], alpha_colors[name])

    for sample in samples[:2]:  # Just 2 actions for spectrum
        action = int(sample[17:20])
        aname = ACTION_NAMES.get(action, f'A{action}')

        seqs = {n: d[sample][:, :75] for n, (d, _) in spectrum.items()}
        T = min(s.shape[0] for s in seqs.values())
        cx, cy, half = compute_limits(list(seqs.values()))

        n_panels = len(spectrum)
        frames = []
        for t in range(1, T):  # Skip t=0 (dummy/init frame)
            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 2.2, 3.5))
            for mi, (mname, (mdata, mcolor)) in enumerate(spectrum.items()):
                ax = axes[mi]
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                seq = mdata[sample][:, :75]
                draw(ax, seq[t].reshape(25, 3), mcolor, lw=2.2, ms=8)
                label = mname if mname in ('Raw', 'Full TMR') else f'\u03b2={mname[2:]}'
                ax.set_title(label, fontsize=9, fontweight='bold', color=mcolor)
            fig.suptitle(f'{aname}  |  frame {t}/{T-1}', fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            frames.append(fig_to_array(fig))
            plt.close(fig)

        path = os.path.join(output_dir, f'spectrum_A{action:03d}_{aname.lower().replace(" ", "_")}.gif')
        imageio.mimsave(path, frames, fps=12, loop=0)
        print(f'  Saved: {path} ({T} frames)')

    print('\nAll GIFs done!')


if __name__ == '__main__':
    main()
