#!/usr/bin/env python3
"""
Rank all NTU-60 actions by retargeting quality.

For each action, tries multiple source->target person pairs, runs retargeting,
and scores how well the output preserves the source motion. Outputs:
1. A ranked CSV of all actions by average retargeting score
2. Gallery images for the top-N best actions

Scoring:
- Velocity cosine similarity: does the retargeted motion move in the same direction as source?
- Pose distance (per-frame MSE after centering): does the retargeted pose look like the source?
- Action amplitude: how much total motion is there? (more = more visually interesting)
"""

import argparse
import os
import sys
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import parse_file_name, sample_frames_fast, load_data
from scripts.visualize_retargeting import (
    load_model, prepare_input, retarget, tensor_to_joints,
    NTU_ACTIONS, NTU_BONES,
    draw_skeleton_2d, setup_2d_axis, apply_limits_2d,
    compute_global_limits_2d,
    COLOR_SOURCE, COLOR_TARGET_ID, COLOR_RETARGETED,
)


def score_retargeting_detailed(src_joints, tgt_joints, retarg_joints):
    """
    Detailed scoring of retargeting quality.
    src_joints, tgt_joints, retarg_joints: (T, V, 3)

    Returns dict with individual metrics and combined score.
    """
    T = min(src_joints.shape[0], retarg_joints.shape[0])
    src = src_joints[:T]
    ret = retarg_joints[:T]
    tgt = tgt_joints[:T]

    # 1. Velocity cosine similarity (does motion direction match?)
    src_vel = np.diff(src, axis=0).reshape(T-1, -1)  # (T-1, V*3)
    ret_vel = np.diff(ret, axis=0).reshape(T-1, -1)
    vel_sims = []
    for t in range(T-1):
        ns = np.linalg.norm(src_vel[t])
        nr = np.linalg.norm(ret_vel[t])
        if ns > 1e-6 and nr > 1e-6:
            vel_sims.append(np.dot(src_vel[t], ret_vel[t]) / (ns * nr))
    vel_cos = np.mean(vel_sims) if vel_sims else 0.0

    # 2. Centered pose similarity (per-frame, after removing translation)
    # Center each frame on its spine (joint 1) to remove global position differences
    src_centered = src - src[:, 1:2, :]  # center on mid-spine
    ret_centered = ret - ret[:, 1:2, :]
    tgt_centered = tgt - tgt[:, 1:2, :]
    # Normalize by skeleton scale (average bone length)
    src_scale = np.mean([np.linalg.norm(src_centered[0, i] - src_centered[0, j])
                         for i, j in NTU_BONES if np.linalg.norm(src_centered[0, i] - src_centered[0, j]) > 0]) + 1e-6
    ret_scale = np.mean([np.linalg.norm(ret_centered[0, i] - ret_centered[0, j])
                         for i, j in NTU_BONES if np.linalg.norm(ret_centered[0, i] - ret_centered[0, j]) > 0]) + 1e-6
    src_normed = src_centered / src_scale
    ret_normed = ret_centered / ret_scale
    pose_mse = np.mean(np.sum((src_normed - ret_normed) ** 2, axis=-1))  # avg over frames and joints

    # 3. How different is retargeted from target? (should be different if action is transferred)
    tgt_normed = tgt_centered / ret_scale
    tgt_diff = np.mean(np.sum((ret_normed - tgt_normed) ** 2, axis=-1))

    # 4. Source motion amplitude (total displacement of key joints)
    key_joints = [3, 7, 11, 15, 19]  # head, hands, feet
    src_amplitude = sum(
        np.sum(np.linalg.norm(np.diff(src[:, j, :], axis=0), axis=1))
        for j in key_joints
    )
    ret_amplitude = sum(
        np.sum(np.linalg.norm(np.diff(ret[:, j, :], axis=0), axis=1))
        for j in key_joints
    )

    # 5. Amplitude ratio (retargeted should have similar amplitude to source)
    amp_ratio = min(ret_amplitude, src_amplitude) / (max(ret_amplitude, src_amplitude) + 1e-6)

    # Combined score:
    # - High velocity cosine = motion direction matches (most important)
    # - Low pose MSE = poses look similar
    # - High amplitude ratio = similar amount of motion
    # - Some source amplitude = visually interesting
    combined = (
        vel_cos * 0.40 +                                  # motion direction
        max(0, 1.0 - pose_mse / 2.0) * 0.25 +           # pose similarity (inverted)
        amp_ratio * 0.20 +                                # amplitude preservation
        min(src_amplitude / 15.0, 1.0) * 0.15            # visual interest bonus
    )

    return {
        'combined': combined,
        'vel_cos': vel_cos,
        'pose_mse': pose_mse,
        'tgt_diff': tgt_diff,
        'src_amplitude': src_amplitude,
        'ret_amplitude': ret_amplitude,
        'amp_ratio': amp_ratio,
    }


def rank_actions(model, raw_data, device, dataset='ntu', seg=64, n_pairs=10):
    """
    For each action, try n_pairs source->target pairs, score, and return ranked list.
    """
    # Group samples by action and person
    action_persons = defaultdict(lambda: defaultdict(list))
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if 1 <= info['A'] <= 49:  # single-person actions only
            action_persons[info['A']][info['P']].append(fname)

    results = []

    for action_id in sorted(action_persons.keys()):
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
        pids = sorted(action_persons[action_id].keys())
        if len(pids) < 2:
            continue

        # Generate pairs
        pairs = []
        for pa in pids:
            for pb in pids:
                if pa != pb:
                    pairs.append((pa, pb))
        rng = np.random.default_rng(42 + action_id)
        rng.shuffle(pairs)
        pairs = pairs[:n_pairs]

        scores = []
        best_score = -1
        best_data = None

        for pa, pb in pairs:
            fname_a = sorted(action_persons[action_id][pa])[0]
            fname_b = sorted(action_persons[action_id][pb])[0]

            src_tensor = prepare_input(raw_data[fname_a], seg)
            tgt_tensor = prepare_input(raw_data[fname_b], seg)
            retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

            src_joints = tensor_to_joints(src_tensor)
            tgt_joints = tensor_to_joints(tgt_tensor)
            retarg_joints = tensor_to_joints(retarg_tensor)

            metrics = score_retargeting_detailed(src_joints, tgt_joints, retarg_joints)
            scores.append(metrics)

            if metrics['combined'] > best_score:
                best_score = metrics['combined']
                best_data = (pa, pb, src_joints, tgt_joints, retarg_joints, metrics)

        # Average across pairs
        avg = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}

        results.append({
            'action_id': action_id,
            'action_name': action_name,
            'n_persons': len(pids),
            'n_pairs_tested': len(pairs),
            'avg_combined': avg['combined'],
            'avg_vel_cos': avg['vel_cos'],
            'avg_pose_mse': avg['pose_mse'],
            'avg_amp_ratio': avg['amp_ratio'],
            'avg_src_amplitude': avg['src_amplitude'],
            'best_combined': best_score,
            'best_data': best_data,
        })

        print(f"  A{action_id:2d} {action_name:25s}  "
              f"avg={avg['combined']:.3f}  vel_cos={avg['vel_cos']:.3f}  "
              f"pose_mse={avg['pose_mse']:.3f}  amp_ratio={avg['amp_ratio']:.3f}  "
              f"amplitude={avg['src_amplitude']:.1f}  best={best_score:.3f}")

    # Sort by average combined score
    results.sort(key=lambda x: -x['avg_combined'])
    return results


def save_rankings_csv(results, output_path):
    """Save ranking results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'action_id', 'action_name', 'n_persons',
                         'avg_combined', 'avg_vel_cos', 'avg_pose_mse',
                         'avg_amp_ratio', 'avg_src_amplitude', 'best_combined'])
        for rank, r in enumerate(results, 1):
            writer.writerow([
                rank, r['action_id'], r['action_name'], r['n_persons'],
                f"{r['avg_combined']:.4f}", f"{r['avg_vel_cos']:.4f}",
                f"{r['avg_pose_mse']:.4f}", f"{r['avg_amp_ratio']:.4f}",
                f"{r['avg_src_amplitude']:.1f}", f"{r['best_combined']:.4f}",
            ])
    print(f"Saved rankings: {output_path}")


def generate_top_gallery(results, output_dir, top_n=12, candidates_per_action=20,
                         model=None, raw_data=None, device=None, dataset='ntu', seg=64):
    """
    For the top-N actions, generate a gallery of retargeting candidates.
    """
    gallery_dir = os.path.join(output_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    action_persons = defaultdict(lambda: defaultdict(list))
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if 1 <= info['A'] <= 49:
            action_persons[info['A']][info['P']].append(fname)

    all_gallery = {}  # action_id -> list of (score, fname, pa, pb, metrics)

    for rank, r in enumerate(results[:top_n]):
        action_id = r['action_id']
        action_name = r['action_name']
        pids = sorted(action_persons[action_id].keys())
        print(f"\n=== Gallery rank {rank+1}: A{action_id} {action_name} (avg={r['avg_combined']:.3f}) ===")

        pairs = []
        for pa in pids:
            for pb in pids:
                if pa != pb:
                    pairs.append((pa, pb))
        rng = np.random.default_rng(100 + action_id)
        rng.shuffle(pairs)
        pairs = pairs[:candidates_per_action]

        candidates = []
        for idx, (pa, pb) in enumerate(pairs):
            fname_a = sorted(action_persons[action_id][pa])[0]
            fname_b = sorted(action_persons[action_id][pb])[0]

            src_tensor = prepare_input(raw_data[fname_a], seg)
            tgt_tensor = prepare_input(raw_data[fname_b], seg)
            retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

            src_joints = tensor_to_joints(src_tensor)
            tgt_joints = tensor_to_joints(tgt_tensor)
            retarg_joints = tensor_to_joints(retarg_tensor)

            metrics = score_retargeting_detailed(src_joints, tgt_joints, retarg_joints)
            candidates.append((metrics['combined'], pa, pb, src_joints, tgt_joints, retarg_joints, metrics))

        # Sort by score
        candidates.sort(key=lambda x: -x[0])

        gallery_entries = []
        for ci, (score, pa, pb, src_j, tgt_j, ret_j, metrics) in enumerate(candidates):
            fname = f"rank{rank+1:02d}_A{action_id:02d}_{ci:02d}_P{pa}_P{pb}_s{score:.2f}.png"
            save_path = os.path.join(gallery_dir, fname)

            # Draw figure: 3 rows x 5 frames
            frame_indices = [8, 16, 32, 48, 63]
            fig, axes = plt.subplots(3, len(frame_indices),
                                     figsize=(len(frame_indices) * 2.0, 3 * 2.2))

            cx, cy, half_range = compute_global_limits_2d([src_j, tgt_j, ret_j])

            for row, (label, joints, color) in enumerate([
                (f"Source P{pa}", src_j, COLOR_SOURCE),
                (f"Target ID P{pb}", tgt_j, COLOR_TARGET_ID),
                ("Retargeted", ret_j, COLOR_RETARGETED),
            ]):
                for fi, fidx in enumerate(frame_indices):
                    ax = axes[row, fi]
                    draw_skeleton_2d(ax, joints[fidx], color=color,
                                     linewidth=2.5, joint_size=22)
                    setup_2d_axis(ax)
                    apply_limits_2d(ax, cx, cy, half_range)
                    if fi == 0:
                        ax.set_ylabel(label, fontsize=9, fontweight='bold', labelpad=4)
                    if row == 0:
                        ax.set_title(f"t={fidx}", fontsize=9, pad=3)

            fig.suptitle(
                f"[Rank {rank+1}] A{action_id}: {action_name}  |  P{pa}→P{pb}  |  "
                f"score={score:.3f}  vel={metrics['vel_cos']:.2f}  "
                f"pose={metrics['pose_mse']:.2f}  amp={metrics['amp_ratio']:.2f}",
                fontsize=9, y=1.01)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

            gallery_entries.append((score, fname, pa, pb, metrics))
            if ci < 3:
                print(f"  [{ci}] P{pa}→P{pb} score={score:.3f}")

        all_gallery[action_id] = gallery_entries

    # Generate index.html
    html_path = os.path.join(gallery_dir, "index.html")
    with open(html_path, 'w') as f:
        f.write("<!DOCTYPE html><html><head>\n")
        f.write("<title>Retargeting Gallery - Ranked by Action Quality</title>\n")
        f.write("<style>\n")
        f.write("body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:20px;max-width:1400px;margin:0 auto;padding:20px;}\n")
        f.write("h1{color:#fff;border-bottom:2px solid #444;padding-bottom:10px;}\n")
        f.write("h2{color:#8cf;margin-top:40px;border-bottom:1px solid #333;padding-bottom:5px;}\n")
        f.write(".card{display:inline-block;margin:8px;background:#2a2a2a;border-radius:8px;padding:10px;text-align:center;vertical-align:top;}\n")
        f.write(".card img{max-width:700px;width:100%;border-radius:4px;cursor:pointer;}\n")
        f.write(".card img:hover{opacity:0.8;}\n")
        f.write(".card .meta{font-size:11px;color:#aaa;margin-top:4px;}\n")
        f.write(".card .score{font-size:13px;color:#6f6;font-weight:bold;}\n")
        f.write(".summary{background:#222;padding:15px;border-radius:8px;margin:20px 0;}\n")
        f.write(".summary table{border-collapse:collapse;width:100%;}\n")
        f.write(".summary th,.summary td{padding:6px 12px;text-align:left;border-bottom:1px solid #333;}\n")
        f.write(".summary th{color:#aaf;}\n")
        f.write(".summary tr:hover{background:#333;}\n")
        f.write("</style></head><body>\n")
        f.write("<h1>Retargeting Gallery - Actions Ranked by Quality</h1>\n")
        f.write("<p>Actions ranked by retargeting quality (velocity similarity, pose matching, amplitude preservation). "
                "Pick 4 actions with visually distinctive and well-retargeted motions for the paper figure.</p>\n")

        # Summary table
        f.write('<div class="summary"><table>\n')
        f.write("<tr><th>Rank</th><th>Action</th><th>Score</th><th>Vel Cos</th><th>Pose MSE</th><th>Amp Ratio</th><th>Amplitude</th></tr>\n")
        for rank, r in enumerate(results[:top_n]):
            f.write(f"<tr><td>{rank+1}</td><td>A{r['action_id']}: {r['action_name']}</td>"
                    f"<td>{r['avg_combined']:.3f}</td><td>{r['avg_vel_cos']:.3f}</td>"
                    f"<td>{r['avg_pose_mse']:.3f}</td><td>{r['avg_amp_ratio']:.3f}</td>"
                    f"<td>{r['avg_src_amplitude']:.1f}</td></tr>\n")
        f.write("</table></div>\n")

        for rank, r in enumerate(results[:top_n]):
            action_id = r['action_id']
            if action_id not in all_gallery:
                continue
            f.write(f'<h2>Rank {rank+1}: A{action_id} - {r["action_name"]} '
                    f'(avg score: {r["avg_combined"]:.3f})</h2>\n')
            for score, fname, pa, pb, metrics in all_gallery[action_id]:
                f.write(f'<div class="card">\n')
                f.write(f'  <img src="{fname}" onclick="window.open(this.src)">\n')
                f.write(f'  <div class="score">Score: {score:.3f}</div>\n')
                f.write(f'  <div class="meta">P{pa}→P{pb} | vel={metrics["vel_cos"]:.2f} '
                        f'pose_mse={metrics["pose_mse"]:.2f} amp={metrics["amp_ratio"]:.2f}</div>\n')
                f.write(f'</div>\n')

        f.write("</body></html>\n")

    total = sum(len(v) for v in all_gallery.values())
    print(f"\n=== Gallery complete: {total} images in {gallery_dir}/ ===")
    print(f"  Browse: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Rank NTU actions by retargeting quality")
    parser.add_argument("--checkpoint", default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth")
    parser.add_argument("--dataset", default="ntu")
    parser.add_argument("--output_dir", default="paper/fig")
    parser.add_argument("--seg", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_pairs", type=int, default=10,
                        help="Person pairs to test per action for ranking")
    parser.add_argument("--top_n", type=int, default=12,
                        help="Number of top actions to generate gallery for")
    parser.add_argument("--candidates_per_action", type=int, default=20,
                        help="Gallery candidates per top action")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    raw_data = load_data(args.dataset, args.seg)
    print(f"  {len(raw_data)} sequences")

    print("Loading model...")
    model = load_model(args.checkpoint, args.device, args.dataset)

    print("\n=== Ranking all actions by retargeting quality ===\n")
    results = rank_actions(model, raw_data, args.device, args.dataset, args.seg,
                           n_pairs=args.n_pairs)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "action_retargeting_rankings.csv")
    save_rankings_csv(results, csv_path)

    # Print top/bottom summary
    print("\n=== TOP 15 ACTIONS (best retargeting) ===")
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:2d}. A{r['action_id']:2d} {r['action_name']:25s}  score={r['avg_combined']:.3f}")

    print("\n=== BOTTOM 10 ACTIONS (worst retargeting) ===")
    for i, r in enumerate(results[-10:]):
        print(f"  {49-9+i:2d}. A{r['action_id']:2d} {r['action_name']:25s}  score={r['avg_combined']:.3f}")

    # Generate gallery for top actions
    print(f"\n=== Generating gallery for top {args.top_n} actions ===")
    generate_top_gallery(
        results, args.output_dir, top_n=args.top_n,
        candidates_per_action=args.candidates_per_action,
        model=model, raw_data=raw_data, device=args.device,
        dataset=args.dataset, seg=args.seg,
    )

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
