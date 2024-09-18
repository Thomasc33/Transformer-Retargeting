#!/usr/bin/env python3
"""
Rank NTU-60 actions by retargeting quality for our model.

For each action, computes:
1. Motion similarity: cosine similarity of velocity profiles (source vs retargeted)
2. Visual interest: total joint displacement (want dynamic, interesting motions)
3. Structural quality: bone length variance (lower = more physically plausible)

Outputs a ranked list of actions to pick the best for qualitative figures.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import load_data, parse_file_name

NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_ACTIONS = {
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
    46: "Back pain", 47: "Neck pain", 48: "Nausea/vomiting", 49: "Fan self",
}


def seq_to_joints(seq):
    T = seq.shape[0]
    return seq.reshape(T, 25, 3)


def compute_motion_similarity(src_joints, ret_joints):
    """Cosine similarity of velocity profiles."""
    T = min(src_joints.shape[0], ret_joints.shape[0])
    src_vel = np.diff(src_joints[:T], axis=0).reshape(T - 1, -1)
    ret_vel = np.diff(ret_joints[:T], axis=0).reshape(T - 1, -1)
    sims = []
    for t in range(T - 1):
        ns, nr = np.linalg.norm(src_vel[t]), np.linalg.norm(ret_vel[t])
        if ns > 1e-6 and nr > 1e-6:
            sims.append(np.dot(src_vel[t], ret_vel[t]) / (ns * nr))
    return np.mean(sims) if sims else 0.0


def compute_visual_interest(joints):
    """Total displacement of key joints (head, hands, feet)."""
    key = [3, 7, 11, 15, 19]
    total = 0.0
    for j in key:
        total += np.sum(np.linalg.norm(np.diff(joints[:, j, :], axis=0), axis=1))
    return total


def compute_bone_consistency(joints):
    """Std of bone lengths across frames (lower = more consistent)."""
    T = joints.shape[0]
    bone_lens = np.zeros((T, len(NTU_BONES)))
    for t in range(T):
        for bi, (i, j) in enumerate(NTU_BONES):
            bone_lens[t, bi] = np.linalg.norm(joints[t, i] - joints[t, j])
    return np.mean(np.std(bone_lens, axis=0))


def main():
    print("Loading raw data...")
    raw_data = load_data('ntu', T=64)

    print("Loading retargeted data (ours)...")
    with open('output/retargeted_data/disentangled_tmr_stable_retargeted.pkl', 'rb') as f:
        ours_data = pickle.load(f)

    # Also load baselines for comparison quality
    print("Loading DMR retargeted data...")
    with open('output/retargeted_data/dmr_ntu_cv_retargeted.pkl', 'rb') as f:
        dmr_data = pickle.load(f)

    print("Loading PMR retargeted data...")
    with open('output/retargeted_data/pmr_ntu_cv_retargeted.pkl', 'rb') as f:
        pmr_data = pickle.load(f)

    # Per-action metrics
    action_metrics = defaultdict(lambda: {
        'motion_sims': [], 'visual_interest': [], 'bone_consistency': [],
        'dmr_sims': [], 'pmr_sims': [], 'count': 0
    })

    common_fnames = set(raw_data.keys()) & set(ours_data.keys()) & set(dmr_data.keys()) & set(pmr_data.keys())
    print(f"\n{len(common_fnames)} common samples")

    # Sample up to 200 per action for speed
    action_samples = defaultdict(list)
    for fname in common_fnames:
        info = parse_file_name(fname, 'ntu')
        aid = info['A']
        if 1 <= aid <= 49:
            action_samples[aid].append(fname)

    rng = np.random.default_rng(42)
    for aid in sorted(action_samples.keys()):
        fnames = action_samples[aid]
        if len(fnames) > 200:
            fnames = list(rng.choice(fnames, 200, replace=False))

        for fname in fnames:
            src = seq_to_joints(raw_data[fname])
            ours = seq_to_joints(ours_data[fname])
            dmr = seq_to_joints(dmr_data[fname])
            pmr = seq_to_joints(pmr_data[fname])

            m = action_metrics[aid]
            m['motion_sims'].append(compute_motion_similarity(src, ours))
            m['visual_interest'].append(compute_visual_interest(ours))
            m['bone_consistency'].append(compute_bone_consistency(ours))
            m['dmr_sims'].append(compute_motion_similarity(src, dmr))
            m['pmr_sims'].append(compute_motion_similarity(src, pmr))
            m['count'] += 1

    # Rank
    print("\n" + "=" * 120)
    print(f"{'ID':>3} {'Action':<25} {'N':>4} {'OursSim':>8} {'DMRSim':>8} {'PMRSim':>8} "
          f"{'OursAdv':>8} {'VisInt':>8} {'BoneStd':>8} {'Score':>8}")
    print("=" * 120)

    scores = {}
    for aid in sorted(action_metrics.keys()):
        m = action_metrics[aid]
        ours_sim = np.mean(m['motion_sims'])
        dmr_sim = np.mean(m['dmr_sims'])
        pmr_sim = np.mean(m['pmr_sims'])
        vis = np.mean(m['visual_interest'])
        bone = np.mean(m['bone_consistency'])

        # Advantage over baselines
        ours_adv = ours_sim - max(dmr_sim, pmr_sim)

        # Combined score: motion quality * visual interest * advantage over baselines
        # Normalize visual interest to [0,1] range roughly
        vis_norm = min(vis / 15.0, 1.0)
        score = ours_sim * 0.4 + ours_adv * 0.3 + vis_norm * 0.3

        scores[aid] = score
        name = NTU_ACTIONS.get(aid, f"Action {aid}")
        print(f"{aid:>3} {name:<25} {m['count']:>4} {ours_sim:>8.3f} {dmr_sim:>8.3f} {pmr_sim:>8.3f} "
              f"{ours_adv:>8.3f} {vis:>8.1f} {bone:>8.4f} {score:>8.3f}")

    print("\n" + "=" * 80)
    print("TOP 10 ACTIONS (best for qualitative figure):")
    print("=" * 80)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (aid, sc) in enumerate(ranked[:10], 1):
        name = NTU_ACTIONS.get(aid, f"Action {aid}")
        m = action_metrics[aid]
        print(f"  {rank}. A{aid:02d} {name:<25} score={sc:.3f}  "
              f"sim={np.mean(m['motion_sims']):.3f}  vis={np.mean(m['visual_interest']):.1f}")

    print("\nBOTTOM 5 ACTIONS (worst):")
    for rank, (aid, sc) in enumerate(ranked[-5:], 1):
        name = NTU_ACTIONS.get(aid, f"Action {aid}")
        print(f"  {rank}. A{aid:02d} {name:<25} score={sc:.3f}")

    # Suggest 4 diverse best actions
    print("\n" + "=" * 80)
    print("SUGGESTED 4 DIVERSE ACTIONS:")
    print("=" * 80)
    # Pick from top, ensuring diversity (different body parts involved)
    categories = {
        'upper': {1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 44, 45, 47, 49},
        'lower': {16, 17, 24, 26},
        'full_body': {5, 6, 7, 8, 9, 25, 27, 35, 36, 42, 43, 46, 48},
    }
    selected = []
    used_cats = set()
    for aid, sc in ranked:
        cat = None
        for c, acts in categories.items():
            if aid in acts:
                cat = c
                break
        # Pick top from each category, then fill
        if len(selected) < 4:
            if cat not in used_cats or len(used_cats) >= 3:
                selected.append(aid)
                if cat:
                    used_cats.add(cat)

    for i, aid in enumerate(selected, 1):
        name = NTU_ACTIONS.get(aid, f"Action {aid}")
        print(f"  {i}. A{aid:02d} {name:<25} score={scores[aid]:.3f}")

    print(f"\nCommand: --actions {' '.join(str(a) for a in selected)}")


if __name__ == '__main__':
    main()
