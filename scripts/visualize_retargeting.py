#!/usr/bin/env python3
"""
Qualitative Visualization of Disentangled TMR Retargeting

Generates publication-quality skeleton visualizations for the paper:
1. Retargeting examples: Source (PersonA, ActionX) + Target identity (PersonB) -> Retargeted (PersonB, ActionX)
2. Self-reconstruction: Same person, same action -> output (shows reconstruction quality)
3. Cross-identity consistency: Same action from different sources, retargeted to same target

Output: paper/fig/qualitative_*.pdf (300 DPI)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.data.datasets import Cross_Data, parse_file_name, sample_frames_fast, load_data

# ---------------------------------------------------------------------------
# NTU RGB+D skeleton bone connections (0-indexed)
# From src/graph/ntu_rgb_d.py inward_ori_index (converted to 0-indexed)
# ---------------------------------------------------------------------------
NTU_BONES = [
    (1, 0), (1, 20), (20, 2), (2, 3),           # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21),  # left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),  # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),       # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),       # right leg
]

# NTU joint names (for reference)
NTU_JOINT_NAMES = [
    "base_spine", "mid_spine", "neck", "head",        # 0-3
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",     # 4-7
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",     # 8-11
    "l_hip", "l_knee", "l_ankle", "l_foot",           # 12-15
    "r_hip", "r_knee", "r_ankle", "r_foot",           # 16-19
    "spine", "l_thumb", "l_tip",                       # 20-22
    "r_thumb", "r_tip",                                # 23-24
]

# NTU-60 single-person action labels (1-indexed, actions 1-49 only)
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
    46: "Back pain", 47: "Neck pain", 48: "Nausea/vomiting",
    49: "Fan self",
}

# Actions chosen for figure diversity: visually distinctive motions
TARGET_ACTIONS = [23, 27, 10, 24]  # Hand waving, Jump up, Clapping, Kicking something


# ---------------------------------------------------------------------------
# Model loading (mirrors generate_retargeted_dataset.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device, dataset_name):
    """Load the stable TMR model from checkpoint."""
    from src.data.datasets import datasets as ds_cfg
    print(f"Loading model from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None:
        if isinstance(ckpt_args, dict):
            ckpt_args = argparse.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    num_class = ds_cfg[dataset_name]['num_class']

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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.")
    return model


def prepare_input(raw_seq, seg=64):
    """
    Prepare raw sequence for TMR input.
    Input: (Frames, V*C)   Output: (1, C, T, V, M) tensor
    """
    seq = sample_frames_fast(raw_seq, seg)
    tensor = torch.from_numpy(seq).float()
    T, VC = tensor.shape
    V, C = 25, 3
    tensor = tensor.reshape(T, V, C).permute(2, 0, 1)  # (C, T, V)
    tensor = tensor.unsqueeze(0).unsqueeze(-1)           # (1, C, T, V, 1)
    return tensor


def retarget(model, source_tensor, target_tensor, device):
    """
    Run retargeting through the model.
    Returns padded output of shape (1, C, T, V, 1).
    """
    x1 = source_tensor.to(device)
    x2 = target_tensor.to(device)
    with torch.no_grad():
        output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)
    # Pad first frame from target identity skeleton
    first_frame = x2[:, :, 0:1, :, :]
    output_padded = torch.cat([first_frame, output], dim=2)
    return output_padded


def tensor_to_joints(tensor):
    """
    Convert (1, C, T, V, 1) tensor to numpy (T, V, 3).
    """
    arr = tensor.squeeze(-1).squeeze(0)  # (C, T, V)
    arr = arr.permute(1, 2, 0)           # (T, V, C)
    return arr.cpu().numpy()


# ---------------------------------------------------------------------------
# Skeleton rendering
# ---------------------------------------------------------------------------

# Color palette (publication-friendly)
COLOR_SOURCE = '#2171b5'       # blue
COLOR_TARGET_ID = '#238b45'    # green
COLOR_RETARGETED = '#d94801'   # orange
COLOR_RECON = '#6a51a3'        # purple

BODY_PART_COLORS = {
    'torso':    '#555555',
    'l_arm':    '#1f77b4',
    'r_arm':    '#ff7f0e',
    'l_leg':    '#2ca02c',
    'r_leg':    '#d62728',
    'head':     '#9467bd',
}

def bone_to_part(i, j):
    """Classify a bone to a body part for coloring."""
    torso = {0, 1, 20}
    head = {2, 3}
    l_arm = {4, 5, 6, 7, 21, 22}
    r_arm = {8, 9, 10, 11, 23, 24}
    l_leg = {12, 13, 14, 15}
    r_leg = {16, 17, 18, 19}
    joints = {i, j}
    if joints & l_arm:
        return 'l_arm'
    if joints & r_arm:
        return 'r_arm'
    if joints & l_leg:
        return 'l_leg'
    if joints & r_leg:
        return 'r_leg'
    if joints & head:
        return 'head'
    return 'torso'



def center_at_hip(joints_3d):
    """Center skeleton at hip joint (joint 0) and return a copy."""
    centered = joints_3d.copy()
    hip = centered[0].copy()
    centered -= hip
    return centered


def rotate_to_view(joints_3d, elev_deg=15, azim_deg=45):
    """Rotate 3D joints to a 3/4 viewing angle.

    Applies Y-axis rotation (azimuth) then X-axis rotation (elevation).
    """
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    Ry = np.array([
        [np.cos(az), 0, np.sin(az)],
        [0, 1, 0],
        [-np.sin(az), 0, np.cos(az)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)],
    ])
    rotated = (Rx @ Ry @ joints_3d.T).T
    return rotated


def transform_frame(joints_3d):
    """Apply hip-centering and 3/4-view rotation to a single frame."""
    joints = center_at_hip(joints_3d)
    joints = rotate_to_view(joints)
    return joints

def draw_skeleton_3d(ax, joints_frame, color=None, alpha=1.0, linewidth=1.8,
                     joint_size=15, use_part_colors=False, label=None):
    """
    Draw a single skeleton frame on a 3D axis.
    joints_frame: (V, 3) numpy array
    """
    x, y, z = joints_frame[:, 0], joints_frame[:, 1], joints_frame[:, 2]

    # Draw joints
    if label is not None:
        ax.scatter(x, z, y, c=color or '#333333', s=joint_size, zorder=5,
                   alpha=alpha, label=label, edgecolors='white', linewidths=0.3)
    else:
        ax.scatter(x, z, y, c=color or '#333333', s=joint_size, zorder=5,
                   alpha=alpha, edgecolors='white', linewidths=0.3)

    # Draw bones
    for (i, j) in NTU_BONES:
        if use_part_colors:
            c = BODY_PART_COLORS[bone_to_part(i, j)]
        else:
            c = color or '#333333'
        ax.plot([x[i], x[j]], [z[i], z[j]], [y[i], y[j]],
                c=c, linewidth=linewidth, alpha=alpha, solid_capstyle='round')


def draw_skeleton_2d(ax, joints_frame, color=None, alpha=1.0, linewidth=2.5,
                     joint_size=25, use_part_colors=False, label=None):
    """
    Draw a single skeleton frame on a 2D axis (front view: X horizontal, Y vertical).
    joints_frame: (V, 3) numpy array — uses columns 0 (X) and 1 (Y).
    """
    transformed = transform_frame(joints_frame)
    x, y = transformed[:, 0], transformed[:, 1]

    # Draw bones first (behind joints)
    for (i, j) in NTU_BONES:
        if use_part_colors:
            c = BODY_PART_COLORS[bone_to_part(i, j)]
        else:
            c = color or '#333333'
        ax.plot([x[i], x[j]], [y[i], y[j]],
                c=c, linewidth=linewidth, alpha=alpha, solid_capstyle='round')

    # Draw joints on top
    if label is not None:
        ax.scatter(x, y, c=color or '#333333', s=joint_size, zorder=5,
                   alpha=alpha, label=label, edgecolors='white', linewidths=0.4)
    else:
        ax.scatter(x, y, c=color or '#333333', s=joint_size, zorder=5,
                   alpha=alpha, edgecolors='white', linewidths=0.4)


def setup_2d_axis(ax):
    """Configure a 2D axis for clean skeleton display (no ticks, no spines)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')


def setup_3d_axis(ax, joints_frame=None, elev=15, azim=-70):
    """Configure a 3D axis for skeleton display."""
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)

    if joints_frame is not None:
        # Set limits based on data
        center = joints_frame.mean(axis=0)
        max_range = np.abs(joints_frame - center).max() * 1.3
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)


def compute_global_limits(joints_list):
    """Compute global axis limits from a list of (T, V, 3) arrays."""
    all_pts = np.concatenate([j.reshape(-1, 3) for j in joints_list], axis=0)
    center = all_pts.mean(axis=0)
    max_range = np.abs(all_pts - center).max() * 1.3
    return center, max_range


def compute_global_limits_2d(joints_list):
    """Compute global 2D axis limits after hip-centering and 3/4-view rotation."""
    all_x, all_y = [], []
    for j_seq in joints_list:
        pts = j_seq.reshape(-1, 25, 3)
        for t in range(pts.shape[0]):
            transformed = transform_frame(pts[t])
            all_x.append(transformed[:, 0])
            all_y.append(transformed[:, 1])
    if not all_x:
        return 0, 0, 1
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    cx = (all_x.max() + all_x.min()) / 2
    cy = (all_y.max() + all_y.min()) / 2
    half_range = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * 1.05
    return cx, cy, half_range


def apply_limits_2d(ax, cx, cy, half_range):
    """Apply pre-computed 2D limits to axis."""
    ax.set_xlim(cx - half_range, cx + half_range)
    ax.set_ylim(cy - half_range, cy + half_range)


def apply_limits(ax, center, max_range, elev=15, azim=-70):
    """Apply pre-computed limits and view to axis."""
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)
    ax.view_init(elev=elev, azim=azim)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_samples_for_action(raw_data, action_id, dataset='ntu', max_persons=4):
    """
    Find filenames grouped by person for a given action.
    Returns dict: person_id -> [filename, ...]
    """
    persons = {}
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if info['A'] == action_id:
            pid = info['P']
            if pid not in persons:
                persons[pid] = []
            persons[pid].append(fname)
    # Return up to max_persons, sorted by person id for reproducibility
    selected = {}
    for pid in sorted(persons.keys())[:max_persons]:
        selected[pid] = sorted(persons[pid])[0]  # pick first filename per person
    return selected


def find_distinct_identity_pair(raw_data, dataset='ntu'):
    """
    Find two persons with very different bone lengths (tall vs short) for visual contrast.
    Returns (person_id_A, person_id_B, common_actions_list).
    """
    from collections import defaultdict
    person_bones = defaultdict(list)
    person_actions = defaultdict(set)

    for fname, seq in raw_data.items():
        info = parse_file_name(fname, dataset)
        pid, aid = info['P'], info['A']
        person_actions[pid].add(aid)

        # Compute average bone length from first frame
        if len(seq) > 0:
            frame = seq[0].reshape(25, 3) if seq[0].shape[0] == 75 else seq[0].reshape(-1, 3)[:25]
            total_bl = 0.0
            n = 0
            for (i, j) in NTU_BONES:
                if i < frame.shape[0] and j < frame.shape[0]:
                    bl = np.linalg.norm(frame[i] - frame[j])
                    if bl > 0:
                        total_bl += bl
                        n += 1
            if n > 0:
                person_bones[pid].append(total_bl / n)

    # Average bone length per person
    avg_bl = {pid: np.mean(bls) for pid, bls in person_bones.items() if len(bls) > 0}
    sorted_pids = sorted(avg_bl.keys(), key=lambda p: avg_bl[p])

    # Find pair with max bone length difference AND enough common actions
    best_pair = None
    best_diff = 0
    for small_pid in sorted_pids[:5]:
        for large_pid in sorted_pids[-5:]:
            if small_pid == large_pid:
                continue
            common = person_actions[small_pid] & person_actions[large_pid]
            # Filter to single-person actions only (1-49)
            common = {a for a in common if 1 <= a <= 49}
            diff = avg_bl[large_pid] - avg_bl[small_pid]
            if len(common) >= 4 and diff > best_diff:
                best_diff = diff
                best_pair = (small_pid, large_pid, sorted(common))

    if best_pair is None:
        # Fallback: just pick first two persons with common actions
        pids = sorted(person_actions.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                common = person_actions[pids[i]] & person_actions[pids[j]]
                common = {a for a in common if 1 <= a <= 49}
                if len(common) >= 4:
                    return (pids[i], pids[j], sorted(common))
        raise RuntimeError("Cannot find suitable person pair")

    return best_pair


def get_sample(raw_data, person_id, action_id, dataset='ntu'):
    """Get a specific sample by person and action."""
    for fname, seq in raw_data.items():
        info = parse_file_name(fname, dataset)
        if info['P'] == person_id and info['A'] == action_id:
            return fname, seq
    return None, None


# ---------------------------------------------------------------------------
# Figure 1: Retargeting Examples
# ---------------------------------------------------------------------------

def score_retargeting(src_joints, retarg_joints):
    """
    Score how well retargeting preserves the source motion.
    Higher = better. Combines:
    - Motion similarity: cosine similarity of per-frame velocity profiles
    - Visual interest: total joint displacement (more motion = more interesting)
    Returns (combined_score, motion_sim, visual_interest).
    """
    # Per-frame velocity: (T-1, V, 3)
    src_vel = np.diff(src_joints, axis=0)
    ret_vel = np.diff(retarg_joints[:src_joints.shape[0]], axis=0)

    # Flatten to (T-1, V*3)
    src_flat = src_vel.reshape(src_vel.shape[0], -1)
    ret_flat = ret_vel.reshape(ret_vel.shape[0], -1)

    # Per-frame cosine similarity, averaged
    sims = []
    for t in range(len(src_flat)):
        norm_s = np.linalg.norm(src_flat[t])
        norm_r = np.linalg.norm(ret_flat[t])
        if norm_s > 1e-6 and norm_r > 1e-6:
            sims.append(np.dot(src_flat[t], ret_flat[t]) / (norm_s * norm_r))
    motion_sim = np.mean(sims) if sims else 0.0

    # Visual interest: total displacement of key joints (wrists, ankles, head)
    key_joints = [3, 7, 11, 15, 19]  # head, l_hand, r_hand, l_foot, r_foot
    total_disp = 0.0
    for j in key_joints:
        total_disp += np.sum(np.linalg.norm(np.diff(retarg_joints[:, j, :], axis=0), axis=1))

    # Combined: motion preservation (0-1) + visual interest bonus
    combined = motion_sim * 0.7 + min(total_disp / 10.0, 1.0) * 0.3
    return combined, motion_sim, total_disp


def find_best_retargeting(model, raw_data, device, action_id, dataset='ntu', seg=64,
                          n_candidates=10, source_pid=None, target_pid=None):
    """
    Try multiple source/target person pairs for a given action.
    Returns the best (source_pid, target_pid, src_joints, tgt_joints, retarg_joints, score).
    """
    from collections import defaultdict

    person_fnames = defaultdict(list)
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if info['A'] == action_id:
            person_fnames[info['P']].append(fname)

    available_pids = sorted(person_fnames.keys())
    if len(available_pids) < 2:
        return None

    # If specific persons requested, try just those
    if source_pid is not None and target_pid is not None:
        if source_pid in person_fnames and target_pid in person_fnames:
            pairs = [(source_pid, target_pid)]
        else:
            return None
    else:
        # Generate candidate pairs: pick diverse pairs
        pairs = []
        for i, pa in enumerate(available_pids):
            for pb in available_pids:
                if pa != pb:
                    pairs.append((pa, pb))
        # Shuffle and limit
        rng = np.random.default_rng(42 + action_id)
        rng.shuffle(pairs)
        pairs = pairs[:n_candidates]

    best = None
    best_score = -1

    for pa, pb in pairs:
        fname_a = sorted(person_fnames[pa])[0]
        fname_b = sorted(person_fnames[pb])[0]
        src_seq = raw_data[fname_a]
        tgt_seq = raw_data[fname_b]

        src_tensor = prepare_input(src_seq, seg)
        tgt_tensor = prepare_input(tgt_seq, seg)
        retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

        src_joints = tensor_to_joints(src_tensor)
        tgt_joints = tensor_to_joints(tgt_tensor)
        retarg_joints = tensor_to_joints(retarg_tensor)

        score, sim, disp = score_retargeting(src_joints, retarg_joints)
        if score > best_score:
            best_score = score
            best = (pa, pb, src_joints, tgt_joints, retarg_joints, score, sim, disp)

    return best


def figure_retargeting_examples(model, raw_data, device, output_dir, dataset='ntu', seg=64,
                                actions=None, source_person=None, target_person=None,
                                n_candidates=20, selections=None):
    """
    For 4 action types, show:
      Row: Source | Target ID (ref pose) | Retargeted
      Each shows 5 key frames side-by-side.

    If selections is provided, uses those exact (action, src_pid, tgt_pid) tuples.
    Otherwise searches multiple person pairs per action to find the best retargeting.
    """
    print("\n=== Figure 1: Retargeting Examples ===")

    frame_indices = [8, 16, 32, 48, 63]  # skip t=0 (always target rest pose)
    n_frames = len(frame_indices)

    best_results = []

    if selections:
        # Use exact user-specified selections
        for action_id, src_pid, tgt_pid in selections:
            action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
            print(f"\n  Using selection: A{action_id} ({action_name}) P{src_pid}->P{tgt_pid}")
            _, src_seq = get_sample(raw_data, src_pid, action_id, dataset)
            _, tgt_seq = get_sample(raw_data, tgt_pid, action_id, dataset)
            if src_seq is None or tgt_seq is None:
                print(f"    WARNING: Missing data for P{src_pid} or P{tgt_pid} A{action_id}")
                continue
            src_tensor = prepare_input(src_seq, seg)
            tgt_tensor = prepare_input(tgt_seq, seg)
            retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)
            src_j = tensor_to_joints(src_tensor)
            tgt_j = tensor_to_joints(tgt_tensor)
            ret_j = tensor_to_joints(retarg_tensor)
            best_results.append((action_id, src_pid, tgt_pid, src_j, tgt_j, ret_j))
    else:
        # Search for best pairs
        if actions is None:
            actions = list(TARGET_ACTIONS)
        n_actions = min(4, len(actions))
        for action_id in actions[:n_actions]:
            action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
            print(f"\n  Searching best retargeting for A{action_id} ({action_name})...")
            result = find_best_retargeting(
                model, raw_data, device, action_id, dataset, seg,
                n_candidates=n_candidates,
                source_pid=source_person, target_pid=target_person,
            )
            if result is not None:
                pa, pb, src_j, tgt_j, ret_j, score, sim, disp = result
                print(f"    Best: P{pa}->P{pb}, score={score:.3f} (sim={sim:.3f}, disp={disp:.1f})")
                best_results.append((action_id, pa, pb, src_j, tgt_j, ret_j))
            else:
                print(f"    WARNING: No valid retargeting found for A{action_id}")

    if len(best_results) == 0:
        print("  ERROR: No valid retargetings found!")
        return

    n_rows = len(best_results)
    fig, axes = plt.subplots(n_rows, n_frames * 3,
                             figsize=(n_frames * 3 * 1.2, n_rows * 2.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (action_id, pid_a, pid_b, src_joints, tgt_joints, retarg_joints) in enumerate(best_results):
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")

        cx, cy, half_range = compute_global_limits_2d([src_joints, tgt_joints, retarg_joints])

        for fi, frame_idx in enumerate(frame_indices):
            # Source
            ax_src = axes[row, fi]
            draw_skeleton_2d(ax_src, src_joints[frame_idx], color=COLOR_SOURCE,
                             linewidth=2.0, joint_size=18)
            setup_2d_axis(ax_src)
            apply_limits_2d(ax_src, cx, cy, half_range)
            if fi == 0:
                ax_src.set_ylabel(f"{action_name}", fontsize=9, labelpad=4)
            if row == 0:
                ax_src.set_title(f"t={frame_idx}", fontsize=8, pad=3)

            # Target identity
            ax_tgt = axes[row, n_frames + fi]
            draw_skeleton_2d(ax_tgt, tgt_joints[frame_idx], color=COLOR_TARGET_ID,
                             linewidth=2.0, joint_size=18)
            setup_2d_axis(ax_tgt)
            apply_limits_2d(ax_tgt, cx, cy, half_range)
            if row == 0:
                ax_tgt.set_title(f"t={frame_idx}", fontsize=8, pad=3)

            # Retargeted
            ax_ret = axes[row, 2 * n_frames + fi]
            draw_skeleton_2d(ax_ret, retarg_joints[frame_idx], color=COLOR_RETARGETED,
                             linewidth=2.0, joint_size=18)
            setup_2d_axis(ax_ret)
            apply_limits_2d(ax_ret, cx, cy, half_range)
            if row == 0:
                ax_ret.set_title(f"t={frame_idx}", fontsize=8, pad=3)

    # Column group labels (use generic labels since persons may differ per row)
    fig.text(0.17, 0.99, "Source Motion", ha='center', fontsize=11,
             fontweight='bold', color=COLOR_SOURCE)
    fig.text(0.50, 0.99, "Target Identity", ha='center', fontsize=11,
             fontweight='bold', color=COLOR_TARGET_ID)
    fig.text(0.83, 0.99, "Retargeted Output", ha='center', fontsize=11,
             fontweight='bold', color=COLOR_RETARGETED)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "qualitative_retargeting.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    out_png = out_path.replace('.pdf', '.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_png}")


# ---------------------------------------------------------------------------
# Figure 2: Self-Reconstruction
# ---------------------------------------------------------------------------

def figure_self_reconstruction(model, raw_data, device, output_dir, dataset='ntu', seg=64):
    """
    Same person, same action: input vs reconstructed output.
    Shows reconstruction quality.
    """
    print("\n=== Figure 2: Self-Reconstruction ===")

    actions_to_show = [23, 27, 10, 8]  # Hand waving, Jump up, Clapping, Sit down
    frame_indices = [8, 16, 32, 48, 63]
    n_frames = len(frame_indices)

    # Find a person with all these actions
    from collections import defaultdict
    person_actions = defaultdict(set)
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if 1 <= info['A'] <= 49:
            person_actions[info['P']].add(info['A'])

    chosen_pid = None
    for pid in sorted(person_actions.keys()):
        if all(a in person_actions[pid] for a in actions_to_show):
            chosen_pid = pid
            break
    if chosen_pid is None:
        # Fallback: find person with most of these actions
        best_pid = max(person_actions.keys(),
                       key=lambda p: len(person_actions[p] & set(actions_to_show)))
        chosen_pid = best_pid
        actions_to_show = [a for a in actions_to_show if a in person_actions[chosen_pid]]

    print(f"  Person: P{chosen_pid}")
    n_actions = len(actions_to_show)

    fig, axes = plt.subplots(n_actions, n_frames * 2,
                             figsize=(n_frames * 2 * 1.3, n_actions * 2.2))
    if n_actions == 1:
        axes = axes[np.newaxis, :]

    for row, action_id in enumerate(actions_to_show):
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
        print(f"  Action {action_id}: {action_name}")

        _, src_seq = get_sample(raw_data, chosen_pid, action_id, dataset)
        if src_seq is None:
            print(f"    WARNING: Missing data for P{chosen_pid} A{action_id}")
            continue

        src_tensor = prepare_input(src_seq, seg)

        # Self-reconstruction: source_motion = target_skeleton = same person
        recon_tensor = retarget(model, src_tensor, src_tensor, device)

        src_joints = tensor_to_joints(src_tensor)
        recon_joints = tensor_to_joints(recon_tensor)

        cx, cy, half_range = compute_global_limits_2d([src_joints, recon_joints])

        for fi, frame_idx in enumerate(frame_indices):
            # Input
            ax_in = axes[row, fi]
            draw_skeleton_2d(ax_in, src_joints[frame_idx], color=COLOR_SOURCE,
                             linewidth=2.0, joint_size=18)
            setup_2d_axis(ax_in)
            apply_limits_2d(ax_in, cx, cy, half_range)
            if fi == 0:
                ax_in.set_ylabel(f"{action_name}", fontsize=9, labelpad=4)
            if row == 0:
                ax_in.set_title(f"Input (t={frame_idx})", fontsize=8, pad=3)

            # Reconstructed
            ax_rec = axes[row, n_frames + fi]
            draw_skeleton_2d(ax_rec, recon_joints[frame_idx], color=COLOR_RECON,
                             linewidth=2.0, joint_size=18)
            setup_2d_axis(ax_rec)
            apply_limits_2d(ax_rec, cx, cy, half_range)
            if row == 0:
                ax_rec.set_title(f"Reconstructed (t={frame_idx})", fontsize=8, pad=3)

    fig.text(0.27, 0.99, f"Input (Person {chosen_pid})", ha='center', fontsize=11,
             fontweight='bold', color=COLOR_SOURCE)
    fig.text(0.73, 0.99, "Self-Reconstruction", ha='center', fontsize=11,
             fontweight='bold', color=COLOR_RECON)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "qualitative_self_reconstruction.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.pdf', '.png')}")


# ---------------------------------------------------------------------------
# Figure 3: Cross-Identity Consistency
# ---------------------------------------------------------------------------

def figure_cross_identity(model, raw_data, device, output_dir, dataset='ntu', seg=64):
    """
    Same action performed by 3 different source identities, all retargeted to the same
    target identity. The retargeted outputs should look visually similar (same motion,
    same target body), demonstrating disentanglement.
    """
    print("\n=== Figure 3: Cross-Identity Consistency ===")

    pid_a, pid_b, common_actions = find_distinct_identity_pair(raw_data, dataset)

    # Find additional source identities
    from collections import defaultdict
    person_actions = defaultdict(set)
    for fname in raw_data.keys():
        info = parse_file_name(fname, dataset)
        if 1 <= info['A'] <= 49:
            person_actions[info['P']].add(info['A'])

    # Find 2 more persons with overlapping actions with pid_b
    source_pids = [pid_a]
    for pid in sorted(person_actions.keys()):
        if pid == pid_a or pid == pid_b:
            continue
        overlap = person_actions[pid] & person_actions[pid_b] & set(common_actions)
        if len(overlap) >= 2:
            source_pids.append(pid)
        if len(source_pids) >= 3:
            break

    n_sources = len(source_pids)
    print(f"  Source persons: {source_pids}")
    print(f"  Target person: P{pid_b}")

    # Pick 2 actions for this figure
    avail_actions = set(common_actions)
    for pid in source_pids:
        avail_actions &= person_actions[pid]
    avail_actions = sorted(avail_actions)
    actions_to_show = []
    for a in TARGET_ACTIONS:
        if a in avail_actions:
            actions_to_show.append(a)
        if len(actions_to_show) >= 2:
            break
    while len(actions_to_show) < 2 and avail_actions:
        a = avail_actions.pop(0)
        if a not in actions_to_show:
            actions_to_show.append(a)

    frame_indices = [8, 16, 32, 48, 63]
    n_frames = len(frame_indices)
    n_actions = len(actions_to_show)

    # Layout: for each action, n_sources+1 rows (sources + retargeted outputs)
    # Better: for each action block, row 0..n_sources-1 = retargeted from source_i,
    # with a strip above showing the source skeleton at the same frames.
    # Simpler: one figure per action with (n_sources * 2) rows:
    #   even rows = source, odd rows = retargeted

    for action_id in actions_to_show:
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
        print(f"\n  Action {action_id}: {action_name}")

        # Collect all data
        all_src_joints = []
        all_retarg_joints = []

        # Get target reference
        _, tgt_seq = get_sample(raw_data, pid_b, action_id, dataset)
        if tgt_seq is None:
            print(f"    WARNING: Target P{pid_b} missing A{action_id}")
            continue
        tgt_tensor = prepare_input(tgt_seq, seg)
        tgt_joints = tensor_to_joints(tgt_tensor)

        valid_sources = []
        for src_pid in source_pids:
            _, src_seq = get_sample(raw_data, src_pid, action_id, dataset)
            if src_seq is None:
                print(f"    WARNING: Source P{src_pid} missing A{action_id}")
                continue
            src_tensor = prepare_input(src_seq, seg)
            retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

            all_src_joints.append(tensor_to_joints(src_tensor))
            all_retarg_joints.append(tensor_to_joints(retarg_tensor))
            valid_sources.append(src_pid)

        if len(valid_sources) == 0:
            continue

        n_valid = len(valid_sources)
        # Figure: 2*n_valid rows (source row, retargeted row), n_frames columns
        fig, axes = plt.subplots(2 * n_valid, n_frames,
                                 figsize=(n_frames * 1.5, 2 * n_valid * 2.0))
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        all_joints_flat = all_src_joints + all_retarg_joints + [tgt_joints]
        cx, cy, half_range = compute_global_limits_2d(all_joints_flat)

        for si in range(n_valid):
            src_pid = valid_sources[si]
            src_j = all_src_joints[si]
            ret_j = all_retarg_joints[si]

            for fi, frame_idx in enumerate(frame_indices):
                # Source row
                ax_src = axes[2 * si, fi]
                draw_skeleton_2d(ax_src, src_j[frame_idx], color=COLOR_SOURCE,
                                 linewidth=2.0, joint_size=18)
                setup_2d_axis(ax_src)
                apply_limits_2d(ax_src, cx, cy, half_range)
                if fi == 0:
                    ax_src.set_ylabel(f"Source P{src_pid}", fontsize=8, labelpad=4)
                if si == 0:
                    ax_src.set_title(f"t={frame_idx}", fontsize=8, pad=3)

                # Retargeted row
                ax_ret = axes[2 * si + 1, fi]
                draw_skeleton_2d(ax_ret, ret_j[frame_idx], color=COLOR_RETARGETED,
                                 linewidth=2.0, joint_size=18)
                setup_2d_axis(ax_ret)
                apply_limits_2d(ax_ret, cx, cy, half_range)
                if fi == 0:
                    ax_ret.set_ylabel(f"Retarg -> P{pid_b}", fontsize=8, labelpad=4)

        fig.suptitle(f"Cross-Identity Consistency: \"{action_name}\" retargeted to Person {pid_b}",
                     fontsize=11, fontweight='bold', y=1.01)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        safe_name = action_name.replace(" ", "_").replace("/", "_").lower()
        out_path = os.path.join(output_dir, f"qualitative_cross_identity_{safe_name}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"    Saved: {out_path}")
        print(f"    Saved: {out_path.replace('.pdf', '.png')}")


# ---------------------------------------------------------------------------
# Figure 4 (bonus): Overlay comparison
# ---------------------------------------------------------------------------

def figure_overlay(model, raw_data, device, output_dir, dataset='ntu', seg=64):
    """
    Overlay source and retargeted on same axes for direct comparison.
    One row per action, 5 frames.
    """
    print("\n=== Figure 4: Overlay Comparison ===")
    pid_a, pid_b, common_actions = find_distinct_identity_pair(raw_data, dataset)

    actions_to_show = [a for a in TARGET_ACTIONS if a in common_actions][:3]
    if len(actions_to_show) == 0:
        actions_to_show = common_actions[:3]

    frame_indices = [8, 16, 32, 48, 63]
    n_frames = len(frame_indices)
    n_actions = len(actions_to_show)

    fig, axes = plt.subplots(n_actions, n_frames,
                             figsize=(n_frames * 1.8, n_actions * 2.5))
    if n_actions == 1:
        axes = axes[np.newaxis, :]

    for row, action_id in enumerate(actions_to_show):
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
        print(f"  Action {action_id}: {action_name}")

        _, src_seq = get_sample(raw_data, pid_a, action_id, dataset)
        _, tgt_seq = get_sample(raw_data, pid_b, action_id, dataset)
        if src_seq is None or tgt_seq is None:
            continue

        src_tensor = prepare_input(src_seq, seg)
        tgt_tensor = prepare_input(tgt_seq, seg)
        retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

        src_joints = tensor_to_joints(src_tensor)
        retarg_joints = tensor_to_joints(retarg_tensor)

        cx, cy, half_range = compute_global_limits_2d([src_joints, retarg_joints])

        for fi, frame_idx in enumerate(frame_indices):
            ax = axes[row, fi]
            # Draw source as semi-transparent
            draw_skeleton_2d(ax, src_joints[frame_idx], color=COLOR_SOURCE,
                             alpha=0.35, linewidth=1.5, joint_size=12,
                             label="Source" if fi == 0 and row == 0 else None)
            # Draw retargeted on top
            draw_skeleton_2d(ax, retarg_joints[frame_idx], color=COLOR_RETARGETED,
                             alpha=0.9, linewidth=2.5, joint_size=20,
                             label="Retargeted" if fi == 0 and row == 0 else None)
            setup_2d_axis(ax)
            apply_limits_2d(ax, cx, cy, half_range)
            if fi == 0:
                ax.set_ylabel(action_name, fontsize=9, labelpad=4)
            if row == 0:
                ax.set_title(f"t={frame_idx}", fontsize=8, pad=3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_SOURCE, linewidth=2, label=f'Source (P{pid_a})'),
        Line2D([0], [0], color=COLOR_RETARGETED, linewidth=2, label=f'Retargeted (P{pid_a} action -> P{pid_b} body)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
               frameon=True, fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    out_path = os.path.join(output_dir, "qualitative_overlay.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.pdf', '.png')}")


# ---------------------------------------------------------------------------
# Gallery mode: generate many candidates for manual selection
# ---------------------------------------------------------------------------

def figure_single_retargeting(src_joints, tgt_joints, retarg_joints,
                               action_name, src_pid, tgt_pid, score, sim, disp,
                               save_path):
    """
    Save a single retargeting example as one image.
    Layout: 3 rows (Source / Target ID / Retargeted) x 5 frame columns.
    Skips t=0 since it's always identical to the target rest pose.
    """
    frame_indices = [8, 16, 32, 48, 63]  # skip t=0
    n_frames = len(frame_indices)

    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 2.0, 3 * 2.2))

    cx, cy, half_range = compute_global_limits_2d([src_joints, tgt_joints, retarg_joints])

    labels_colors = [
        ("Source (P{})".format(src_pid), src_joints, COLOR_SOURCE),
        ("Target ID (P{})".format(tgt_pid), tgt_joints, COLOR_TARGET_ID),
        ("Retargeted", retarg_joints, COLOR_RETARGETED),
    ]

    for row, (row_label, joints, color) in enumerate(labels_colors):
        for fi, frame_idx in enumerate(frame_indices):
            ax = axes[row, fi]
            draw_skeleton_2d(ax, joints[frame_idx], color=color,
                             linewidth=2.5, joint_size=22)
            setup_2d_axis(ax)
            apply_limits_2d(ax, cx, cy, half_range)
            if fi == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold', labelpad=4)
            if row == 0:
                ax.set_title(f"t={frame_idx}", fontsize=9, pad=3)

    fig.suptitle(f"{action_name}  |  P{src_pid} -> P{tgt_pid}  |  "
                 f"score={score:.3f} (sim={sim:.3f}, disp={disp:.1f})",
                 fontsize=10, y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def generate_gallery(model, raw_data, device, output_dir, dataset='ntu', seg=64,
                     actions=None, n_candidates=25):
    """
    Generate a gallery of retargeting candidates for manual selection.
    For each action, tries n_candidates person pairs and saves each as a separate image.
    Images are named: gallery_{action_id}_{rank:02d}_P{src}_P{tgt}_s{score}.png
    Also generates an index.html for easy browsing.
    """
    from collections import defaultdict

    if actions is None:
        actions = list(TARGET_ACTIONS)

    gallery_dir = os.path.join(output_dir, "gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    all_results = {}  # action_id -> list of (score, path, metadata)

    for action_id in actions:
        action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
        print(f"\n=== Gallery: A{action_id} ({action_name}) ===")

        # Find all person pairs for this action
        person_fnames = defaultdict(list)
        for fname in raw_data.keys():
            info = parse_file_name(fname, dataset)
            if info['A'] == action_id:
                person_fnames[info['P']].append(fname)

        available_pids = sorted(person_fnames.keys())
        if len(available_pids) < 2:
            print(f"  Only {len(available_pids)} persons, skipping")
            continue

        # Generate all possible pairs, shuffle, take n_candidates
        pairs = []
        for pa in available_pids:
            for pb in available_pids:
                if pa != pb:
                    pairs.append((pa, pb))
        rng = np.random.default_rng(42 + action_id)
        rng.shuffle(pairs)
        pairs = pairs[:n_candidates]

        results = []
        for idx, (pa, pb) in enumerate(pairs):
            fname_a = sorted(person_fnames[pa])[0]
            fname_b = sorted(person_fnames[pb])[0]

            src_tensor = prepare_input(raw_data[fname_a], seg)
            tgt_tensor = prepare_input(raw_data[fname_b], seg)
            retarg_tensor = retarget(model, src_tensor, tgt_tensor, device)

            src_joints = tensor_to_joints(src_tensor)
            tgt_joints = tensor_to_joints(tgt_tensor)
            retarg_joints = tensor_to_joints(retarg_tensor)

            score, sim, disp = score_retargeting(src_joints, retarg_joints)
            results.append((score, sim, disp, pa, pb, src_joints, tgt_joints, retarg_joints))

        # Sort by score descending
        results.sort(key=lambda x: -x[0])

        action_results = []
        for rank, (score, sim, disp, pa, pb, src_j, tgt_j, ret_j) in enumerate(results):
            fname = f"gallery_A{action_id:02d}_{rank:02d}_P{pa}_P{pb}_s{score:.2f}.png"
            save_path = os.path.join(gallery_dir, fname)
            figure_single_retargeting(
                src_j, tgt_j, ret_j,
                action_name, pa, pb, score, sim, disp,
                save_path,
            )
            action_results.append((score, fname, pa, pb, sim, disp))
            print(f"  [{rank:2d}] P{pa}->P{pb} score={score:.3f} (sim={sim:.3f}, disp={disp:.1f})")

        all_results[action_id] = action_results

    # Generate index.html for easy browsing
    html_path = os.path.join(gallery_dir, "index.html")
    with open(html_path, 'w') as f:
        f.write("<!DOCTYPE html><html><head>\n")
        f.write("<title>Retargeting Gallery</title>\n")
        f.write("<style>body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:20px;}")
        f.write("h1{color:#fff;}h2{color:#aaf;margin-top:40px;}")
        f.write(".card{display:inline-block;margin:8px;background:#2a2a2a;border-radius:8px;")
        f.write("padding:8px;text-align:center;vertical-align:top;}")
        f.write(".card img{max-width:600px;border-radius:4px;}")
        f.write(".card .meta{font-size:12px;color:#aaa;margin-top:4px;}")
        f.write(".card .score{font-size:14px;color:#6f6;font-weight:bold;}")
        f.write("</style></head><body>\n")
        f.write("<h1>Retargeting Gallery - Pick Your Favorites</h1>\n")
        f.write("<p>Images sorted by retargeting quality score (highest first). "
                "t=0 skipped since it's always the target rest pose.</p>\n")

        for action_id in actions:
            if action_id not in all_results:
                continue
            action_name = NTU_ACTIONS.get(action_id, f"A{action_id}")
            f.write(f"<h2>A{action_id}: {action_name}</h2>\n")
            for score, fname, pa, pb, sim, disp in all_results[action_id]:
                f.write(f'<div class="card">\n')
                f.write(f'  <img src="{fname}" alt="{fname}">\n')
                f.write(f'  <div class="score">Score: {score:.3f}</div>\n')
                f.write(f'  <div class="meta">P{pa} → P{pb} | sim={sim:.3f} disp={disp:.1f}</div>\n')
                f.write(f'</div>\n')

        f.write("</body></html>\n")

    total = sum(len(v) for v in all_results.values())
    print(f"\n=== Gallery complete: {total} images in {gallery_dir}/ ===")
    print(f"  Browse: {html_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate qualitative retargeting visualizations")
    parser.add_argument("--checkpoint", default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth",
                        help="Path to TMR checkpoint")
    parser.add_argument("--dataset", default="ntu", help="Dataset name")
    parser.add_argument("--output_dir", default="paper/fig", help="Output directory for figures")
    parser.add_argument("--seg", type=int, default=64, help="Sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", type=int, nargs="+", default=None,
                        help="Action IDs to visualize (default: 23 27 10 24)")
    parser.add_argument("--source_person", type=int, default=None,
                        help="Force a specific source person ID")
    parser.add_argument("--target_person", type=int, default=None,
                        help="Force a specific target person ID")
    parser.add_argument("--n_candidates", type=int, default=20,
                        help="Number of person pairs to try per action")
    parser.add_argument("--retargeting_only", action="store_true",
                        help="Only generate the retargeting figure (skip self-recon, cross-id, overlay)")
    parser.add_argument("--gallery", action="store_true",
                        help="Generate gallery of ~100 candidates for manual selection")
    parser.add_argument("--gallery_actions", type=int, nargs="+", default=None,
                        help="Action IDs for gallery (default: all single-person actions 1-49)")
    parser.add_argument("--selections", type=str, nargs="+", default=None,
                        help="Exact selections as action:src_pid:tgt_pid (e.g. 27:9:13 40:33:4)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load raw data (we need access to individual sequences by filename)
    print("Loading raw data...")
    raw_data = load_data(args.dataset, args.seg)
    print(f"  Loaded {len(raw_data)} sequences.")

    # 2. Load model
    model = load_model(args.checkpoint, args.device, args.dataset)

    # Parse selections if provided
    selections = None
    if args.selections:
        selections = []
        for s in args.selections:
            parts = s.split(':')
            selections.append((int(parts[0]), int(parts[1]), int(parts[2])))
        print(f"Using {len(selections)} manual selections:")
        for a, sp, tp in selections:
            print(f"  A{a} ({NTU_ACTIONS.get(a, '?')}): P{sp} -> P{tp}")

    # 3. Generate figures
    if args.gallery:
        gallery_actions = args.gallery_actions or args.actions or list(TARGET_ACTIONS)
        generate_gallery(
            model, raw_data, args.device, args.output_dir, args.dataset, args.seg,
            actions=gallery_actions, n_candidates=args.n_candidates,
        )
    else:
        figure_retargeting_examples(
            model, raw_data, args.device, args.output_dir, args.dataset, args.seg,
            actions=args.actions, source_person=args.source_person,
            target_person=args.target_person, n_candidates=args.n_candidates,
            selections=selections,
        )
        if not args.retargeting_only:
            figure_self_reconstruction(model, raw_data, args.device, args.output_dir, args.dataset, args.seg)
            figure_cross_identity(model, raw_data, args.device, args.output_dir, args.dataset, args.seg)
            figure_overlay(model, raw_data, args.device, args.output_dir, args.dataset, args.seg)

    print("\n=== All figures generated successfully ===")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
