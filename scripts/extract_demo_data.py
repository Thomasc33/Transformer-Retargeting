#!/usr/bin/env python3
"""
Extract cherry-picked skeleton examples for the interactive GitHub Pages demo.
Loads raw + retargeted pickle files (no GPU needed) and exports to compact JSON.
"""
import pickle
import json
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.datasets import load_data, parse_file_name

# NTU-60 action labels (1-indexed, single-person only)
ACTION_NAMES = {
    1: "drink water", 2: "eat meal", 3: "brush teeth", 4: "brush hair",
    5: "drop", 6: "pick up", 7: "throw", 8: "sit down",
    9: "stand up", 10: "clapping", 11: "reading", 12: "writing",
    13: "tear up paper", 14: "put on jacket", 15: "take off jacket",
    16: "put on shoe", 17: "take off shoe", 18: "put on glasses",
    19: "take off glasses", 20: "put on hat/cap", 21: "take off hat/cap",
    22: "cheer up", 23: "hand waving", 24: "kicking something",
    25: "reach into pocket", 26: "hopping", 27: "jump up", 28: "phone call",
    29: "play with phone", 30: "type on keyboard", 31: "point to something",
    32: "taking a selfie", 33: "check time", 34: "rub two hands",
    35: "nod head/bow", 36: "shake head", 37: "wipe face", 38: "salute",
    39: "put palms together", 40: "cross hands in front", 41: "sneeze/cough",
    42: "staggering", 43: "falling down", 44: "headache", 45: "chest pain",
    46: "back pain", 47: "neck pain", 48: "nausea/vomiting", 49: "fan self"
}

# Cherry-pick visually distinctive actions
TARGET_ACTIONS = [1, 7, 10, 23, 24, 26, 27, 38, 43, 49]
# drink water, throw, clapping, hand waving, kicking, hopping, jump up, salute, falling down, fan self

def reshape_to_joints(seq):
    """(T, 75) -> (T, 25, 3) and round for compact JSON."""
    T = seq.shape[0]
    joints = seq.reshape(T, 25, 3)
    return np.round(joints, 4).tolist()

def main():
    print("Loading raw NTU data...")
    raw_data = load_data('ntu', 64)
    print(f"  Raw data: {len(raw_data)} sequences")

    # Index raw data by (person, action) for fast lookup
    by_person_action = {}
    for fname, seq in raw_data.items():
        info = parse_file_name(fname, 'ntu')
        key = (info['P'], info['A'])
        if key not in by_person_action:
            by_person_action[key] = (fname, seq)

    # Get all available persons and actions
    all_persons = sorted(set(k[0] for k in by_person_action))
    all_actions = sorted(set(k[1] for k in by_person_action))
    print(f"  {len(all_persons)} persons, {len(all_actions)} actions")

    # Load retargeted data (ours)
    retarget_path = ROOT / "output/retargeted_data/disentangled_tmr_stable_retargeted.pkl"
    print(f"Loading retargeted data from {retarget_path}...")
    with open(retarget_path, 'rb') as f:
        retargeted_data = pickle.load(f)
    print(f"  Retargeted data: {len(retargeted_data)} sequences")

    # Load DMR retargeted
    dmr_path = ROOT / "output/retargeted_data/dmr_ntu_cv_retargeted.pkl"
    print(f"Loading DMR retargeted data...")
    with open(dmr_path, 'rb') as f:
        dmr_data = pickle.load(f)

    # Load PMR retargeted
    pmr_path = ROOT / "output/retargeted_data/pmr_ntu_cv_retargeted.pkl"
    print(f"Loading PMR retargeted data...")
    with open(pmr_path, 'rb') as f:
        pmr_data = pickle.load(f)

    # Cherry-pick examples: for each target action, find a good source person
    # We want diverse persons and clear motion
    examples = []

    for action_id in TARGET_ACTIONS:
        action_name = ACTION_NAMES.get(action_id, f"action_{action_id}")
        print(f"\nLooking for action {action_id}: {action_name}")

        # Find filenames with this action in both raw and retargeted
        candidates = []
        for fname in retargeted_data:
            info = parse_file_name(fname, 'ntu')
            if info['A'] == action_id and fname in raw_data:
                # Check if also in DMR/PMR
                if fname in dmr_data and fname in pmr_data:
                    # Compute motion magnitude (prefer sequences with more movement)
                    seq = raw_data[fname]
                    velocity = np.diff(seq.reshape(-1, 25, 3), axis=0)
                    motion_mag = np.mean(np.abs(velocity))
                    candidates.append((fname, motion_mag))

        if not candidates:
            print(f"  No candidates found, skipping")
            continue

        # Pick the one with most motion (but not too extreme)
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Pick from top quartile but not the absolute max (avoid outliers)
        idx = min(len(candidates) // 4, 2)
        fname, motion_mag = candidates[idx]
        info = parse_file_name(fname, 'ntu')

        print(f"  Selected: {fname} (person={info['P']}, motion={motion_mag:.4f})")

        # Get the skeleton sequences
        source_raw = raw_data[fname]         # (64, 75)
        retargeted = retargeted_data[fname]  # (64, 75)
        dmr_retarg = dmr_data[fname]         # (64, 75)
        pmr_retarg = pmr_data[fname]         # (64, 75)

        example = {
            "id": fname,
            "action_id": action_id,
            "action_name": action_name,
            "person_id": info['P'],
            "source": reshape_to_joints(source_raw),
            "ours": reshape_to_joints(retargeted),
            "dmr": reshape_to_joints(dmr_retarg),
            "pmr": reshape_to_joints(pmr_retarg),
        }
        examples.append(example)

    # Save
    output_path = ROOT / "demo_data.json"
    print(f"\nSaving {len(examples)} examples to {output_path}...")

    output = {
        "examples": examples,
        "joint_names": [
            "base_spine", "mid_spine", "neck", "head",
            "l_shoulder", "l_elbow", "l_wrist", "l_hand",
            "r_shoulder", "r_elbow", "r_wrist", "r_hand",
            "l_hip", "l_knee", "l_ankle", "l_foot",
            "r_hip", "r_knee", "r_ankle", "r_foot",
            "spine", "l_thumb", "l_tip", "r_thumb", "r_tip"
        ],
        "bones": [
            [1, 0], [1, 20], [20, 2], [2, 3],
            [20, 4], [4, 5], [5, 6], [6, 7], [7, 22], [7, 21],
            [20, 8], [8, 9], [9, 10], [10, 11], [11, 24], [11, 23],
            [0, 12], [12, 13], [13, 14], [14, 15],
            [0, 16], [16, 17], [17, 18], [18, 19]
        ],
        "num_frames": 64,
        "num_joints": 25
    }

    with open(output_path, 'w') as f:
        json.dump(output, f)

    # Also report file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output size: {size_mb:.1f} MB")
    print("Done!")

if __name__ == "__main__":
    main()
