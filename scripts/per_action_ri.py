#!/usr/bin/env python3
"""Per-action re-identification accuracy on raw NTU60 test set.

Loads raw-trained SGN_RI checkpoint, predicts identity for each test sample,
buckets accuracy by action class. Output: per_action_ri.json + summary.txt.

Run via SLURM (--partition=GPU). Single GPU, ~5 min.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets as DATASETS_CONFIG, ActionRecognitionDataset
from src.model.sgn import SGN

NTU60_ACTION_NAMES = {
    1: "drink water", 2: "eat meal", 3: "brush teeth", 4: "brush hair",
    5: "drop", 6: "pick up", 7: "throw", 8: "sit down", 9: "stand up",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "put on jacket", 15: "take off jacket", 16: "put on shoe",
    17: "take off shoe", 18: "put on glasses", 19: "take off glasses",
    20: "put on hat/cap", 21: "take off hat/cap", 22: "cheer up",
    23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jump up", 28: "phone call", 29: "play with phone",
    30: "type on keyboard", 31: "point to something", 32: "taking a selfie",
    33: "check time (watch)", 34: "rub two hands", 35: "nod head/bow",
    36: "shake head", 37: "wipe face", 38: "salute", 39: "put palms together",
    40: "cross hands in front", 41: "sneeze/cough", 42: "staggering",
    43: "falling down", 44: "headache", 45: "chest pain", 46: "back pain",
    47: "neck pain", 48: "nausea/vomiting", 49: "fan self",
}

ACTION_RE = re.compile(r"A(\d+)")


def parse_action_id(filename: str) -> int:
    m = ACTION_RE.search(filename)
    if not m:
        return -1
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output/downstream_ntu60_raw/ntu_sgn_ri_paired/model_best.pth.tar")
    parser.add_argument("--dataset", default="ntu")
    parser.add_argument("--setting", default="cv")
    parser.add_argument("--seg", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", default="output/analysis/per_action_ri")
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path or DATASETS_CONFIG[args.dataset]["path"]
    print(f"Loading data: {data_path}")
    import pickle
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    test_ds = ActionRecognitionDataset(
        data_dict, args.dataset, args.setting,
        split="test", task="ri", seg=args.seg,
        augment=False, drop_two_person_actions=False,
    )
    num_classes = test_ds.num_classes
    print(f"Test set: {len(test_ds)} samples, {num_classes} identities")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    ckpt_classes = state_dict["fc.weight"].shape[0] if "fc.weight" in state_dict else num_classes
    model = SGN(num_classes=ckpt_classes, dataset=args.dataset, seg=args.seg, bias=True)
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    print(f"Loaded SGN_RI: {args.checkpoint} ({ckpt_classes} identities)")

    # Inference loop without DataLoader collate (we need filenames for action)
    # Iterate test_ds.samples one by one is slow; do it batched with custom indices.
    correct_per_action: dict[int, int] = defaultdict(int)
    total_per_action: dict[int, int] = defaultdict(int)
    total_correct = 0
    total_seen = 0

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Walk in dataset order so sample index aligns with test_ds.samples
    sample_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            batch_n = y.size(0)
            preds_cpu = preds.cpu().tolist()
            y_cpu = y.cpu().tolist()
            for i in range(batch_n):
                fn, _ = test_ds.samples[sample_idx + i]
                action_id = parse_action_id(fn)
                total_per_action[action_id] += 1
                if preds_cpu[i] == y_cpu[i]:
                    correct_per_action[action_id] += 1
                    total_correct += 1
                total_seen += 1
            sample_idx += batch_n

    overall = total_correct / max(1, total_seen)
    print(f"Overall RI: {overall:.4f} ({total_correct}/{total_seen})")

    rows = []
    for action_id in sorted(total_per_action.keys()):
        n = total_per_action[action_id]
        c = correct_per_action[action_id]
        acc = c / n if n else 0.0
        rows.append({
            "action_id": action_id,
            "action_name": NTU60_ACTION_NAMES.get(action_id, f"action_{action_id}"),
            "correct": c,
            "total": n,
            "ri_accuracy": acc,
        })

    accs = np.array([r["ri_accuracy"] for r in rows])
    summary_stats = {
        "overall_ri": overall,
        "n_actions": len(rows),
        "min_ri": float(accs.min()) if accs.size else 0.0,
        "max_ri": float(accs.max()) if accs.size else 0.0,
        "mean_ri": float(accs.mean()) if accs.size else 0.0,
        "median_ri": float(np.median(accs)) if accs.size else 0.0,
        "std_ri": float(accs.std()) if accs.size else 0.0,
        "p25": float(np.percentile(accs, 25)) if accs.size else 0.0,
        "p75": float(np.percentile(accs, 75)) if accs.size else 0.0,
        "n_above_chance": int((accs > (1.0 / num_classes)).sum()),
        "n_total": len(rows),
        "chance_rate": 1.0 / num_classes,
    }
    print(json.dumps(summary_stats, indent=2))

    output = {"summary": summary_stats, "per_action": rows}
    json_path = out_dir / "per_action_ri.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {json_path}")

    # Sorted summary text
    rows_sorted = sorted(rows, key=lambda r: r["ri_accuracy"], reverse=True)
    summary_lines = ["Per-action RI on raw NTU60 (X-View) test set",
                     f"Overall: {overall*100:.2f}%  ({total_correct}/{total_seen})",
                     f"Min: {summary_stats['min_ri']*100:.2f}%  Max: {summary_stats['max_ri']*100:.2f}%",
                     f"Mean: {summary_stats['mean_ri']*100:.2f}%  Median: {summary_stats['median_ri']*100:.2f}%  Std: {summary_stats['std_ri']*100:.2f}%",
                     f"Chance: {summary_stats['chance_rate']*100:.2f}%  Above-chance actions: {summary_stats['n_above_chance']}/{summary_stats['n_total']}",
                     "",
                     f"{'AID':>4s} {'Action':<25s} {'RI%':>7s} {'n':>5s}"]
    for r in rows_sorted:
        summary_lines.append(f"{r['action_id']:4d} {r['action_name']:<25s} {r['ri_accuracy']*100:7.2f} {r['total']:5d}")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    print(f"Wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
