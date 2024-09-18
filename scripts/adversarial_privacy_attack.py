#!/usr/bin/env python3
"""
Adversarial privacy attack: train RI classifiers SPECIFICALLY on retargeted data.

This simulates an attacker who has access to retargeted training data and attempts
to learn identity patterns from the retargeted distribution itself.

Protocol:
1. Generate retargeted train + test data using the TMR model
2. Train SGN RI and MixFormer RI on retargeted training data
3. Test on retargeted test data
4. Also test ensemble (SGN + MixFormer average logits)
5. Compare against standard RI result (12.2%)

Key: labels are SOURCE identity (P1) — the identity the attacker tries to recover.

Must be submitted via SLURM.
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import (
    datasets as DATASETS_CONFIG,
    parse_file_name,
    sample_frames_fast,
    load_data,
    organize_data,
)
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel
from src.model.disentangled_tmr import create_disentangled_tmr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ntu")
    p.add_argument("--setting", default="cv")
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tmr_checkpoint",
                   default="output/disentangled_tmr_stable/checkpoint_stage3_best.pth")
    p.add_argument("--output_dir", default="output/analysis/adversarial_privacy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_retarget", action="store_true",
                   help="Skip retargeting if retargeted pickle already exists")
    return p.parse_args()


class RetargetedRIDataset(Dataset):
    """Dataset for RI task on retargeted data.

    Each sample: (skeleton (T, V*C), source_identity_label).
    Labels are SOURCE identity (what the attacker tries to recover).
    """

    def __init__(self, retargeted_data, dataset, setting, split, seg=64,
                 augment=False):
        self.seg = seg
        self.augment = augment
        self.samples = []  # (filename, label)

        # Organize into train/test
        train_data, test_data = organize_data(retargeted_data, setting, dataset)
        target_split = train_data if split == "train" else test_data

        # Collect unique actors for label mapping
        all_actors = set()
        for _, summary in target_split.items():
            pa_map = summary["pa_map"]
            for (person, action), filenames in pa_map.items():
                all_actors.add(person)
                for fname in filenames:
                    self.samples.append((fname, person))

        # Create contiguous label mapping
        unique_actors = sorted(all_actors)
        self.label_map = {actor: idx for idx, actor in enumerate(unique_actors)}
        self.num_classes = len(unique_actors)
        self.data = retargeted_data

    def __getitem__(self, index):
        fname, actor = self.samples[index]
        skeleton = self.data[fname]  # (T, V*C)

        # Normalize layout
        if skeleton.ndim == 2 and skeleton.shape[1] == 75:
            pass
        elif skeleton.ndim == 2 and skeleton.shape[0] == 75:
            skeleton = skeleton.T

        skeleton = sample_frames_fast(skeleton, self.seg)
        skeleton = torch.from_numpy(skeleton).float()
        label = self.label_map[actor]
        return skeleton, label

    def __len__(self):
        return len(self.samples)


def prepare_mixformer_input(x):
    """(B, T, V*C) -> (B, C, T, V, 1)"""
    B, T, VC = x.shape
    V = 25
    C = 3
    return x.view(B, T, V, C).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()


def retarget_dataset(raw_data, tmr_model, device, seg=64, seed=42):
    """Retarget all samples using TMR, preserving filenames."""
    np.random.seed(seed)
    all_filenames = list(raw_data.keys())
    retargeted = {}

    print(f"Retargeting {len(raw_data)} samples...")
    from tqdm import tqdm

    for fname, src_seq in tqdm(raw_data.items()):
        src_info = parse_file_name(fname, "ntu")
        src_p = src_info["P"]

        # Pick random different-identity target
        for _ in range(100):
            tgt_fname = np.random.choice(all_filenames)
            tgt_info = parse_file_name(tgt_fname, "ntu")
            if tgt_info["P"] != src_p:
                break

        tgt_seq = raw_data[tgt_fname]

        # Prepare inputs: (1, C, T, V, 1)
        x1 = _prepare(src_seq, seg).to(device)
        x2 = _prepare(tgt_seq, seg).to(device)

        with torch.no_grad():
            output, _, _ = tmr_model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

        first_frame = x2[:, :, 0:1, :, :]
        output_padded = torch.cat([first_frame, output], dim=2)

        out = output_padded.squeeze(-1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        retargeted[fname] = out.reshape(seg, -1)

    return retargeted


def _prepare(raw_seq, seg):
    """Raw numpy (frames, V*C) -> (1, C, T, V, 1) tensor."""
    seq = sample_frames_fast(raw_seq, seg)
    t = torch.from_numpy(seq).float()
    T, VC = t.shape
    t = t.reshape(T, 25, 3).permute(2, 0, 1)  # (C, T, V)
    return t.unsqueeze(0).unsqueeze(-1)  # (1, C, T, V, 1)


def load_tmr_model(checkpoint_path, device, dataset_name):
    """Load TMR model for retargeting."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None and isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    d_action = getattr(ckpt_args, "d_action", 768) if ckpt_args else 768
    d_identity = getattr(ckpt_args, "d_identity", 256) if ckpt_args else 256
    d_model = getattr(ckpt_args, "d_model", 320) if ckpt_args else 320
    num_class = DATASETS_CONFIG[dataset_name]["num_class"]

    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    model = create_disentangled_tmr(
        dataset=dataset_name, num_class=num_class, device=device,
        d_action=d_action, d_identity=d_identity, d_model=d_model,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args else True,
        identity_use_full_sequence=(getattr(ckpt_args, "identity_mode", "static") == "full_seq") if ckpt_args else False,
        tokenizer_type=tokenizer,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def train_model(model, train_loader, val_loader, device, epochs, lr, wd,
                name, is_mixformer=False):
    """Train a downstream model. Returns (model, best_val_acc)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        correct = 0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if is_mixformer:
                x = prepare_mixformer_input(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if is_mixformer:
                    x = prepare_mixformer_input(x)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_seen += y.size(0)

        val_acc = val_correct / max(1, val_seen)
        print(f"[{name}] Epoch {epoch+1}/{epochs} "
              f"TrainAcc={correct/max(1,seen):.4f} ValAcc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def evaluate_ensemble(sgn_model, mix_model, test_loader, device, num_classes):
    """Ensemble prediction: average logits from SGN + MixFormer."""
    correct = 0
    total = 0

    sgn_model.eval()
    mix_model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            sgn_logits = sgn_model(x)
            mix_logits = mix_model(prepare_mixformer_input(x))

            # Truncate to common number of classes
            nc = min(sgn_logits.shape[1], mix_logits.shape[1], num_classes)
            avg_logits = (sgn_logits[:, :nc] + mix_logits[:, :nc]) / 2
            preds = avg_logits.argmax(1)
            valid = y < nc
            correct += (preds[valid] == y[valid]).sum().item()
            total += valid.sum().item()

    return correct / max(1, total)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    retargeted_pkl = os.path.join(args.output_dir, "retargeted_for_attack.pkl")

    # Step 1: Generate retargeted data (or load if exists)
    if args.skip_retarget and os.path.exists(retargeted_pkl):
        print(f"Loading existing retargeted data from {retargeted_pkl}")
        with open(retargeted_pkl, "rb") as f:
            retargeted_data = pickle.load(f)
    else:
        print("Loading raw data...")
        raw_data = load_data(args.dataset, args.seg)
        print(f"Raw data: {len(raw_data)} samples")

        print("Loading TMR model...")
        tmr_model = load_tmr_model(args.tmr_checkpoint, device, args.dataset)

        retargeted_data = retarget_dataset(raw_data, tmr_model, device, args.seg, args.seed)

        print(f"Saving retargeted data to {retargeted_pkl}")
        with open(retargeted_pkl, "wb") as f:
            pickle.dump(retargeted_data, f)

        # Free TMR model memory
        del tmr_model
        torch.cuda.empty_cache()

    # Step 2: Build train/test datasets from retargeted data
    print("\nBuilding datasets...")
    train_ds = RetargetedRIDataset(retargeted_data, args.dataset, args.setting,
                                    split="train", seg=args.seg)
    test_ds = RetargetedRIDataset(retargeted_data, args.dataset, args.setting,
                                   split="test", seg=args.seg)
    num_classes = train_ds.num_classes
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, RI classes: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    results = {"num_classes": num_classes, "chance": 1.0 / num_classes}

    # Step 3: Train SGN RI on retargeted data
    print("\n" + "=" * 60)
    print("Training adversarial SGN RI on retargeted data")
    print("=" * 60)
    sgn_ri = SGN(num_classes=num_classes, dataset=args.dataset, seg=args.seg, bias=True)
    sgn_ri, sgn_acc = train_model(
        sgn_ri, train_loader, test_loader, device,
        args.epochs, args.lr, args.weight_decay,
        name="adv_sgn_ri",
    )
    results["sgn_ri_adversarial"] = sgn_acc
    print(f"\nAdversarial SGN RI accuracy: {sgn_acc:.4f} (chance: {1/num_classes:.4f})")

    # Save SGN checkpoint
    sgn_path = os.path.join(args.output_dir, "sgn_ri_adversarial.pth.tar")
    torch.save({"state_dict": sgn_ri.state_dict()}, sgn_path)

    # Step 4: Train MixFormer RI on retargeted data
    print("\n" + "=" * 60)
    print("Training adversarial MixFormer RI on retargeted data")
    print("=" * 60)
    mix_ri = MixFormerModel(
        num_class=num_classes, num_point=25, num_person=1,
        graph="src.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    )
    mix_ri, mix_acc = train_model(
        mix_ri, train_loader, test_loader, device,
        args.epochs, args.lr, args.weight_decay,
        name="adv_mix_ri", is_mixformer=True,
    )
    results["mixformer_ri_adversarial"] = mix_acc
    print(f"\nAdversarial MixFormer RI accuracy: {mix_acc:.4f}")

    # Save MixFormer checkpoint
    mix_path = os.path.join(args.output_dir, "mix_ri_adversarial.pth.tar")
    torch.save({"state_dict": mix_ri.state_dict()}, mix_path)

    # Step 5: Ensemble
    print("\n" + "=" * 60)
    print("Evaluating ensemble (SGN + MixFormer)")
    print("=" * 60)
    ensemble_acc = evaluate_ensemble(sgn_ri, mix_ri, test_loader, device, num_classes)
    results["ensemble_ri_adversarial"] = ensemble_acc
    print(f"Ensemble RI accuracy: {ensemble_acc:.4f}")

    # Reference values
    results["reference_sgn_ri_standard"] = 0.122  # From standard eval
    results["reference_mixf_ri_standard"] = 0.125

    # Save
    json_path = os.path.join(args.output_dir, "adversarial_privacy_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ADVERSARIAL PRIVACY ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"Chance level:              {1/num_classes:.4f}")
    print(f"Standard SGN RI:           {results['reference_sgn_ri_standard']:.4f}")
    print(f"Standard MixF RI:          {results['reference_mixf_ri_standard']:.4f}")
    print(f"Adversarial SGN RI:        {sgn_acc:.4f}")
    print(f"Adversarial MixFormer RI:  {mix_acc:.4f}")
    print(f"Adversarial Ensemble RI:   {ensemble_acc:.4f}")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
