#!/usr/bin/env python3
"""
Evaluate DisentangledTMR checkpoints (ablation suite or a single model).

Metrics:
- Action recognition (AR) accuracy via SGN and MixFormer on retargeted outputs.
- Re-identification (RI) accuracy via SGN and MixFormer on retargeted outputs.
- Physical plausibility (bone length, temporal smoothness, velocity) against ground-truth target motion.

Defaults cover all ablation runs; you can also pass explicit run directories or a single checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.losses.physical_plausibility import PhysicalPlausibilityLoss
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel


DEFAULT_RUNS = {
    "tmr_10k_final": "output/tmr_10k_final_50k",
    "ablate_full_baseline": "output/tmr_ablate_full_baseline_50k",
    "ablate_no_temporal_convs": "output/tmr_ablate_notconv_50k",
    "ablate_nolstm": "output/tmr_ablate_nolstm_50k",
    "ablate_identity_fullseq": "output/tmr_ablate_identity_fullseq_50k",
    "ablate_token_pos": "output/tmr_ablate_token_pos_50k",
    "ablate_token_dynamics": "output/tmr_ablate_token_dyn_50k",
    "ablate_token_dyn_codebook": "output/tmr_ablate_token_dyn_cb_50k",
    # Token fusion/codebook variants (new ablations)
    "ablate_token_pos_replace": "output/tmr_ablate_token_pos_replace_50k",
    "ablate_token_pos_codebook": "output/tmr_ablate_token_pos_cb_50k",
    "ablate_token_pos_codebook_replace": "output/tmr_ablate_token_pos_cb_replace_50k",
    "ablate_token_dyn_replace": "output/tmr_ablate_token_dyn_replace_50k",
    "ablate_token_dyn_codebook_cosine": "output/tmr_ablate_token_dyn_cb_cosine_50k",
    "ablate_token_dyn_codebook_replace": "output/tmr_ablate_token_dyn_cb_replace_50k",
    "ablate_token_dyn_codebook_replace_cosine": "output/tmr_ablate_token_dyn_cb_replace_cosine_50k",
    # Comprehensive run
    "comprehensive_new_losses": "output/tmr_comprehensive_new_losses",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DisentangledTMR ablations (AR/RI/physics)")
    p.add_argument("--runs", nargs="+", help="Run directories to evaluate (defaults to ablation suite)")
    p.add_argument("--checkpoint", help="Explicit checkpoint path (single-model eval)")
    p.add_argument("--stage", type=int, default=3, help="Preferred stage checkpoint to load")
    p.add_argument(
        "--checkpoint_preference",
        choices=["best", "latest"],
        default="best",
        help="When multiple checkpoints exist, prefer this filename pattern first",
    )
    p.add_argument("--data_path", default="data/ntu_cv_paired_10k.pt", help="Paired dataset .pt path")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--dataset", choices=["ntu", "ntu120", "etri", "ntu_small", "ntu_smoke"], default="ntu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_batches", type=int, default=-1, help="Limit batches for quick sweep (-1 = all)")
    p.add_argument("--output", default="logs/eval/tmr_ablation_eval.json")
    p.add_argument("--require_all", action="store_true", help="Error if any checkpoint is missing")
    p.add_argument("--no_allow_latest", action="store_true", help="Do not fall back to latest checkpoint when best is missing")
    p.add_argument("--include_baselines", action="store_true", help="Also evaluate downstream AR/RI on raw x1/x2/y2 sequences")
    p.add_argument(
        "--drop_invalid_labels",
        action="store_true",
        help="Drop samples whose action/identity labels fall outside downstream head sizes",
    )
    p.add_argument(
        "--sampling_mode",
        choices=["random_action", "constant_action"],
        default="random_action",
        help="Target skeleton input: random_action uses x2 (P2,A2); constant_action uses y2 (P2,A1).",
    )
    p.add_argument(
        "--autoregressive_full_context",
        action="store_true",
        help="Use full-prefix autoregressive decoding (slower, closer to teacher-forcing behavior)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Post-hoc blending: output = beta*retargeted + (1-beta)*source. None=1.0 (full retarget).",
    )

    # Downstream AR/RI checkpoints
    p.add_argument("--sgn_ar_ckpt", default="output/ntu_sgn_ar_paired/model_best.pth.tar")
    p.add_argument("--sgn_ri_ckpt", default="output/ntu_sgn_ri_paired/model_best.pth.tar")
    p.add_argument("--mix_ar_ckpt", default="output/ntu_mixformer_ar_paired/model_best.pth.tar")
    p.add_argument("--mix_ri_ckpt", default="output/ntu_mixformer_ri_paired/model_best.pth.tar")

    # Physical weights (same as training defaults)
    p.add_argument("--weight_bone_length", type=float, default=0.5)
    p.add_argument("--weight_temporal_smoothness", type=float, default=0.3)
    p.add_argument("--weight_velocity", type=float, default=0.2)
    return p.parse_args()


def dataset_meta(dataset: str) -> Tuple[int, int]:
    """
    Returns (num_actions, num_identities) for the dataset label space used in
    this repository. Note that NTU two-person actions are removed, leaving 49
    action classes.
    """
    if dataset in ["ntu", "ntu_small", "ntu_smoke"]:
        return 49, 40
    if dataset == "ntu120":
        return 120, 106
    if dataset == "etri":
        return 55, 100
    raise ValueError(f"Unknown dataset: {dataset}")


def load_state(path: str, device: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    return {k.replace("module.", ""): v for k, v in state.items()}


def infer_num_classes_from_state(state: Dict[str, torch.Tensor], fallback: int) -> int:
    """Infer classifier head size (num_classes) from a state dict."""
    for key in ["fc.weight", "module.fc.weight"]:
        if key in state:
            return state[key].shape[0]
    return fallback


def require_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_sgn(num_classes: int, checkpoint: str, device: str) -> Tuple[SGN, int]:
    require_file(checkpoint, "SGN checkpoint")
    state = load_state(checkpoint, device)
    inferred_classes = infer_num_classes_from_state(state, num_classes)
    if inferred_classes != num_classes:
        print(f"⚠ SGN checkpoint head={inferred_classes} differs from expected {num_classes}; using {inferred_classes}.")
    model = SGN(num_classes=inferred_classes, dataset="ntu", seg=64, bias=True).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, inferred_classes


def load_mixformer(num_classes: int, checkpoint: str, device: str) -> Tuple[MixFormerModel, int]:
    require_file(checkpoint, "MixFormer checkpoint")
    state = load_state(checkpoint, device)
    inferred_classes = infer_num_classes_from_state(state, num_classes)
    if inferred_classes != num_classes:
        print(f"⚠ MixFormer checkpoint head={inferred_classes} differs from expected {num_classes}; using {inferred_classes}.")
    model = MixFormerModel(
        num_class=inferred_classes,
        num_point=25,
        num_person=1,
        graph="src.graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, inferred_classes


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    """Count how many labels fall within the top-k predictions."""
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).sum().item()


def _label_valid_mask(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (labels >= 0) & (labels < num_classes)


def get_loader(data_path: str, split: str, batch_size: int, num_workers: int):
    data = torch.load(data_path, weights_only=False)
    dataset = data["test"] if split == "test" else data["train"]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def resolve_checkpoint(run_dir: Path, stage: int, allow_latest: bool, checkpoint_preference: str) -> Optional[Path]:
    if checkpoint_preference not in {"best", "latest"}:
        raise ValueError(f"Unknown checkpoint_preference: {checkpoint_preference}")
    order = ("latest", "best") if checkpoint_preference == "latest" else ("best", "latest")

    candidates = [run_dir / f"checkpoint_stage{stage}_{suffix}.pth" for suffix in order]
    if allow_latest:
        for s in [3, 2, 1]:
            for suffix in order:
                candidates.append(run_dir / f"checkpoint_stage{s}_{suffix}.pth")
    for c in candidates:
        if c.exists():
            return c
    return None


def build_model_from_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", None)

    if ckpt_args is not None and not isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = argparse.Namespace(**vars(ckpt_args)) if hasattr(ckpt_args, "__dict__") else argparse.Namespace(**ckpt_args)

    dataset = getattr(ckpt_args, "dataset", "ntu") if ckpt_args else "ntu"
    num_actions, _ = dataset_meta(dataset)

    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    model = create_disentangled_tmr(
        dataset=dataset,
        num_class=num_actions,
        device=device,
        d_action=getattr(ckpt_args, "d_action", 512) if ckpt_args else 512,
        d_identity=getattr(ckpt_args, "d_identity", 128) if ckpt_args else 128,
        d_model=getattr(ckpt_args, "d_model", 320) if ckpt_args else 320,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args else 6,
        use_pretrained_action=True,
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

    state = ckpt.get("model_state_dict", ckpt)
    
    # Filter out adjacency matrices (A) if they mismatch, to allow loading checkpoints 
    # trained with broken graph (Identity) into fixed model (Spatial Graph)
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                if k.endswith('.A'):
                    print(f"  ⚠ Skipping {k} due to shape mismatch (likely fixing graph architecture): {v.shape} vs {model_state[k].shape}")
                    continue
                else:
                     print(f"  ⚠ Shape mismatch for {k}: {v.shape} vs {model_state[k].shape}")
            filtered_state[k] = v
        else:
            filtered_state[k] = v
            
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    # Log checkpoint stage/epoch for clarity (decoder weights are inside model_state_dict)
    stage_loaded = ckpt.get("stage", "unknown")
    epoch_loaded = ckpt.get("epoch", "unknown")
    print(f"  ✓ Loaded checkpoint (stage {stage_loaded}, epoch {epoch_loaded}) with full model weights.")
    if isinstance(stage_loaded, int) and stage_loaded < 3:
        print("  ⚠ Using a pre-finetune checkpoint (decoder may be less trained than stage 3).")

    return model, ckpt_args, ckpt


@torch.no_grad()
def evaluate_run(
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    sgn_ar: SGN,
    sgn_ri: SGN,
    mix_ar: MixFormerModel,
    mix_ri: MixFormerModel,
    physical_loss: PhysicalPlausibilityLoss,
    physical_weights: Dict[str, float],
    max_batches: int,
    sampling_mode: str,
    drop_invalid_labels: bool,
    autoregressive_full_context: bool,
    beta: float = None,
):
    # Pre-search for Actor 8 sample if needed
    actor8_sample = None
    if sampling_mode == "constant_actor":
        print("Searching for an Actor 8 sample in the loader...")
        for batch in loader:
            x1, x2, _, _, actors, _ = batch
            # actors is (B, 2), values are 1-based actor IDs
            # Check source actors (col 0)
            mask_src = (actors[:, 0] == 8)
            if mask_src.any():
                actor8_sample = x1[mask_src][0:1].to(device) # (1, C, T, V, M)
                print("✓ Found Actor 8 sample (from source).")
                break
            # Check target actors (col 1)
            mask_tgt = (actors[:, 1] == 8)
            if mask_tgt.any():
                actor8_sample = x2[mask_tgt][0:1].to(device)
                print("✓ Found Actor 8 sample (from target).")
                break
        
        if actor8_sample is None:
            print("⚠ Warning: Actor 8 (ID=8) not found in dataset! Falling back to random_action mode.")
            sampling_mode = "random_action"

    stats = {
        "ar_sgn_correct1": 0,
        "ar_sgn_correct5": 0,
        "ri_src_sgn_correct1": 0,
        "ri_tgt_sgn_correct1": 0,
        "ar_mix_correct1": 0,
        "ar_mix_correct5": 0,
        "ri_src_mix_correct1": 0,
        "ri_tgt_mix_correct1": 0,
        "total": 0,
        "physical": {"bone_length": 0.0, "temporal_smoothness": 0.0, "velocity": 0.0, "total_physical": 0.0},
        "batches": 0,
        "dropped": 0,
        "invalid_action_labels": 0,
        "invalid_identity_labels": 0,
    }

    for batch_idx, batch in enumerate(loader):
        x1, x2, y1, y2, actors, actions = batch
        x1 = x1.to(device).unsqueeze(-1)
        x2 = x2.to(device).unsqueeze(-1)
        y2 = y2.to(device).unsqueeze(-1)
        actions = actions.to(device)
        actors = actors.to(device)

        action_labels = (actions[:, 0] - 1).long()
        source_id_labels = (actors[:, 0] - 1).long()
        target_id_labels = (actors[:, 1] - 1).long()

        num_actions = sgn_ar.fc.out_features
        num_ids = sgn_ri.fc.out_features
        valid_actions = _label_valid_mask(action_labels, num_actions)
        valid_src_ids = _label_valid_mask(source_id_labels, num_ids)
        valid_tgt_ids = _label_valid_mask(target_id_labels, num_ids)
        valid = valid_actions & valid_src_ids & valid_tgt_ids
        stats["invalid_action_labels"] += (~valid_actions).sum().item()
        stats["invalid_identity_labels"] += (~(valid_src_ids & valid_tgt_ids)).sum().item()
        if drop_invalid_labels and not valid.all():
            stats["dropped"] += (~valid).sum().item()
            if valid.sum().item() == 0:
                if 0 < max_batches <= (batch_idx + 1):
                    break
                continue
            x1 = x1[valid]
            x2 = x2[valid]
            y2 = y2[valid]
            action_labels = action_labels[valid]
            source_id_labels = source_id_labels[valid]
            target_id_labels = target_id_labels[valid]

        if sampling_mode == "constant_actor":
            # Force target to Actor 8
            target_skeleton = actor8_sample.expand(x1.shape[0], -1, -1, -1, -1)
            # Update ground truth identity labels to Actor 8 (index 7)
            target_id_labels[:] = 7
        elif sampling_mode == "constant_action":
            target_skeleton = y2
        else:
            target_skeleton = x2

        # Generate without access to target motion (matches inference-time setup).
        output, _, _ = model(
            x1,
            target_skeleton,
            target_motion=None,
            teacher_forcing_ratio=0.0,
            autoregressive_full_context=autoregressive_full_context,
        )

        # Post-hoc beta blending: output = beta*retargeted + (1-beta)*source
        if beta is not None and beta < 1.0:
            x1_sliced = x1[:, :, 1:, :, :]
            output = beta * output + (1.0 - beta) * x1_sliced

        first_frame = target_skeleton[:, :, 0:1, :, :]
        output_full = torch.cat([first_frame, output], dim=2)

        # SGN expects (B, T, V*C)
        B, C, T, V, _ = output_full.size()
        sgn_in = output_full.squeeze(-1).permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        ar_logits_sgn = sgn_ar(sgn_in)
        ri_logits_sgn = sgn_ri(sgn_in)

        # MixFormer expects (B, C, T, V, 1)
        # Reshape to 5D tensor
        if output_full.dim() == 4:
            output_full = output_full.unsqueeze(-1)
        
        ar_logits_mix = mix_ar(output_full)
        ri_logits_mix = mix_ri(output_full)

        stats["ar_sgn_correct1"] += (ar_logits_sgn.argmax(dim=1) == action_labels).sum().item()
        stats["ar_sgn_correct5"] += topk_correct(ar_logits_sgn, action_labels, 5)
        stats["ri_src_sgn_correct1"] += (ri_logits_sgn.argmax(dim=1) == source_id_labels).sum().item()
        stats["ri_tgt_sgn_correct1"] += (ri_logits_sgn.argmax(dim=1) == target_id_labels).sum().item()
        stats["ar_mix_correct1"] += (ar_logits_mix.argmax(dim=1) == action_labels).sum().item()
        stats["ar_mix_correct5"] += topk_correct(ar_logits_mix, action_labels, 5)
        stats["ri_src_mix_correct1"] += (ri_logits_mix.argmax(dim=1) == source_id_labels).sum().item()
        stats["ri_tgt_mix_correct1"] += (ri_logits_mix.argmax(dim=1) == target_id_labels).sum().item()
        stats["total"] += B

        _, phys = physical_loss(output_full, y2, physical_weights)
        for k in stats["physical"]:
            stats["physical"][k] += phys[k]
        stats["batches"] += 1

        if 0 < max_batches <= (batch_idx + 1):
            break

    total = max(1, stats["total"])
    batches = max(1, stats["batches"])
    return {
        "name": name,
        "ar_acc1_sgn": stats["ar_sgn_correct1"] / total,
        "ar_acc5_sgn": stats["ar_sgn_correct5"] / total,
        "ri_src_sgn": stats["ri_src_sgn_correct1"] / total,
        "ri_tgt_sgn": stats["ri_tgt_sgn_correct1"] / total,
        "ar_acc1_mix": stats["ar_mix_correct1"] / total,
        "ar_acc5_mix": stats["ar_mix_correct5"] / total,
        "ri_src_mix": stats["ri_src_mix_correct1"] / total,
        "ri_tgt_mix": stats["ri_tgt_mix_correct1"] / total,
        "physical": {k: v / batches for k, v in stats["physical"].items()},
        "samples": total,
        "batches": batches,
        "dropped": stats["dropped"],
        "invalid_action_labels": stats["invalid_action_labels"],
        "invalid_identity_labels": stats["invalid_identity_labels"],
    }


@torch.no_grad()
def evaluate_baseline(
    which: str,
    loader: DataLoader,
    device: str,
    sgn_ar: SGN,
    sgn_ri: SGN,
    mix_ar: MixFormerModel,
    mix_ri: MixFormerModel,
    max_batches: int,
    drop_invalid_labels: bool,
):
    """
    Evaluate downstream AR/RI on raw sequences from the paired dataset.

    which:
      - x1: source actor P1, action A1 (labels: action=a1, identity=p1)
      - x2: target actor P2, action A2 (labels: action=a2, identity=p2)
      - y2: target actor P2, action A1 (labels: action=a1, identity=p2)
    """
    if which not in {"x1", "x2", "y2"}:
        raise ValueError(f"Unknown baseline source: {which}")

    stats = {
        "ar_sgn_correct1": 0,
        "ar_sgn_correct5": 0,
        "ri_sgn_correct1": 0,
        "ri_sgn_correct5": 0,
        "ar_mix_correct1": 0,
        "ar_mix_correct5": 0,
        "ri_mix_correct1": 0,
        "ri_mix_correct5": 0,
        "total": 0,
        "batches": 0,
        "dropped": 0,
        "invalid_action_labels": 0,
        "invalid_identity_labels": 0,
    }

    for batch_idx, batch in enumerate(loader):
        x1, x2, y1, y2, actors, actions = batch
        x1 = x1.to(device).unsqueeze(-1)
        x2 = x2.to(device).unsqueeze(-1)
        y2 = y2.to(device).unsqueeze(-1)
        actions = actions.to(device)
        actors = actors.to(device)

        if which == "x1":
            seq = x1
            action_labels = (actions[:, 0] - 1).long()
            identity_labels = (actors[:, 0] - 1).long()
        elif which == "x2":
            seq = x2
            action_labels = (actions[:, 1] - 1).long()
            identity_labels = (actors[:, 1] - 1).long()
        else:  # y2
            seq = y2
            action_labels = (actions[:, 0] - 1).long()
            identity_labels = (actors[:, 1] - 1).long()

        num_actions = sgn_ar.fc.out_features
        num_ids = sgn_ri.fc.out_features
        valid_actions = _label_valid_mask(action_labels, num_actions)
        valid_ids = _label_valid_mask(identity_labels, num_ids)
        valid = valid_actions & valid_ids
        stats["invalid_action_labels"] += (~valid_actions).sum().item()
        stats["invalid_identity_labels"] += (~valid_ids).sum().item()
        if drop_invalid_labels and not valid.all():
            stats["dropped"] += (~valid).sum().item()
            if valid.sum().item() == 0:
                if 0 < max_batches <= (batch_idx + 1):
                    break
                continue
            seq = seq[valid]
            action_labels = action_labels[valid]
            identity_labels = identity_labels[valid]

        # SGN expects (B, T, V*C)
        B, C, T, V, _ = seq.size()
        sgn_in = seq.squeeze(-1).permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        ar_logits_sgn = sgn_ar(sgn_in)
        ri_logits_sgn = sgn_ri(sgn_in)

        # MixFormer expects (B, C, T, V, 1)
        ar_logits_mix = mix_ar(seq)
        ri_logits_mix = mix_ri(seq)

        stats["ar_sgn_correct1"] += (ar_logits_sgn.argmax(dim=1) == action_labels).sum().item()
        stats["ar_sgn_correct5"] += topk_correct(ar_logits_sgn, action_labels, 5)
        stats["ri_sgn_correct1"] += (ri_logits_sgn.argmax(dim=1) == identity_labels).sum().item()
        stats["ri_sgn_correct5"] += topk_correct(ri_logits_sgn, identity_labels, 5)
        stats["ar_mix_correct1"] += (ar_logits_mix.argmax(dim=1) == action_labels).sum().item()
        stats["ar_mix_correct5"] += topk_correct(ar_logits_mix, action_labels, 5)
        stats["ri_mix_correct1"] += (ri_logits_mix.argmax(dim=1) == identity_labels).sum().item()
        stats["ri_mix_correct5"] += topk_correct(ri_logits_mix, identity_labels, 5)
        stats["total"] += B
        stats["batches"] += 1

        if 0 < max_batches <= (batch_idx + 1):
            break

    total = max(1, stats["total"])
    batches = max(1, stats["batches"])
    return {
        "name": f"baseline_{which}",
        "ar_acc1_sgn": stats["ar_sgn_correct1"] / total,
        "ar_acc5_sgn": stats["ar_sgn_correct5"] / total,
        "ar_acc_sgn": stats["ar_sgn_correct1"] / total,
        "ri_acc1_sgn": stats["ri_sgn_correct1"] / total,
        "ri_acc5_sgn": stats["ri_sgn_correct5"] / total,
        "ri_acc_sgn": stats["ri_sgn_correct1"] / total,
        "ar_acc1_mix": stats["ar_mix_correct1"] / total,
        "ar_acc5_mix": stats["ar_mix_correct5"] / total,
        "ar_acc_mix": stats["ar_mix_correct1"] / total,
        "ri_acc1_mix": stats["ri_mix_correct1"] / total,
        "ri_acc5_mix": stats["ri_mix_correct5"] / total,
        "ri_acc_mix": stats["ri_mix_correct1"] / total,
        "samples": total,
        "batches": batches,
        "dropped": stats["dropped"],
        "invalid_action_labels": stats["invalid_action_labels"],
        "invalid_identity_labels": stats["invalid_identity_labels"],
    }


def main():
    args = parse_args()
    allow_latest = not args.no_allow_latest
    device = args.device if torch.cuda.is_available() else "cpu"

    # Downstream checkpoints must exist
    for path, label in [
        (args.sgn_ar_ckpt, "SGN AR checkpoint"),
        (args.sgn_ri_ckpt, "SGN RI checkpoint"),
        (args.mix_ar_ckpt, "MixFormer AR checkpoint"),
        (args.mix_ri_ckpt, "MixFormer RI checkpoint"),
    ]:
        require_file(path, label)

    # FIXED: If explicit checkpoint is provided, only evaluate that one.
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        run_dirs = {ckpt_path.stem: str(ckpt_path.parent)}
        print(f"Evaluating single explicit checkpoint: {ckpt_path}")
    else:
        run_dirs: Dict[str, str] = DEFAULT_RUNS if args.runs is None else {Path(r).name: r for r in args.runs}

    loader = get_loader(args.data_path, args.split, args.batch_size, args.num_workers)

    dataset_guess = args.dataset
    num_actions, num_ids = dataset_meta(dataset_guess)
    sgn_ar, sgn_ar_classes = load_sgn(num_actions, args.sgn_ar_ckpt, device)
    sgn_ri, sgn_ri_classes = load_sgn(num_ids, args.sgn_ri_ckpt, device)
    mix_ar, mix_ar_classes = load_mixformer(num_actions, args.mix_ar_ckpt, device)
    mix_ri, mix_ri_classes = load_mixformer(num_ids, args.mix_ri_ckpt, device)

    # Fail fast if checkpoint heads do not match expected label spaces
    if sgn_ar_classes != num_actions or mix_ar_classes != num_actions:
        raise ValueError(
            f"Action head mismatch: expected {num_actions} classes "
            f"(dataset {dataset_guess}), got SGN={sgn_ar_classes}, MixFormer={mix_ar_classes}. "
            "Retrain downstream AR models with the correct label space."
        )
    if sgn_ri_classes != num_ids or mix_ri_classes != num_ids:
        raise ValueError(
            f"Identity head mismatch: expected {num_ids} identities "
            f"(dataset {dataset_guess}), got SGN={sgn_ri_classes}, MixFormer={mix_ri_classes}. "
            "Retrain downstream RI models with the correct label space."
        )

    physical_loss = PhysicalPlausibilityLoss(dataset=dataset_guess, device=device)
    physical_weights = {
        "bone_length": args.weight_bone_length,
        "temporal_smoothness": args.weight_temporal_smoothness,
        "velocity": args.weight_velocity,
    }

    results = []
    baselines = []
    missing_runs: List[str] = []

    if args.include_baselines:
        print("\n=== Baselines (raw sequences) ===")
        for which in ["x1", "x2", "y2"]:
            b = evaluate_baseline(
                which=which,
                loader=loader,
                device=device,
                sgn_ar=sgn_ar,
                sgn_ri=sgn_ri,
                mix_ar=mix_ar,
                mix_ri=mix_ri,
                max_batches=args.max_batches,
                drop_invalid_labels=args.drop_invalid_labels,
            )
            baselines.append(b)
            print(
                f"{b['name']}: "
                f"AR(SGN)={b['ar_acc1_sgn']:.4f}, RI(SGN)={b['ri_acc1_sgn']:.4f}, "
                f"AR(Mix)={b['ar_acc1_mix']:.4f}, RI(Mix)={b['ri_acc1_mix']:.4f}"
            )

    for name, run_dir_str in run_dirs.items():
        run_dir = Path(run_dir_str)
        ckpt_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else resolve_checkpoint(run_dir, args.stage, allow_latest, args.checkpoint_preference)
        )

        if ckpt_path is None or not ckpt_path.exists():
            msg = f"No checkpoint found for {name} in {run_dir}"
            if args.require_all:
                raise FileNotFoundError(msg)
            print(f"⚠ {msg}; skipping.")
            missing_runs.append(name)
            continue

        print(f"\n=== Evaluating {name} ===")
        print(f"Checkpoint: {ckpt_path}")

        model, ckpt_args, ckpt_raw = build_model_from_checkpoint(ckpt_path, device)

        run_dataset = getattr(ckpt_args, "dataset", dataset_guess) if ckpt_args else dataset_guess
        if run_dataset != dataset_guess:
            dataset_guess = run_dataset
            num_actions, num_ids = dataset_meta(dataset_guess)
            sgn_ar, sgn_ar_classes = load_sgn(num_actions, args.sgn_ar_ckpt, device)
            sgn_ri, sgn_ri_classes = load_sgn(num_ids, args.sgn_ri_ckpt, device)
            mix_ar, mix_ar_classes = load_mixformer(num_actions, args.mix_ar_ckpt, device)
            mix_ri, mix_ri_classes = load_mixformer(num_ids, args.mix_ri_ckpt, device)
            if sgn_ar_classes != num_actions or mix_ar_classes != num_actions:
                raise ValueError(
                    f"Action head mismatch after switching dataset to {dataset_guess}: "
                    f"expected {num_actions}, got SGN={sgn_ar_classes}, MixFormer={mix_ar_classes}"
                )
            if sgn_ri_classes != num_ids or mix_ri_classes != num_ids:
                raise ValueError(
                    f"Identity head mismatch after switching dataset to {dataset_guess}: "
                    f"expected {num_ids}, got SGN={sgn_ri_classes}, MixFormer={mix_ri_classes}"
                )
            physical_loss = PhysicalPlausibilityLoss(dataset=dataset_guess, device=device)

        run_result = evaluate_run(
            name=name,
            model=model,
            loader=loader,
            device=device,
            sgn_ar=sgn_ar,
            sgn_ri=sgn_ri,
            mix_ar=mix_ar,
            mix_ri=mix_ri,
            physical_loss=physical_loss,
            physical_weights=physical_weights,
            max_batches=args.max_batches,
            sampling_mode=args.sampling_mode,
            drop_invalid_labels=args.drop_invalid_labels,
            autoregressive_full_context=args.autoregressive_full_context,
            beta=args.beta,
        )

        # Meta
        run_result["meta"] = {
            "checkpoint": str(ckpt_path),
            "epoch": ckpt_raw.get("epoch"),
            "stage": ckpt_raw.get("stage"),
            "dataset": getattr(ckpt_args, "dataset", "ntu") if ckpt_args else "ntu",
            "sampling_mode": args.sampling_mode,
            "autoregressive_full_context": args.autoregressive_full_context,
            "checkpoint_preference": args.checkpoint_preference,
        }
        results.append(run_result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "baselines": baselines, "missing": missing_runs}, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for r in results:
        print(
            f"{r['name']}: "
            f"AR(SGN)={r['ar_acc1_sgn']:.4f}, "
            f"RI_src(SGN)={r['ri_src_sgn']:.4f}, RI_tgt(SGN)={r['ri_tgt_sgn']:.4f}, "
            f"AR(Mix)={r['ar_acc1_mix']:.4f}, "
            f"RI_src(Mix)={r['ri_src_mix']:.4f}, RI_tgt(Mix)={r['ri_tgt_mix']:.4f}, "
            f"Phys={r['physical']['total_physical']:.4f}"
        )
    if missing_runs:
        print(f"\nSkipped (missing checkpoints): {', '.join(missing_runs)}")

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
