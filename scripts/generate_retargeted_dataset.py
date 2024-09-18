
#!/usr/bin/env python3
"""
Generate a retargeted dataset by applying TMR to the entire dataset.
Each sample is retargeted to a random target identity.
"""

import argparse
import pickle
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.data.datasets import datasets, parse_file_name, sample_frames_fast, load_data

def load_model(checkpoint_path, device, dataset_name):
    """Load the TMR model."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None:
        # Convert dict to Namespace if needed
        if isinstance(ckpt_args, dict):
            ckpt_args = argparse.Namespace(**ckpt_args)
    
    # Defaults
    d_action = 768
    d_identity = 256
    d_model = 320
    
    if ckpt_args:
        d_action = getattr(ckpt_args, "d_action", d_action)
        d_identity = getattr(ckpt_args, "d_identity", d_identity)
        d_model = getattr(ckpt_args, "d_model", d_model)
    
    # Get num_class from dataset config
    num_class = datasets[dataset_name]['num_class']
    
    # Handle tokenizer argument
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
        # Add other args as needed, defaulting to standard values
    )
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  Note: {len(missing)} missing, {len(unexpected)} unexpected keys (expected for cross-dataset transfer)")
        for k in missing:
            print(f"    missing: {k}")
        for k in unexpected:
            print(f"    unexpected: {k}")
        
    model.eval()
    return model

def prepare_input(raw_seq, seg=64):
    """
    Prepare raw sequence for TMR input.
    Input: (Frames, V*C)
    Output: (1, C, T, V, M) tensor
    """
    # Sample/Pad to fixed length
    seq = sample_frames_fast(raw_seq, seg) # (T, V*C)
    
    # Reshape
    # (T, V*C) -> (T, V, C) -> (C, T, V)
    tensor = torch.from_numpy(seq).float()
    T, VC = tensor.shape
    V = 25
    C = 3
    tensor = tensor.reshape(T, V, C).permute(2, 0, 1) # (C, T, V)
    
    # Add Batch and Person dims: (B, C, T, V, M)
    tensor = tensor.unsqueeze(0).unsqueeze(-1) # (1, C, T, V, 1)
    return tensor

def prepare_input_np(raw_seq: np.ndarray, seg: int = 64) -> np.ndarray:
    """Prepare raw sequence without creating a tensor (for batching later).
    Returns: (C, T, V) numpy array.
    """
    seq = sample_frames_fast(raw_seq, seg)  # (T, V*C)
    T, VC = seq.shape
    V, C = 25, 3
    return seq.reshape(T, V, C).transpose(2, 0, 1)  # (C, T, V)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to TMR checkpoint")
    parser.add_argument("--dataset", default="ntu", help="Dataset name")
    parser.add_argument("--output_path", required=True, help="Path to save retargeted pickle")
    parser.add_argument("--seg", type=int, default=64, help="Sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_identity", action="store_true",
                        help="Zero out identity features at inference (action-only decoding)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Blend factor: 1.0=full retarget, <1.0=blend with source")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (higher = faster, more VRAM)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load Data — use pickle.load (same as train_downstream_models /
    #    cross_evaluate_downstream) so that sample_frames_fast in
    #    prepare_input_np handles frame selection identically to the
    #    downstream evaluation pipeline.
    print(f"Loading data for {args.dataset}...")
    data_path = datasets[args.dataset]['path']
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)

    # Filter two-person actions for NTU datasets (same as load_data)
    if args.dataset in ('ntu', 'ntu120'):
        remove = {50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                  106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119, 120}
        raw_data = {k: v for k, v in raw_data.items()
                    if parse_file_name(k, args.dataset)['A'] not in remove}

    # Slice to single-person columns if needed
    max_actors = datasets[args.dataset]['max_actors']
    if max_actors == 1:
        n_cols = datasets[args.dataset]['joints'] * datasets[args.dataset]['channels']
        raw_data = {k: v[:, :n_cols] for k, v in raw_data.items()}

    print(f"  {len(raw_data)} samples loaded")

    # 2. Load Model
    model = load_model(args.checkpoint, args.device, args.dataset)

    # 3. Pre-select targets (different identity for each source)
    all_filenames = list(raw_data.keys())
    identity_map: dict[str, str] = {}
    for fn in all_filenames:
        identity_map[fn] = parse_file_name(fn, args.dataset)['P']

    src_fnames = list(raw_data.keys())
    tgt_fnames: list[str] = []
    for fn in src_fnames:
        src_p = identity_map[fn]
        while True:
            tgt = np.random.choice(all_filenames)
            if identity_map[tgt] != src_p:
                break
        tgt_fnames.append(tgt)

    # 4. Prepare all inputs as one contiguous numpy array (N, C, T, V)
    #    -- pre-stack once (avoids per-batch np.stack copies) and pin host
    #    memory so H2D transfers can run non-blocking.
    print("Preparing inputs...")
    n = len(src_fnames)
    V, C = 25, 3
    T = args.seg
    src_all_np = np.empty((n, C, T, V), dtype=np.float32)
    tgt_all_np = np.empty((n, C, T, V), dtype=np.float32)
    for i, fn in enumerate(src_fnames):
        src_all_np[i] = prepare_input_np(raw_data[fn], args.seg)
    for i, fn in enumerate(tgt_fnames):
        tgt_all_np[i] = prepare_input_np(raw_data[fn], args.seg)

    # Wrap once; add singleton person dim to match (N, C, T, V, 1).
    src_host = torch.from_numpy(src_all_np).unsqueeze(-1)
    tgt_host = torch.from_numpy(tgt_all_np).unsqueeze(-1)
    use_pinned = args.device == "cuda" and torch.cuda.is_available()
    if use_pinned:
        src_host = src_host.pin_memory()
        tgt_host = tgt_host.pin_memory()

    # Preallocate CPU output buffer for the final (N, T, V*C) dataset so we
    # avoid touching a Python dict inside the inference loop and avoid
    # fragmented allocations.
    out_all_np = np.empty((n, T, V * C), dtype=np.float32)

    # 5. Batched inference
    bs = args.batch_size
    num_batches = (n + bs - 1) // bs
    print(f"Retargeting {n} samples in {num_batches} batches (bs={bs})...")

    with torch.inference_mode():
        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * bs
            end = min(start + bs, n)

            x1 = src_host[start:end].to(args.device, non_blocking=use_pinned)
            x2 = tgt_host[start:end].to(args.device, non_blocking=use_pinned)

            if getattr(args, 'zero_identity', False):
                action_features = model.action_encoder(x1)
                identity_features = model.identity_encoder(x2)
                identity_features = torch.zeros_like(identity_features)
                output = model.decoder(
                    action_features, identity_features,
                    target_skeleton=x2, target_motion=None,
                    teacher_forcing_ratio=0.0,
                    autoregressive_full_context=True,
                )
            else:
                output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

            # Soft retargeting: blend with source
            if args.beta < 1.0:
                x1_sliced = x1[:, :, 1:, :, :]  # match decoder output dim (T-1)
                output = args.beta * output + (1.0 - args.beta) * x1_sliced
                first_frame = x1[:, :, 0:1, :, :]
            else:
                first_frame = x2[:, :, 0:1, :, :]

            output_padded = torch.cat([first_frame, output], dim=2)  # (B, C, T, V, 1)

            # (B, C, T, V, 1) -> (B, T, V*C) in one step, then copy to host.
            out = (
                output_padded.squeeze(-1)       # (B, C, T, V)
                .permute(0, 2, 3, 1)            # (B, T, V, C)
                .reshape(-1, T, V * C)
                .contiguous()
                .cpu()
                .numpy()
            )
            out_all_np[start:end] = out

    # Assemble final dict from preallocated buffer.
    retargeted_data: dict[str, np.ndarray] = {
        src_fnames[i]: out_all_np[i] for i in range(n)
    }

    # 6. Save
    print(f"Saving retargeted dataset to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(retargeted_data, f)

    print("Done.")

if __name__ == "__main__":
    main()
