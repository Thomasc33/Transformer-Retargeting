#!/usr/bin/env python3
"""
Evaluate Disentangled TMR model through SGN and Mixformer
Also generate visualizations of anonymized data
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.disentangled_tmr import create_disentangled_tmr
from src.model.sgn import SGN
from src.model.ske_mixf import Model as MixFormerModel
from src.data.datasets import Cross_Data, parse_file_name, sample_frames_fast, NTU120_ACTION_REMAP


def load_model(checkpoint_path, dataset, num_class, device, d_action=768, d_identity=256, d_model=320):
    """Load trained Disentangled TMR model"""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint first so we can rebuild the model with matching tokenizer/codebook flags if present
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    if ckpt_args is not None and not isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = argparse.Namespace(**vars(ckpt_args)) if hasattr(ckpt_args, "__dict__") else argparse.Namespace(**ckpt_args)

    if ckpt_args is not None:
        print(f"Checkpoint args: {vars(ckpt_args)}")
        print(f"no_action_backbone: {getattr(ckpt_args, 'no_action_backbone', 'Not Found')}")

    if ckpt_args is not None:
        dataset = getattr(ckpt_args, "dataset", dataset)
        d_action = getattr(ckpt_args, "d_action", d_action)
        d_identity = getattr(ckpt_args, "d_identity", d_identity)
        d_model = getattr(ckpt_args, "d_model", d_model)

    tokenizer = getattr(ckpt_args, "tokenizer", None) if ckpt_args is not None else None
    if tokenizer in ("none", "None"):
        tokenizer = None

    model = create_disentangled_tmr(
        dataset=dataset,
        num_class=num_class,
        device=device,
        d_action=d_action,
        d_identity=d_identity,
        d_model=d_model,
        num_decoder_layers=getattr(ckpt_args, "num_decoder_layers", 6) if ckpt_args is not None else 6,
        use_pretrained_action=getattr(ckpt_args, "use_action_backbone", True) if ckpt_args is not None else True,
        use_temporal_convs=not getattr(ckpt_args, "no_temporal_convs", False) if ckpt_args is not None else True,
        use_lstm=not getattr(ckpt_args, "no_lstm", False) if ckpt_args is not None else True,
        identity_use_full_sequence=(getattr(ckpt_args, "identity_mode", "static") == "full_seq") if ckpt_args is not None else False,
        tokenizer_type=tokenizer,
        tokenizer_dim=getattr(ckpt_args, "tokenizer_dim", 256) if ckpt_args is not None else 256,
        token_fusion=getattr(ckpt_args, "token_fusion", "add") if ckpt_args is not None else "add",
        use_codebook=getattr(ckpt_args, "use_codebook", False) if ckpt_args is not None else False,
        codebook_size=getattr(ckpt_args, "codebook_size", 256) if ckpt_args is not None else 256,
        codebook_dim=getattr(ckpt_args, "codebook_dim", 256) if ckpt_args is not None else 256,
        codebook_distance=getattr(ckpt_args, "codebook_distance", "euclidean") if ckpt_args is not None else "euclidean",
        vq_commitment_weight=getattr(ckpt_args, "vq_commitment_weight", 0.25) if ckpt_args is not None else 0.25,
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ Model loaded successfully!")
    return model


def load_downstream_models(dataset, num_class, device, task='ar'):
    """Load SGN and Mixformer models for evaluation.

    Args:
        task: 'ar' for action recognition, 'ri' for re-identification
    """
    print(f"\nLoading downstream models (task={task})...")

    # Map dataset to weight source (fallback to ntu for subsets)
    weight_dataset = dataset
    if dataset in ['ntu_small', 'ntu_smoke']:
        weight_dataset = 'ntu'

    # Determine SGN path
    sgn_path = f"output/{weight_dataset}_sgn_{task}_paired/model_best.pth.tar"

    sgn_num_classes = num_class

    if os.path.exists(sgn_path):
        sgn_checkpoint = torch.load(sgn_path, map_location=device, weights_only=False)
        state_dict = sgn_checkpoint.get('state_dict', sgn_checkpoint.get('model_state_dict', sgn_checkpoint))

        if isinstance(state_dict, dict) and 'fc.weight' in state_dict:
            sgn_num_classes = state_dict['fc.weight'].shape[0]
            print(f"  SGN {task.upper()} checkpoint has {sgn_num_classes} classes")
    else:
        print(f"⚠ SGN {task.upper()} weights not found at {sgn_path}")

    # Create SGN model
    sgn_model = SGN(
        num_classes=sgn_num_classes,
        dataset=dataset,
        seg=64,
        bias=True
    ).to(device)

    # Load weights
    if os.path.exists(sgn_path):
        sgn_checkpoint = torch.load(sgn_path, map_location=device, weights_only=False)
        state_dict = sgn_checkpoint.get('state_dict', sgn_checkpoint.get('model_state_dict', sgn_checkpoint))

        new_state_dict = {}
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            sgn_model.load_state_dict(new_state_dict)
        else:
            sgn_model.load_state_dict(state_dict)

        print(f"✓ SGN {task.upper()} loaded from {sgn_path}")

    sgn_model.eval()

    # Load Mixformer
    mixformer_path = f"output/{weight_dataset}_mixformer_{task}_paired/model_best.pth.tar"
    mixformer_num_classes = num_class

    if os.path.exists(mixformer_path):
        mixformer_checkpoint = torch.load(mixformer_path, map_location=device, weights_only=False)
        if 'state_dict' in mixformer_checkpoint:
            mixformer_num_classes = mixformer_checkpoint['state_dict']['fc.weight'].shape[0]
            print(f"  Mixformer {task.upper()} checkpoint has {mixformer_num_classes} classes")
    else:
        if dataset not in ['ntu_smoke', 'ntu_small']:
            print(f"⚠ Mixformer {task.upper()} weights not found at {mixformer_path}, using random initialization with {mixformer_num_classes} classes")
        else:
            print(f"  (Note: Mixformer {task.upper()} weights not found for test dataset {dataset}, using random init)")

    # Create Mixformer model
    mixformer_model = MixFormerModel(
        num_class=mixformer_num_classes,
        num_point=25,
        num_person=1,
        graph='src.graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3
    ).to(device)

    if os.path.exists(mixformer_path):
        mixformer_checkpoint = torch.load(mixformer_path, map_location=device, weights_only=False)
        if 'state_dict' in mixformer_checkpoint:
            mixformer_model.load_state_dict(mixformer_checkpoint['state_dict'])
        else:
            mixformer_model.load_state_dict(mixformer_checkpoint)
        print(f"✓ Mixformer {task.upper()} loaded from {mixformer_path}")

    mixformer_model.eval()

    return sgn_model, mixformer_model


def anonymize_batch(model, x1, x2, y2, device):
    """
    Anonymize a batch using the TMR model

    Args:
        model: Disentangled TMR model
        x1: Source skeleton (P1, A1) - (B, C, T, V, M)
        x2: Target skeleton (P2, A2) - (B, C, T, V, M) - DIFFERENT action!
        y2: Ground truth (P2, A1) - (B, C, T, V, M)

    Returns:
        output: Anonymized skeleton (P2, A1) - (B, C, T-1, V, M)
    """
    with torch.no_grad():
        # Forward pass with no teacher forcing (autoregressive generation)
        # Inference-time setup: do not pass target_motion.
        output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

    return output


def _run_downstream_model(model, data_input, model_name):
    """Run a downstream model on prepared input data (B, C, T, V, M)."""
    if "SGN" in model_name:
        B, C, T, V, M = data_input.size()
        x = data_input.squeeze(-1).permute(0, 2, 3, 1).reshape(B, T, V * C)
        return model(x)
    else:
        return model(data_input)


_NTU120_REMAP_TABLE = None

def _remap_action_labels(action_labels, dataset):
    """Remap non-contiguous NTU120 action labels to contiguous 0-93."""
    if dataset != 'ntu120':
        return action_labels
    global _NTU120_REMAP_TABLE
    if _NTU120_REMAP_TABLE is None or _NTU120_REMAP_TABLE.device != action_labels.device:
        table = torch.zeros(120, dtype=torch.long, device=action_labels.device)
        for orig, new in NTU120_ACTION_REMAP.items():
            table[orig] = new
        _NTU120_REMAP_TABLE = table
    return _NTU120_REMAP_TABLE[action_labels]


def evaluate_raw_baseline(model, dataloader, device, model_name="Model",
                          task='ar', num_classes=None, dataset='ntu'):
    """Evaluate downstream model on RAW (un-retargeted) data as sanity check."""
    label_kind = "action" if task == 'ar' else "actor (P1)"
    print(f"\n[BASELINE] Evaluating {model_name} on RAW data ({task.upper()}, labels={label_kind})...")

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{model_name} {task.upper()} raw")):
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(device).unsqueeze(-1)  # (B, C, T, V, M)

            if task == 'ar':
                labels = _remap_action_labels((actions[:, 0] - 1).long(), dataset).to(device)
            else:
                labels = (actors[:, 0] - 1).long().to(device)

            if num_classes is not None:
                valid = (labels >= 0) & (labels < num_classes)
            else:
                valid = labels >= 0

            if not valid.any():
                continue

            logits = _run_downstream_model(model, x1, model_name)
            preds = logits.argmax(dim=1)
            correct += (preds[valid] == labels[valid]).sum().item()
            total += valid.sum().item()

            if batch_idx >= 100:
                break

    accuracy = correct / total if total > 0 else 0.0
    print(f"[BASELINE] {model_name} {task.upper()} on raw data: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_on_downstream(model, tmr_model, dataloader, device, model_name="Model",
                           task='ar', num_classes=None,
                           actor_pool=None, seg=64,
                           constant_action=False, constant_actor=False,
                           dataset='ntu'):
    """Evaluate anonymized data on downstream model.

    Args:
        task: 'ar' to evaluate action labels, 'ri' to evaluate source-actor labels.
              For RI, low accuracy = good privacy (source identity hidden).
        num_classes: Number of output classes for the model. If provided, samples
                     whose ground-truth label >= num_classes are skipped (handles
                     train/test actor-set mismatches).
    """
    label_kind = "action" if task == 'ar' else "source actor (P1)"
    print(f"\nEvaluating on {model_name} ({task.upper()}, labels={label_kind})...")

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{model_name} {task.upper()}")):
            x1, x2, y1, y2, actors, actions = batch

            x1 = x1.to(device).unsqueeze(-1)

            if actor_pool is not None:
                indices = np.random.randint(0, len(actor_pool), size=x1.size(0))
                x2_new_list = []
                for idx in indices:
                    raw = actor_pool[idx]
                    processed = sample_frames_fast(raw, seg)
                    tensor = torch.from_numpy(processed).float()
                    tensor = tensor.reshape(seg, 25, 3).permute(2, 0, 1)
                    x2_new_list.append(tensor)
                x2 = torch.stack(x2_new_list).to(device).unsqueeze(-1)
            elif constant_actor:
                x2 = x1
            elif constant_action:
                x2 = y2.to(device).unsqueeze(-1)
            else:
                x2 = x2.to(device).unsqueeze(-1)

            y2 = y2.to(device).unsqueeze(-1)

            # Choose labels based on task
            if task == 'ar':
                labels = _remap_action_labels((actions[:, 0] - 1).long(), dataset).to(device)
            else:
                # RI: source actor P1. 1-indexed in dataset → 0-indexed for classifier.
                labels = (actors[:, 0] - 1).long().to(device)

            # Build valid-sample mask: skip out-of-range labels
            if num_classes is not None:
                valid = (labels >= 0) & (labels < num_classes)
            else:
                valid = labels >= 0

            if not valid.any():
                continue

            # Anonymize full batch (retargeting doesn't depend on labels)
            output = anonymize_batch(tmr_model, x1, x2, y2, device)

            first_frame = x2[:, :, 0:1, :, :]
            output_padded = torch.cat([first_frame, output], dim=2)

            logits = _run_downstream_model(model, output_padded, model_name)
            preds = logits.argmax(dim=1)
            correct += (preds[valid] == labels[valid]).sum().item()
            total += valid.sum().item()

            if batch_idx >= 100:
                break

    accuracy = correct / total if total > 0 else 0.0
    print(f"{model_name} {task.upper()} Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def visualize_skeleton(skeleton, title, save_path):
    """
    Visualize a skeleton sequence
    
    Args:
        skeleton: (C, T, V) or (T, V, C) numpy array
        title: Plot title
        save_path: Path to save visualization
    """
    # Ensure shape is (T, V, C)
    if skeleton.shape[0] == 3:  # (C, T, V)
        skeleton = skeleton.transpose(1, 2, 0)  # (T, V, C)
    
    T, V, C = skeleton.shape
    
    # Create figure with multiple frames
    fig = plt.figure(figsize=(20, 4))
    frames_to_plot = [0, T//4, T//2, 3*T//4, T-1]
    
    for idx, frame_idx in enumerate(frames_to_plot):
        ax = fig.add_subplot(1, 5, idx+1, projection='3d')
        
        # Get frame
        frame = skeleton[frame_idx]  # (V, C)
        
        # Plot joints
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='blue', marker='o')
        
        # Plot skeleton connections (NTU skeleton)
        connections = [
            (0, 1), (1, 20), (20, 2), (2, 3),  # Spine
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Right arm
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Left arm
            (0, 12), (12, 13), (13, 14), (14, 15),  # Right leg
            (0, 16), (16, 17), (17, 18), (18, 19)  # Left leg
        ]
        
        for connection in connections:
            if connection[0] < V and connection[1] < V:
                ax.plot([frame[connection[0], 0], frame[connection[1], 0]],
                       [frame[connection[0], 1], frame[connection[1], 1]],
                       [frame[connection[0], 2], frame[connection[1], 2]], 'r-')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}')
        
        # Set equal aspect ratio
        max_range = np.array([frame[:, 0].max()-frame[:, 0].min(),
                             frame[:, 1].max()-frame[:, 1].min(),
                             frame[:, 2].max()-frame[:, 2].min()]).max() / 2.0
        mid_x = (frame[:, 0].max()+frame[:, 0].min()) * 0.5
        mid_y = (frame[:, 1].max()+frame[:, 1].min()) * 0.5
        mid_z = (frame[:, 2].max()+frame[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {save_path}")


def save_skeleton_gif(skeleton, title, save_path):
    """
    Save skeleton sequence as GIF
    """
    if skeleton.shape[0] == 3:
        skeleton = skeleton.transpose(1, 2, 0)
    
    T, V, C = skeleton.shape
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    connections = [
        (0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]
    
    # Pre-calculate bounds
    max_range = np.array([skeleton[:, :, 0].max()-skeleton[:, :, 0].min(),
                         skeleton[:, :, 1].max()-skeleton[:, :, 1].min(),
                         skeleton[:, :, 2].max()-skeleton[:, :, 2].min()]).max() / 2.0
    mid_x = (skeleton[:, :, 0].max()+skeleton[:, :, 0].min()) * 0.5
    mid_y = (skeleton[:, :, 1].max()+skeleton[:, :, 1].min()) * 0.5
    mid_z = (skeleton[:, :, 2].max()+skeleton[:, :, 2].min()) * 0.5

    def update(frame_idx):
        ax.clear()
        frame = skeleton[frame_idx]
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='blue', marker='o')
        for c in connections:
             if c[0] < V and c[1] < V:
                ax.plot([frame[c[0], 0], frame[c[1], 0]],
                       [frame[c[0], 1], frame[c[1], 1]],
                       [frame[c[0], 2], frame[c[1], 2]], 'r-')
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title(f"{title} (Frame {frame_idx})")

    ani = FuncAnimation(fig, update, frames=T, interval=50)
    ani.save(save_path, writer=PillowWriter(fps=20))
    plt.close()
    print(f"✓ Saved GIF to {save_path}")


def evaluate_visualizations(model, dataloader, device, output_dir, num_vis=5):
    """Generate visualizations for a few samples"""
    print(f"\nGenerating {num_vis} visualizations...")
    model.eval()
    
    vis_count = 0
    with torch.no_grad():
        for batch in dataloader:
            x1, x2, y1, y2, actors, actions = batch
            
            x1 = x1.to(device).unsqueeze(-1)
            x2 = x2.to(device).unsqueeze(-1)
            y2 = y2.to(device).unsqueeze(-1)
            
            output = anonymize_batch(model, x1, x2, y2, device)
            
            # Pad output
            first_frame = x2[:, :, 0:1, :, :]
            output_padded = torch.cat([first_frame, output], dim=2)
            
            # Convert to numpy
            x1_np = x1.cpu().numpy()
            x2_np = x2.cpu().numpy()
            out_np = output_padded.cpu().numpy()
            
            for i in range(x1.size(0)):
                if vis_count >= num_vis:
                    return
                
                # Squeeze extra dims: (1, C, T, V, 1) -> (C, T, V)
                src = x1_np[i, :, :, :, 0]
                tgt = x2_np[i, :, :, :, 0]
                res = out_np[i, :, :, :, 0]
                
                # Save static plots
                visualize_skeleton(src, f"Source (A{int(actions[i,0])})", 
                                 os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_source.png'))
                visualize_skeleton(tgt, f"Target Identity (A{int(actions[i,1])})", 
                                 os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_target.png'))
                visualize_skeleton(res, "Retargeted Output", 
                                 os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_output.png'))
                
                # Save GIFs
                save_skeleton_gif(src, f"Source", 
                                os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_source.gif'))
                save_skeleton_gif(tgt, f"Target Identity", 
                                os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_target.gif'))
                save_skeleton_gif(res, "Retargeted Output", 
                                os.path.join(output_dir, 'visualizations', f'sample_{vis_count}_output.gif'))

                vis_count += 1


def main():
    parser = argparse.ArgumentParser(description='Evaluate Disentangled TMR')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/ntu_cv_paired_comprehensive.pt')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri', 'ntu_smoke', 'ntu_small'])
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='output/disentangled_tmr_eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--d_action', type=int, default=768)
    parser.add_argument('--d_identity', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=320, help='Decoder model dimension')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--target_actor', type=int, default=None, help='Target actor ID (1-based) for all retargeting')
    parser.add_argument('--constant_action', action='store_true', help='Use constant action evaluation mode (Target Action = Source Action)')
    parser.add_argument('--constant_actor', action='store_true', help='Use constant actor evaluation mode (Target Actor = Source Actor)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get number of classes
    if args.dataset in ['ntu', 'ntu_smoke', 'ntu_small']:
        num_class = 49
    elif args.dataset == 'ntu120':
        num_class = 94   # 120 - 26 two-person actions (50-60, 106-120)
    elif args.dataset == 'etri':
        num_class = 55
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    test_dataset = data['test']

    # Build Actor Pool if needed
    actor_pool = None
    if args.target_actor is not None:
        print(f"Building pool for Target Actor {args.target_actor}...")
        actor_pool = []
        # Access raw data dictionary from Cross_Data
        # test_dataset.X is {fname: skeleton}
        count = 0
        for fname, skel_data in test_dataset.X.items():
            info = parse_file_name(fname, args.dataset)
            if info['P'] == args.target_actor:
                actor_pool.append(skel_data)
                count += 1
        
        if not actor_pool:
            print(f"Error: No samples found for Actor {args.target_actor} in test set!")
            sys.exit(1)
            
        print(f"Found {len(actor_pool)} samples for Actor {args.target_actor}")

    # Limit samples - handle both Cross_Data objects and lists
    if args.num_samples > 0 and args.num_samples < len(test_dataset):
        if isinstance(test_dataset, list):
            # If it's a list, just slice it
            test_dataset = test_dataset[:args.num_samples]
        elif hasattr(test_dataset, 'sampled_data'):
            # If it's a Cross_Data object, slice its attributes
            test_dataset.sampled_data = test_dataset.sampled_data[:args.num_samples]
            test_dataset.actors = test_dataset.actors[:args.num_samples]
            test_dataset.actions = test_dataset.actions[:args.num_samples]

    print(f"Loaded {len(test_dataset)} test samples")
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )
    
    # Load TMR model
    tmr_model = load_model(args.checkpoint, args.dataset, num_class, device, args.d_action, args.d_identity, args.d_model)

    # Load downstream models for AR (action recognition / utility)
    sgn_ar, mixformer_ar = load_downstream_models(args.dataset, num_class, device, task='ar')

    # Load downstream models for RI (re-identification / privacy)
    sgn_ri, mixformer_ri = load_downstream_models(args.dataset, num_class, device, task='ri')

    seg = getattr(test_dataset, 'seg', 64)

    # --- Raw baseline sanity check ---
    # Evaluate downstream models on ORIGINAL data (no retargeting) to confirm they work
    print("\n" + "="*80)
    print("RAW BASELINE (sanity check — downstream models on original data)")
    print("="*80)

    sgn_ar_nc = sgn_ar.fc.weight.shape[0] if hasattr(sgn_ar, 'fc') else None
    mf_ar_nc = mixformer_ar.fc.weight.shape[0] if hasattr(mixformer_ar, 'fc') else None
    sgn_ri_nc = sgn_ri.fc.weight.shape[0] if hasattr(sgn_ri, 'fc') else None
    mf_ri_nc = mixformer_ri.fc.weight.shape[0] if hasattr(mixformer_ri, 'fc') else None

    raw_sgn_ar = evaluate_raw_baseline(sgn_ar, test_loader, device, "SGN", task='ar', num_classes=sgn_ar_nc, dataset=args.dataset)
    raw_mf_ar = evaluate_raw_baseline(mixformer_ar, test_loader, device, "Mixformer", task='ar', num_classes=mf_ar_nc, dataset=args.dataset)
    raw_sgn_ri = evaluate_raw_baseline(sgn_ri, test_loader, device, "SGN", task='ri', num_classes=sgn_ri_nc, dataset=args.dataset)
    raw_mf_ri = evaluate_raw_baseline(mixformer_ri, test_loader, device, "Mixformer", task='ri', num_classes=mf_ri_nc, dataset=args.dataset)

    eval_kwargs = dict(
        tmr_model=tmr_model, dataloader=test_loader, device=device,
        actor_pool=actor_pool, seg=seg,
        constant_action=args.constant_action, constant_actor=args.constant_actor,
        dataset=args.dataset,
    )

    # --- Action Recognition (utility) ---
    print("\n" + "="*80)
    print("ACTION RECOGNITION (UTILITY) — on retargeted data")
    print("="*80)

    sgn_ar_acc = evaluate_on_downstream(
        sgn_ar, model_name="SGN", task='ar', num_classes=sgn_ar_nc, **eval_kwargs)
    mixformer_ar_acc = evaluate_on_downstream(
        mixformer_ar, model_name="Mixformer", task='ar', num_classes=mf_ar_nc, **eval_kwargs)

    # --- Re-Identification (privacy) ---
    # Low accuracy = good privacy (source identity P1 is hidden)
    print("\n" + "="*80)
    print("RE-IDENTIFICATION (PRIVACY) — on retargeted data")
    print("="*80)

    sgn_ri_acc = evaluate_on_downstream(
        sgn_ri, model_name="SGN", task='ri', num_classes=sgn_ri_nc, **eval_kwargs)
    mixformer_ri_acc = evaluate_on_downstream(
        mixformer_ri, model_name="Mixformer", task='ri', num_classes=mf_ri_nc, **eval_kwargs)

    # Visualize
    if args.visualize:
        evaluate_visualizations(tmr_model, test_loader, device, args.output_dir, args.num_vis)

    # Save results
    results = {
        'raw_sgn_ar_accuracy': raw_sgn_ar,
        'raw_mixformer_ar_accuracy': raw_mf_ar,
        'raw_sgn_ri_accuracy': raw_sgn_ri,
        'raw_mixformer_ri_accuracy': raw_mf_ri,
        'sgn_ar_accuracy': sgn_ar_acc,
        'mixformer_ar_accuracy': mixformer_ar_acc,
        'sgn_ri_accuracy': sgn_ri_acc,
        'mixformer_ri_accuracy': mixformer_ri_acc,
        'checkpoint': args.checkpoint,
        'num_samples': len(test_dataset),
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.pt')
    torch.save(results, results_path)
    print(f"\n✓ Results saved to {results_path}")

    # Generate visualizations
    if args.visualize:
        print(f"\nGenerating {args.num_vis} visualizations...")
        vis_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        for idx, batch in enumerate(vis_loader):
            if idx >= args.num_vis:
                break

            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(device).unsqueeze(-1)
            y2 = y2.to(device).unsqueeze(-1)
            y1 = y1.to(device).unsqueeze(-1)

            output = anonymize_batch(tmr_model, x1, y2, y1, device)

            x1_np = x1[0].squeeze(-1).cpu().numpy()
            output_np = output[0].squeeze(-1).cpu().numpy()

            action_id = int(actions[0, 0].item())
            actor_src = int(actors[0, 0].item())
            actor_tgt = int(actors[0, 1].item())

            vis_path_src = os.path.join(args.output_dir, 'visualizations',
                                       f'sample_{idx:03d}_source_P{actor_src}_A{action_id}.png')
            vis_path_anon = os.path.join(args.output_dir, 'visualizations',
                                        f'sample_{idx:03d}_anonymized_P{actor_tgt}_A{action_id}.png')

            visualize_skeleton(x1_np, f'Source: Person {actor_src}, Action {action_id}', vis_path_src)
            visualize_skeleton(output_np, f'Anonymized: Person {actor_tgt}, Action {action_id}', vis_path_anon)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("  Raw baseline (original data):")
    print(f"    SGN AR:       {raw_sgn_ar:.4f}")
    print(f"    Mixformer AR: {raw_mf_ar:.4f}")
    print(f"    SGN RI:       {raw_sgn_ri:.4f}")
    print(f"    Mixformer RI: {raw_mf_ri:.4f}")
    print("  Retargeted data:")
    print(f"    SGN AR:       {sgn_ar_acc:.4f}  (utility, higher=better)")
    print(f"    Mixformer AR: {mixformer_ar_acc:.4f}  (utility, higher=better)")
    if sgn_ri_nc:
        print(f"    SGN RI:       {sgn_ri_acc:.4f}  (privacy, lower=better, chance={1/sgn_ri_nc:.4f})")
    if mf_ri_nc:
        print(f"    Mixformer RI: {mixformer_ri_acc:.4f}  (privacy, lower=better, chance={1/mf_ri_nc:.4f})")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
