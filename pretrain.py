"""
Masked Language Model Pretraining for Skeleton-based Action Recognition.

This script handles pretraining of skeleton transformer models on various datasets
(NTU-60, NTU-120, ETRI) using a masked autoencoder approach.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from util import init_seed
from data import load_data, process_mlm, Masked_AE_Data
from model.encoder import Encoder, pre_process
from model.mlm_decoder import MLMDecoder, post_process
import numpy as np
import json # For saving metrics
from sklearn.metrics import accuracy_score # For calculating accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition Pretraining')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ntu120', choices=['ntu120', 'ntu', 'etri'],
                        help='Dataset to use for pretraining (default: ntu120)')
    parser.add_argument('--setting', type=str, default='cs', choices=['cs', 'cv'],
                        help='Evaluation setting: cs (cross-subject) or cv (cross-view) (default: cs)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length for temporal dimension (default: 64)')

    # Masking parameters
    parser.add_argument('--temporal_masking_ratio', type=float, default=0.5,
                        help='Ratio of frames to mask (temporal masking) (default: 0.5)')
    parser.add_argument('--spatial_masking_ratio', type=float, default=0.5,
                        help='Ratio of joints to mask (spatial masking) (default: 0.5)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # System parameters
    parser.add_argument('--cudnn-enabled', action='store_true',
                        help='Enable cuDNN benchmarking (may improve performance)')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training mode')

    # Linear Probing parameters
    parser.add_argument('--enable_linear_probe', action='store_true',
                        help='Enable linear probing after pretraining')
    parser.add_argument('--probe_epochs', type=int, default=50,
                        help='Number of epochs for linear probing')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Learning rate for linear probing optimizer')
    parser.add_argument('--probe_batch_size', type=int, default=64,
                        help='Batch size for linear probing')
    # num_action_classes and num_actor_classes will be determined dynamically later if possible,
    # or could be added as arguments if dynamic determination is too complex for all datasets.
    # For now, let's assume they can be inferred or are fixed for common datasets.
    parser.add_argument('--num_action_classes', type=int, default=None, help='Number of action classes for probing (e.g., 60 for NTU, 120 for NTU120)')
    parser.add_argument('--num_actor_classes', type=int, default=None, help='Number of actor classes for probing (e.g., 40 for NTU CS)')


    return parser.parse_args()


class SkeletonAutoEncoder(nn.Module):
    """
    Skeleton Autoencoder for masked reconstruction of motion sequences.

    Combines a spatial-temporal encoder with an MLM decoder for reconstruction.
    """
    def __init__(self, dataset, seq_len):
        super(SkeletonAutoEncoder, self).__init__()
        self.seq_len = seq_len

        # Encoder (spatial-temporal transformer)
        self.encoder = Encoder(
            num_class=120 if dataset == 'ntu120' else (60 if dataset == 'ntu' else 55),
            num_point=25,
            num_person=1,
            graph='graph.ntu_rgb_d.Graph',
            graph_args={'labeling_mode': 'spatial'},
            in_channels=3,
            debug=False,
            dataset=dataset,
            load_pretrained=False,
            freeze_layers=False
        )

        # Decoder (MLM transformer decoder)
        self.decoder = MLMDecoder(
            d_model=320,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

        # Final projection to output coordinates
        self.output_layer = nn.Linear(320, 3)  # Map d_model to channels (x,y,z)

    def forward(self, x):
        """Forward pass through the autoencoder."""
        # Preprocess input to expected format
        x = pre_process(x, x.size(0), self.seq_len, 25, 3)

        # Encode the motion sequence
        x = self.encoder(x)

        # Decode with MLM transformer decoder
        x = self.decoder(x)

        # Project to output coordinates
        x = self.output_layer(x)  # Shape: (sequence_length, batch_size, channels)

        # Postprocess output to original format
        return post_process(x, self.seq_len, x.size(1), 1, 25, 3)


def setup_distributed():
    """Initialize the distributed training environment.

    This function reads environment variables set by torchrun or SLURM
    and initializes the distributed process group.
    """
    # Check if environment variables are already set by torchrun
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get local rank from environment variable if available
    local_rank = int(os.environ.get('LOCAL_RANK', rank))

    # Set device
    torch.cuda.set_device(local_rank)

    # Print distributed training info
    print(f"[DDP Init] rank={rank}, local_rank={local_rank}, world_size={world_size}")

    return rank, world_size, torch.device('cuda', local_rank)


def prepare_data(args, rank):
    """
    Prepare datasets and data loaders for training.

    Args:
        args: Command line arguments
        rank: Process rank for distributed training

    Returns:
        train_loader, test_loader: Data loaders for training and testing
    """
    dataset = args.dataset
    T = args.seq_len
    setting = args.setting

    # Create directory if it doesn't exist
    os.makedirs(f'data/{dataset}', exist_ok=True)

    # Define file paths for both regular and comprehensive datasets
    regular_data_path = f'data/{dataset}/pretraining_data_{setting}.pt'
    comprehensive_data_path = f'data/{dataset}/pretraining_data_{setting}_comprehensive.pt'

    # Check if comprehensive dataset exists
    if os.path.exists(comprehensive_data_path):
        if rank == 0:
            print(f"Loading COMPREHENSIVE dataset from {comprehensive_data_path}")
        saved_data = torch.load(comprehensive_data_path, map_location='cpu')
        train_dataset = saved_data['train_dataset']
        test_dataset = saved_data['test_dataset']
        if rank == 0:
            print(f"Loaded comprehensive dataset with {len(train_dataset)} training and {len(test_dataset)} test samples")

    # Check if regular preprocessed data exists
    elif os.path.exists(regular_data_path):
        if rank == 0:
            print(f"Loading dataset from {regular_data_path}")
        saved_data = torch.load(regular_data_path, map_location='cpu')
        train_dataset = saved_data['train_dataset']
        test_dataset = saved_data['test_dataset']
        if rank == 0:
            print(f"Loaded regular dataset with {len(train_dataset)} training and {len(test_dataset)} test samples")

    # If no preprocessed data exists, generate it
    else:
        if rank == 0:
            print(f"Generating dataset for {args.dataset} with {args.setting} setting")
        
        # Store original X and train/test file lists for potential probing
        X_all_data = load_data(args.dataset, T=args.seq_len) # Load all data once
        train_files_list, test_files_list = process_mlm(X_all_data, args.setting, args.dataset, args.seq_len)


        # Check if comprehensive paired data exists
        comprehensive_paired_path = f'data/{dataset}_{setting}_paired_comprehensive.pt'
        if os.path.exists(comprehensive_paired_path) and setting == 'cv':
            if rank == 0:
                print(f"Found comprehensive paired data at {comprehensive_paired_path}, using it for pretraining")

            # Load the comprehensive paired data
            paired_data = torch.load(comprehensive_paired_path)

            # Extract all unique skeletons from the paired data for pretraining
            X = load_data(dataset, T=T)

            # Get all unique filenames from the paired data
            train_files = set()
            test_files = set()

            for sample in paired_data['train'].sampled_data:
                train_files.add(sample[0][2])  # p1, a1, fname
                train_files.add(sample[1][2])  # p1, a2, fname
                train_files.add(sample[2][2])  # p2, a1, fname
                train_files.add(sample[3][2])  # p2, a2, fname

            for sample in paired_data['test'].sampled_data:
                test_files.add(sample[0][2])
                test_files.add(sample[1][2])
                test_files.add(sample[2][2])
                test_files.add(sample[3][2])

            if rank == 0:
                print(f"Extracted {len(train_files)} training files and {len(test_files)} test files from comprehensive paired data")

            # Create datasets
            train_dataset = Masked_AE_Data(
                torch.tensor(np.array([X[f] for f in train_files]), dtype=torch.float32),
                args.temporal_masking_ratio, # Changed from args.frame_masking_ratio
                args.spatial_masking_ratio,  # Changed from args.joint_masking_ratio
                seg=T,
                augment=True,
                theta=0.5  # Use 0.5 for cross-view
            )
            test_dataset = Masked_AE_Data(
                torch.tensor(np.array([X[f] for f in test_files]), dtype=torch.float32),
                args.temporal_masking_ratio, # Changed from args.frame_masking_ratio
                args.spatial_masking_ratio,  # Changed from args.joint_masking_ratio
                seg=T,
                augment=False,  # No augmentation for test data
                theta=0.5
            )

            # Save as comprehensive dataset
            if rank == 0:
                torch.save({
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset
                }, comprehensive_data_path)
                print(f"Saved comprehensive dataset to {comprehensive_data_path}")

        # Otherwise, use the regular process_mlm approach
        else:
            if rank == 0:
                print(f"Using regular data processing approach")
            # X_all_data and train_files_list, test_files_list are already available from above
            train_dataset = Masked_AE_Data(
                torch.tensor([X_all_data[f] for f in train_files_list], dtype=torch.float32),
                args.temporal_masking_ratio, 
                args.spatial_masking_ratio,  
                seg=T,
                augment=True,
                theta=0.3 if setting == 'cs' else 0.5  # Use 0.3 for cross-subject, 0.5 for cross-view
            )
            test_dataset = Masked_AE_Data(
                torch.tensor([X_all_data[f] for f in test_files_list], dtype=torch.float32),
                args.temporal_masking_ratio, 
                args.spatial_masking_ratio,  
                seg=T,
                augment=False,  # No augmentation for test data
                theta=0.3 if setting == 'cs' else 0.5
            )

            # Save as regular dataset
            if rank == 0:
                torch.save({
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset
                }, regular_data_path)
                print(f"Saved regular dataset to {regular_data_path}")

    # Create data loaders with appropriate samplers
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

    # Return filenames for probing if not DDP or on rank 0, otherwise they are not needed by other ranks for probing
    # And also return X_all_data for probing dataset creation
    if args.enable_linear_probe and (not args.distributed or rank == 0):
        return train_loader, test_loader, train_sampler if args.distributed else None, train_files_list, test_files_list, X_all_data
    else:
        return train_loader, test_loader, train_sampler if args.distributed else None, None, None, None


# --- Helper function to parse labels from filename (NTU-specific example) ---
def parse_ntu_filename(filename):
    parts = filename.split('/')[-1].split('.')[0] # Get 'S001C001P001R001A001'
    action_id = int(parts[parts.find('A'):].replace('A', '')) - 1 # 0-indexed
    actor_id = int(parts[parts.find('P'):parts.find('R')].replace('P', '')) - 1 # 0-indexed
    return action_id, actor_id

# --- New ProbingDataset class ---
class ProbingDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, data_X_dict, seq_len, dataset_name, pre_process_fn, is_train_split=True):
        self.file_paths = file_paths
        self.data_X_dict = data_X_dict
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.pre_process_fn = pre_process_fn
        self.is_train_split = is_train_split # To handle potential augmentation differently if needed

        self.skeletons = []
        self.action_labels = []
        self.actor_labels = []
        
        action_map = {}
        actor_map = {}

        for filename in self.file_paths:
            raw_data = self.data_X_dict[filename] # This is (C, T, V, M) = (3, T_orig, 25, 1 or 2)
            
            # Pad or truncate T dimension to self.seq_len
            c, t_orig, v, m = raw_data.shape
            if t_orig < self.seq_len:
                padding = np.zeros((c, self.seq_len - t_orig, v, m))
                processed_data = np.concatenate((raw_data, padding), axis=1)
            else:
                # Simple truncation, or could do sampling for training
                if self.is_train_split and t_orig > self.seq_len: # Random crop for training
                    start_idx = np.random.randint(0, t_orig - self.seq_len + 1)
                    processed_data = raw_data[:, start_idx:start_idx + self.seq_len, :, :]
                else: # Center crop for testing or if t_orig == self.seq_len
                    start_idx = (t_orig - self.seq_len) // 2
                    processed_data = raw_data[:, start_idx:start_idx + self.seq_len, :, :]


            self.skeletons.append(torch.tensor(processed_data, dtype=torch.float32))

            if 'ntu' in self.dataset_name.lower(): # Basic parsing for NTU datasets
                action_id_orig, actor_id_orig = parse_ntu_filename(filename)
                self.action_labels.append(action_id_orig)
                self.actor_labels.append(actor_id_orig)
            else:
                # Placeholder for other datasets - requires specific parsing logic
                self.action_labels.append(0) # Default label
                self.actor_labels.append(0)  # Default label
        
        # Create unique label maps
        unique_actions = sorted(list(set(self.action_labels)))
        self.action_to_idx = {act: i for i, act in enumerate(unique_actions)}
        self.idx_to_action = {i: act for act, i in self.action_to_idx.items()}
        self.num_unique_actions = len(unique_actions)

        unique_actors = sorted(list(set(self.actor_labels)))
        self.actor_to_idx = {act: i for i, act in enumerate(unique_actors)}
        self.idx_to_actor = {i: act for act, i in self.actor_to_idx.items()}
        self.num_unique_actors = len(unique_actors)

        # Convert labels to mapped indices
        self.action_labels = [self.action_to_idx[lbl] for lbl in self.action_labels]
        self.actor_labels = [self.actor_to_idx[lbl] for lbl in self.actor_labels]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Skeleton data is already processed to (C, T, V, M) and tensor
        skeleton_tensor = self.skeletons[idx] 
        
        # The pre_process_fn from model.encoder expects (N,C,T,V,M) or similar
        # and returns the processed tensor for the encoder.
        # Here, we have a single sample (C,T,V,M). We unsqueeze to add batch dim.
        # The pre_process_fn in model.encoder.py is:
        # pre_process(tensor, N, T, V, C) -> (N, T, V*C) if M=1 and then (N, T, D_embed) by patch_embed
        # Or pre_process(x, N, T, V, C, M) -> (N, M, T, V, C) -> (N*M, T, V, C) -> (N*M, T, D_embed)
        # The encoder itself (MixSkeletonFormerEncoder) expects (B, C, T, V, M) as input to patch_embed
        # So, skeleton_tensor is already (C,T,V,M), which is fine for one sample.
        # The DataLoader will batch them up to (B,C,T,V,M)

        return skeleton_tensor, self.action_labels[idx], self.actor_labels[idx]

# --- New LinearProbeNet class ---
class LinearProbeNet(nn.Module):
    def __init__(self, encoder_instance, encoder_feature_dim, num_action_classes, num_actor_classes):
        super().__init__()
        self.encoder = encoder_instance
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.action_head = nn.Linear(encoder_feature_dim, num_action_classes)
        self.actor_head = nn.Linear(encoder_feature_dim, num_actor_classes)
        self.encoder_feature_dim = encoder_feature_dim


    def forward(self, x_skeleton_batch): # x_skeleton_batch is (B, C, T, V, M)
        with torch.no_grad():
            # The encoder is MixSkeletonFormerEncoder, its forward pass:
            # x = self.patch_embed(x) -> (B, num_patches=T, embed_dim=D)
            # ... goes through transformer blocks ...
            # output is (B, T, D). CLS token is typically x[:, 0]
            encoder_sequence_output = self.encoder.forward_features(x_skeleton_batch) # (B, T, D)
            # Assuming forward_features gives the sequence output before any head
            # And CLS token is the first token if used by MixSkeletonFormerEncoder design
            cls_features = encoder_sequence_output[:, 0, :] # (B, D)

        action_logits = self.action_head(cls_features)
        actor_logits = self.actor_head(cls_features)
        return action_logits, actor_logits

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Execute one training epoch."""
    model.train()
    train_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Reshape data to match output format
        data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]

        # Compute loss
        loss = criterion(output, data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.6f}")

    # Average loss over all batches
    avg_loss = train_loss / total_batches
    print(f"Epoch {epoch} completed. Average training loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            # Forward pass
            output = model(data)

            # Reshape data to match output format
            data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]

            # Compute loss
            loss = criterion(output, data)
            test_loss += loss.item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Evaluation - Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.6f}")

    # Average loss over all batches
    avg_loss = test_loss / total_batches
    print(f"Evaluation completed. Average test loss: {avg_loss:.6f}")
    return avg_loss


def main():
    """Main training function."""
    args = parse_args()

    # Set CUDNN mode
    torch.backends.cudnn.enabled = args.cudnn_enabled

    # Set random seed for reproducibility
    init_seed(args.seed)

    # Set up distributed training if enabled
    if args.distributed:
        rank, world_size, device = setup_distributed()
        print(f"Process {rank}/{world_size} using device {device}")
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running in non-distributed mode on device {device}")

    # Initialize wandb for experiment tracking (only on rank 0)
    best_overall_test_loss = float('inf')
    best_overall_epoch = 0
    # This path will be relative to the root of the project, inside the specific experiment's folder
    best_encoder_path_for_probe = None 

    if rank == 0:
        # Check if this is a comprehensive dataset training
        is_comprehensive = False
        comprehensive_data_path = f'data/{args.dataset}/pretraining_data_{args.setting}_comprehensive.pt'
        if os.path.exists(comprehensive_data_path) or os.path.exists(f'data/{args.dataset}_{args.setting}_paired_comprehensive.pt'):
            is_comprehensive = True

        # Set project name with setting and comprehensive flag
        project_name = f'MLM_PT_{args.dataset}_{args.setting}'
        if is_comprehensive:
            project_name += '_comprehensive'

        wandb.init(
            project=project_name,
            config=vars(args)
        )

    # Prepare data loaders
    # Modify prepare_data to return train_files, test_files, and X_all_data for probing
    if args.enable_linear_probe and (not args.distributed or rank == 0) :
        train_loader, test_loader, train_sampler, train_files_list, test_files_list, X_all_data = prepare_data(args, rank)
    else:
        train_loader, test_loader, train_sampler, _, _, _ = prepare_data(args, rank)

    # Create the model
    model = SkeletonAutoEncoder(args.dataset, args.seq_len).to(device)

    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler to ensure different data ordering per epoch
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Print epoch information (all ranks)
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        print(f"Rank {rank}/{world_size} starting epoch {epoch}")

        # Synchronize before starting epoch
        if args.distributed:
            dist.barrier()

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Synchronize after training
        if args.distributed:
            dist.barrier()

        # Evaluate on test set
        test_loss = evaluate(model, test_loader, criterion, device)

        # Synchronize after evaluation
        if args.distributed:
            dist.barrier()

        # Log metrics and save model (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs} Summary | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss
            })

            # Save model checkpoint for each epoch
            # Determine if this is a comprehensive dataset training
            is_comprehensive = False
            comprehensive_data_path = f'data/{args.dataset}/pretraining_data_{args.setting}_comprehensive.pt'
            if os.path.exists(comprehensive_data_path):
                is_comprehensive = True

            # Create checkpoint directory with setting and comprehensive flag
            checkpoint_dir = f'eval/mixformer/pretrained/{args.dataset}/epochs_{args.setting}'
            if is_comprehensive: # is_comprehensive was defined earlier based on data path
                checkpoint_dir += '_comprehensive'
            checkpoint_dir += f'_temporal_{args.temporal_masking_ratio}_spatial_{args.spatial_masking_ratio}'
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Get the encoder state dict (accounting for DDP wrapper if needed)
            if args.distributed:
                encoder_state_dict = model.module.encoder.state_dict()
            else:
                encoder_state_dict = model.encoder.state_dict()

            epoch_model_path = f'{checkpoint_dir}/encoder_{epoch}.pth'
            torch.save(encoder_state_dict, epoch_model_path)
            print(f"Saved model checkpoint for epoch {epoch} to {epoch_model_path}")

            if test_loss < best_overall_test_loss:
                best_overall_test_loss = test_loss
                best_overall_epoch = epoch
                best_encoder_path_for_probe = os.path.join(checkpoint_dir, 'encoder_best.pth')
                torch.save(encoder_state_dict, best_encoder_path_for_probe)
                print(f"Saved new best encoder for probing at epoch {epoch} to {best_encoder_path_for_probe}")


    # Final synchronization before saving
    if args.distributed:
        dist.barrier()

    # Save final model (only on rank 0) - This might be redundant if best_encoder_path_for_probe is used
    # The 'final model' here is just the model from the last epoch.
    # We should ensure that `best_encoder_path_for_probe` points to the truly best one.

    # Linear Probing (on rank 0 after main training)
    if rank == 0 and args.enable_linear_probe:
        print("\\n" + "="*20 + " Starting Linear Probing " + "="*20)

        if not best_encoder_path_for_probe or not os.path.exists(best_encoder_path_for_probe):
            print("Error: Best encoder model for probing not found or not saved. Skipping probing.")
        elif not train_files_list or not test_files_list or not X_all_data:
            print("Error: Data for probing (file lists or X_all_data) not available. Skipping probing.")
        else:
            print(f"Loading best encoder from: {best_encoder_path_for_probe} for linear probing.")
            # Instantiate the original Encoder class used in SkeletonAutoEncoder
            # The encoder in SkeletonAutoEncoder is model.encoder.Encoder
            # We need to ensure its arguments are correctly passed.
            # Let's get them from how it's defined in SkeletonAutoEncoder
            probe_encoder_instance = Encoder(
                num_class=120 if args.dataset == 'ntu120' else (60 if args.dataset == 'ntu' else 55), # This num_class is for the original task, not directly used by encoder if pretraining
                num_point=25, # Assuming NTU default
                num_person=1, # Assuming single person skeletons for pretraining
                graph='graph.ntu_rgb_d.Graph', # Assuming NTU default
                graph_args={'labeling_mode': 'spatial'},
                in_channels=3,
                dataset=args.dataset, # Pass dataset to encoder if it uses it for internal config
                # load_pretrained=False, freeze_layers=False are defaults for a new instance
            )
            probe_encoder_instance.load_state_dict(torch.load(best_encoder_path_for_probe, map_location=device))
            probe_encoder_instance.to(device)
            probe_encoder_instance.eval() # Ensure it's in eval mode and params are frozen by LinearProbeNet

            # Prepare Probing DataLoaders
            # The pre_process function from model.encoder.pre_process might be needed by ProbingDataset
            # For now, ProbingDataset handles its own data prep to (C,T,V,M)
            # The Encoder's patch_embed will handle (B,C,T,V,M) -> (B,T,D)
            print("Preparing probing datasets...")
            # We need to determine num_action_classes and num_actor_classes for the ProbingDataset and LinearProbeNet
            # This can be done by inspecting the dataset or passed as args.
            # For now, ProbingDataset calculates these internally based on the provided file_paths.
            
            # Create a temporary dataset to get num_classes if not provided
            # This is a bit inefficient but ensures correct class numbers for the current data split
            temp_train_probe_ds_for_meta = ProbingDataset(train_files_list, X_all_data, args.seq_len, args.dataset, None, is_train_split=True)
            num_actions = args.num_action_classes if args.num_action_classes is not None else temp_train_probe_ds_for_meta.num_unique_actions
            # For actors, it's trickier as train/test might have different actors.
            # We need a global actor mapping or use the max number of unique actors found.
            # Let's use the number of unique actors found in the training set for the actor head size.
            num_actors = args.num_actor_classes if args.num_actor_classes is not None else temp_train_probe_ds_for_meta.num_unique_actors
            
            print(f"Probing with Num Action Classes: {num_actions}, Num Actor Classes: {num_actors}")

            train_probe_dataset = ProbingDataset(train_files_list, X_all_data, args.seq_len, args.dataset, None, is_train_split=True)
            test_probe_dataset = ProbingDataset(test_files_list, X_all_data, args.seq_len, args.dataset, None, is_train_split=False)
            
            # Ensure the label mapping is consistent if train/test have different sets of actors/actions
            # For simplicity, ProbingDataset creates maps per split. This is fine if heads are trained on train and eval on test.
            # If num_actions/num_actors are passed as args, ProbingDataset should use them to filter/map.

            train_probe_loader = torch.utils.data.DataLoader(train_probe_dataset, batch_size=args.probe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_probe_loader = torch.utils.data.DataLoader(test_probe_dataset, batch_size=args.probe_batch_size, shuffle=False, num_workers=4, pin_memory=True)

            encoder_feature_dim = probe_encoder_instance.embed_dims # MixSkeletonFormerEncoder has embed_dims (e.g. 320)
            probe_net = LinearProbeNet(probe_encoder_instance, encoder_feature_dim, num_actions, num_actors).to(device)
            
            optimizer_probe = torch.optim.Adam(filter(lambda p: p.requires_grad, probe_net.parameters()), lr=args.probe_lr) # Only optimize linear heads
            criterion_action = nn.CrossEntropyLoss()
            criterion_actor = nn.CrossEntropyLoss()

            print("Starting linear probing training...")
            for epoch_probe in range(1, args.probe_epochs + 1):
                probe_net.train() # Set LinearProbeNet to train (only heads are trainable)
                total_probe_loss_epoch = 0
                for skeletons, action_labels, actor_labels in train_probe_loader:
                    skeletons = skeletons.to(device) # Expected (B, C, T, V, M)
                    action_labels = action_labels.to(device).long()
                    actor_labels = actor_labels.to(device).long()
                    
                    optimizer_probe.zero_grad()
                    action_logits, actor_logits = probe_net(skeletons)
                    
                    loss_action = criterion_action(action_logits, action_labels)
                    loss_actor = criterion_actor(actor_logits, actor_labels)
                    total_probe_loss = loss_action + loss_actor
                    
                    total_probe_loss.backward()
                    optimizer_probe.step()
                    total_probe_loss_epoch += total_probe_loss.item()
                avg_probe_loss = total_probe_loss_epoch / len(train_probe_loader)
                print(f"Probe Epoch {epoch_probe}/{args.probe_epochs}, Avg Loss: {avg_probe_loss:.4f}")

            print("Evaluating linear probe...")
            probe_net.eval()
            all_action_preds, all_action_labels_true = [], []
            all_actor_preds, all_actor_labels_true = [], []
            with torch.no_grad():
                for skeletons, action_labels, actor_labels in test_probe_loader:
                    skeletons = skeletons.to(device)
                    action_labels = action_labels.to(device).long()
                    actor_labels = actor_labels.to(device).long()

                    action_logits, actor_logits = probe_net(skeletons)
                    
                    action_preds = torch.argmax(action_logits, dim=1)
                    actor_preds = torch.argmax(actor_logits, dim=1)
                    
                    all_action_preds.extend(action_preds.cpu().numpy())
                    all_action_labels_true.extend(action_labels.cpu().numpy())
                    all_actor_preds.extend(actor_preds.cpu().numpy())
                    all_actor_labels_true.extend(actor_labels.cpu().numpy())
            
            action_accuracy = accuracy_score(all_action_labels_true, all_action_preds) if len(all_action_labels_true) > 0 else 0.0
            actor_accuracy = accuracy_score(all_actor_labels_true, all_actor_preds) if len(all_actor_labels_true) > 0 else 0.0
            
            print(f"Linear Probe Action Accuracy: {action_accuracy:.4f}")
            print(f"Linear Probe Actor Accuracy: {actor_accuracy:.4f}")

            metrics_data = {
                "dataset": args.dataset, "setting": args.setting,
                "temporal_masking_ratio": args.temporal_masking_ratio,
                "spatial_masking_ratio": args.spatial_masking_ratio,
                "best_pretrain_epoch": best_overall_epoch,
                "best_pretrain_validation_mse": best_overall_test_loss,
                "linear_probe_action_accuracy": action_accuracy,
                "linear_probe_actor_accuracy": actor_accuracy,
                "encoder_model_path": best_encoder_path_for_probe # Relative path from project root
            }
            # checkpoint_dir is already specific to the config
            metrics_file_path = os.path.join(checkpoint_dir, "probe_metrics.json") 
            with open(metrics_file_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            print(f"Saved probing metrics to {metrics_file_path}")

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()