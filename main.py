print('in main.py')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import pickle
import os
import json
import time
import logging
from datetime import datetime
from model.autoencoder import Model
# from train import Trainer  # Using OptimizedTrainer instead
from data import get_cross_data, load_data
from data_optimized import optimize_data_loading, estimate_memory_usage
from util import init_seed
import argparse
import warnings

# Suppress CUDNN warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings("ignore", message=".*CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

# Set CUDNN configuration for stability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_bool_env(varname, default=False):
    """
    Reads an environment variable (like 'true' / '1') as boolean.
    """
    val = os.environ.get(varname, str(default)).lower()
    return val in ['true', '1', 't', 'y', 'yes']

# Get dataset from argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ntu', help='Dataset to use (ntu or ntu120 or etri)')
parser.add_argument('--setting', type=str, default='cs', help='Evaluation setting (cs or cv)')
parser.add_argument('--hpc', action='store_true', help='Enable HPC distributed mode.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--train-samples', type=int, default=50000, help='Number of training samples to use')
parser.add_argument('--test-samples', type=int, default=5000, help='Number of test samples to use')
parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0,
                  help='Teacher forcing ratio (1.0=always use teacher forcing, 0.0=never use teacher forcing)')
parser.add_argument('--teacher-forcing-decay', type=float, default=0.0,
                  help='Teacher forcing decay rate per epoch (0.0=no decay)')
parser.add_argument('--data-path', type=str, default=None,
                  help='Path to paired data file. If not provided, will use default naming convention.')
parser.add_argument('--output-model-path', type=str, default='model.pth',
                  help='Path to save the trained model')
parser.add_argument('--run-eval', action='store_true', help='Run evaluation after training')

# Loss weight arguments
parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained encoder')
parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false', help='Do not use pretrained encoder')
parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder parameters')
parser.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false', help='Do not freeze encoder parameters')
parser.set_defaults(use_pretrained=True, freeze_encoder=True)

# Loss weight arguments
parser.add_argument('--loss-mse', type=float, default=7.0, help='Weight for MSE loss')
parser.add_argument('--loss-l1', type=float, default=0.0, help='Weight for L1 loss')
parser.add_argument('--loss-smoothl1', type=float, default=0.0, help='Weight for Smooth L1 loss')
parser.add_argument('--loss-kl', type=float, default=0.0, help='Weight for KL loss')
parser.add_argument('--loss-ce', type=float, default=0.0, help='Weight for Cross Entropy loss')
parser.add_argument('--loss-ee', type=float, default=5.0, help='Weight for End-Effector loss')
parser.add_argument('--loss-smoothing', type=float, default=0.075, help='Weight for Smoothing loss')
parser.add_argument('--loss-latent', type=float, default=0.0, help='Weight for Latent loss')
parser.add_argument('--loss-triplet', type=float, default=0.0, help='Weight for Triplet loss')
parser.add_argument('--loss-inception', type=float, default=0.05, help='Weight for Inception loss')
parser.add_argument('--loss-fid-vel', type=float, default=1.0, help='Weight for FID Velocity loss')
parser.add_argument('--loss-bone', type=float, default=10.0, help='Weight for Bone Length loss')
parser.add_argument('--loss-foot', type=float, default=3.0, help='Weight for Foot Contact loss')
parser.add_argument('--loss-joint-limit', type=float, default=1.0, help='Weight for Joint Limit loss')
parser.add_argument('--decoder-dropout', type=float, default=0.1, help='Dropout rate for the decoder')
parser.add_argument('--use-checkpoint', action='store_true', help='Use gradient checkpointing to save memory')
parser.add_argument('--no-checkpoint', dest='use_checkpoint', action='store_false', help='Do not use gradient checkpointing')
parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume training from')
parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
parser.add_argument('--mixed-precision', action='store_true', help='Use automatic mixed precision training')
parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients')
parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
parser.add_argument('--nccl-timeout', type=int, default=1800, help='NCCL timeout in seconds (default: 30 minutes)')
parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save training logs')
parser.set_defaults(use_checkpoint=True)
args = parser.parse_args()

# If not passing --hpc, we can also read from an env var
hpc = args.hpc or parse_bool_env("HPC_MODE", False)

init_seed(42)

# Parameters
cache_samples = True
save_samples = False
batch_size = args.batch_size
T = 64          # Number of frames (64)
M = 1           # Number of persons
V = 25          # Number of joints
setting = args.setting
dataset = args.dataset
lr = args.lr
train_samples = args.train_samples
test_samples = args.test_samples
teacher_forcing_ratio = args.teacher_forcing_ratio
teacher_forcing_decay = args.teacher_forcing_decay

def setup_logging(log_dir, rank):
    """Setup comprehensive logging system."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger, log_file
    else:
        return None, None

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, rank, is_best=False):
    """Save training checkpoint."""
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get model state dict (handle DDP case)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save epoch-specific checkpoint
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)

        # Save best checkpoint if specified
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint_path)

        return checkpoint_path
    return None

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training checkpoint."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dict (handle DDP case)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))

        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

def main():
    global hpc
    print('in main')

    # If HPC mode, read environment variables set by torchrun or srun
    # (One or more might be None if not set.)
    rank = 0
    world_size = 1
    local_rank = 0

    if hpc:
        # We check if WORLD_SIZE is set
        w = os.environ.get("WORLD_SIZE", None)
        if w is not None:
            world_size = int(w)
        # If you are using torchrun or the new torch.distributed.run script,
        # these should exist. Otherwise, you must export them manually in Slurm.
        r = os.environ.get("RANK", None)
        lrk = os.environ.get("LOCAL_RANK", None)
        if r is not None:
            rank = int(r)
        if lrk is not None:
            local_rank = int(lrk)

        # Set NCCL timeout
        os.environ['NCCL_TIMEOUT'] = str(args.nccl_timeout)
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # Use new env var name

        # Suppress CUDNN warnings
        os.environ['CUDNN_DETERMINISTIC'] = '1'
        os.environ['CUDNN_BENCHMARK'] = '0'

        # If world_size > 1, we do init_process_group
        if world_size > 1:
            print('Initializing distributed process group...')
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=torch.distributed.default_pg_timeout * 3  # Increase timeout
            )
            print('Process group initialized.')
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            print(f"[DDP Init] rank={rank}, local_rank={local_rank}, world_size={world_size}")
        else:
            # Single-process fallback
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Single-process HPC fallback. No distributed init.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running in non-HPC mode on device {device}.")

    # Setup logging
    logger, log_file = setup_logging(args.log_dir, rank)

    # Load data
    if cache_samples:
        # Try to load paired data from torch file
        if args.data_path:
            paired_file_path = args.data_path
        else:
            paired_file_path = f'data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt'

        # Check if file exists with new naming convention
        if not os.path.exists(paired_file_path):
            # Try legacy naming convention as fallback
            legacy_file_path = f'data/{dataset}_{setting}_paired.pt'
            if os.path.exists(legacy_file_path):
                print(f"Warning: Using legacy data file {legacy_file_path}")
                paired_file_path = legacy_file_path

        if os.path.exists(paired_file_path):
            print(f"Loading paired data from {paired_file_path}")
            paired_data = torch.load(paired_file_path)
            paired_train = paired_data['train']
            paired_test = paired_data['test']

            # Check if this is the comprehensive dataset
            if "comprehensive" in paired_file_path:
                print(f"Loaded COMPREHENSIVE paired dataset with {len(paired_train.sampled_data)} training and {len(paired_test.sampled_data)} test samples")
                print(f"Will sample {train_samples} training and {test_samples} test samples from this dataset")
            else:
                print(f"Loaded paired data with {len(paired_train.sampled_data)} training and {len(paired_test.sampled_data)} test samples")
        else:
            print(f"No paired data file found at {paired_file_path}, generating new data...")
            X = load_data(dataset, T)
            paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                       batch_size,
                                                       return_loader=False,
                                                       train_samples=train_samples,
                                                       test_samples=test_samples,
                                                       threads=1,
                                                       seg=T,
                                                       augment=True,
                                                       train_theta=0.3 if setting == 'cs' else 0.5)
            if save_samples:
                new_file_path = f'data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt'
                torch.save({'train': paired_train, 'test': paired_test}, new_file_path)
                print(f'Paired data saved to {new_file_path}')
    else:
        X = load_data(dataset, T)
        paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                   batch_size,
                                                   return_loader=False,
                                                   train_samples=train_samples,
                                                   test_samples=test_samples,
                                                   threads=1,
                                                   seg=T,
                                                   augment=True,
                                                   train_theta=0.3 if setting == 'cs' else 0.5,
                                                   val_theta=0.3 if setting == 'cs' else 0.5)

    # Check if we should use all data (very large sample count indicates "use all")
    use_all_train = train_samples >= 999999
    use_all_test = test_samples >= 999999

    # For training data
    if use_all_train:
        print(f"Using ALL {len(paired_train.sampled_data)} available training samples")
    elif len(paired_train.sampled_data) > train_samples:
        print(f"Trimming training data from {len(paired_train.sampled_data)} to {train_samples} samples")
        # Randomly sample to get a diverse subset
        import random
        random.seed(42)  # For reproducibility
        paired_train.sampled_data = random.sample(paired_train.sampled_data, train_samples)
    else:
        print(f"Using all {len(paired_train.sampled_data)} available training samples")

    # For test data
    if use_all_test:
        print(f"Using ALL {len(paired_test.sampled_data)} available test samples")
    elif len(paired_test.sampled_data) > test_samples:
        print(f"Trimming test data from {len(paired_test.sampled_data)} to {test_samples} samples")
        # Randomly sample to get a diverse subset
        import random
        random.seed(42)  # For reproducibility
        paired_test.sampled_data = random.sample(paired_test.sampled_data, test_samples)
    else:
        print(f"Using all {len(paired_test.sampled_data)} available test samples")

    if save_samples:
        import pickle
        with open(f'data/{dataset}_{setting}_paired.pkl', 'wb') as f:
            pickle.dump({'train': paired_train, 'test': paired_test}, f)
            print('Paired data saved to pickle file')

    # Estimate memory usage for optimization
    if rank == 0:
        memory_stats = estimate_memory_usage(paired_train, batch_size)
        print(f"Memory usage estimates:")
        print(f"  Average item size: {memory_stats['avg_item_size_mb']:.2f} MB")
        print(f"  Batch memory: {memory_stats['batch_memory_mb']:.2f} MB")
        print(f"  Estimated peak memory: {memory_stats['estimated_peak_memory_gb']:.2f} GB")

    # Create optimized data loaders
    train_loader, test_loader = optimize_data_loading(
        paired_train, paired_test, batch_size,
        distributed=(hpc and world_size > 1),
        rank=rank, world_size=world_size
    )

    # Initialize the model
    model = Model(num_class=120 if dataset == 'ntu120' else 60,
                  num_point=V, num_person=M, graph='graph.ntu_rgb_d.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  debug=False, dataset=dataset,
                  decoder_dropout=args.decoder_dropout,
                  use_checkpoint=args.use_checkpoint)

    # Load pretrained encoder weights if requested
    if args.use_pretrained:
        # Check for comprehensive pretrained encoder first
        comprehensive_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder_{setting}_comprehensive.pth'
        setting_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder_{setting}.pth'
        default_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder.pth'

        # Try to load in this order: comprehensive, setting-specific, default
        if os.path.exists(comprehensive_encoder_path):
            pretrained_encoder_path = comprehensive_encoder_path
            print(f"Loading COMPREHENSIVE pretrained encoder from {pretrained_encoder_path}")
        elif os.path.exists(setting_encoder_path):
            pretrained_encoder_path = setting_encoder_path
            print(f"Loading {setting} pretrained encoder from {pretrained_encoder_path}")
        elif os.path.exists(default_encoder_path):
            pretrained_encoder_path = default_encoder_path
            print(f"Loading default pretrained encoder from {pretrained_encoder_path}")
        else:
            pretrained_encoder_path = None
            print(f"Warning: No pretrained encoder found. Training encoder from scratch.")

        # Load the encoder if found
        if pretrained_encoder_path:
            # Load onto the correct device directly
            encoder_state_dict = torch.load(pretrained_encoder_path, map_location=device)
            model.encoder.load_state_dict(encoder_state_dict)
            print("Pretrained encoder loaded successfully.")
    else:
        print("Using randomly initialized encoder (no pretraining).")

    # Freeze encoder parameters if requested
    if args.freeze_encoder:
        print("Freezing encoder parameters...")
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")
    else:
        print("Encoder parameters will be trained (not frozen).")

    model = model.to(device) # Move model to device *after* loading state dict

    # --- Debug: Print all parameter names and requires_grad status ---
    print("\n--- Model Parameters & requires_grad Status ---")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        is_trainable = param.requires_grad
        if is_trainable:
            trainable_params += param.numel()
        # Only print on rank 0 to avoid clutter in distributed training
        if rank == 0:
            print(f"{name:<70} | Trainable: {is_trainable:<5} | Size: {param.numel()}")
    if rank == 0:
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters (reported by loop): {trainable_params}")
        print("---------------------------------------------\n")
    # --- End Debug ---

    # Apply gradient checkpointing to the model if requested
    if args.use_checkpoint:
        if rank == 0:
            print("Enabling gradient checkpointing for memory efficiency")
        # Enable gradient checkpointing for the encoder
        model.encoder.apply(lambda m: m.register_forward_hook(lambda module, _, output: output))

    # Wrap in DDP if multi-GPU AFTER loading and freezing
    if hpc and world_size > 1:
        # Use find_unused_parameters=True only if not using static graph
        use_static_graph = args.use_checkpoint
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=not use_static_graph)

        # Set static graph for DDP to handle gradient checkpointing
        if use_static_graph:
            if rank == 0:
                print("Setting static graph for DDP with gradient checkpointing")
            model._set_static_graph()

    # Define optimizer (only includes parameters where requires_grad is True)
    # Filter parameters *after* potential DDP wrapping if needed, though filtering before is usually fine.
    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Use more conservative optimizer settings to prevent NaN
    optimizer = optim.Adam(params_to_optimize, lr=lr, eps=1e-8, weight_decay=1e-6)

    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Print count from rank 0
    if rank == 0:
        print(f"Optimizer initialized with {len(params_to_optimize)} trainable parameter tensors.")

        # Initialize model weights more conservatively
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Use Xavier initialization with smaller scale
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Apply conservative initialization only to decoder (encoder is pretrained)
        if hasattr(model, 'decoder'):
            model.decoder.apply(init_weights)
        elif hasattr(model, 'module') and hasattr(model.module, 'decoder'):
            model.module.decoder.apply(init_weights)

    # Number of epochs
    num_epochs = args.epochs

    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')

    # Setup checkpoint directory
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')

    # Load checkpoint if resuming
    if args.resume_from:
        start_epoch, best_loss = load_checkpoint(args.resume_from, model, optimizer, device)

    # Initialize mixed precision scaler if using AMP
    scaler = None
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        if rank == 0:
            print("Using Automatic Mixed Precision (AMP) training")

    # Create Trainer instance
    from train_optimized import OptimizedTrainer

    # Configure loss weights from arguments
    loss_weights = {}
    # Only add losses with non-zero weights
    if args.loss_mse > 0: loss_weights['mse'] = args.loss_mse
    if args.loss_l1 > 0: loss_weights['l1'] = args.loss_l1
    if args.loss_smoothl1 > 0: loss_weights['smoothl1'] = args.loss_smoothl1
    if args.loss_kl > 0: loss_weights['kl'] = args.loss_kl
    if args.loss_ce > 0: loss_weights['ce'] = args.loss_ce
    if args.loss_ee > 0: loss_weights['ee'] = args.loss_ee
    if args.loss_smoothing > 0: loss_weights['smoothing'] = args.loss_smoothing
    if args.loss_latent > 0: loss_weights['latent'] = args.loss_latent
    if args.loss_triplet > 0: loss_weights['triplet'] = args.loss_triplet
    if args.loss_inception > 0: loss_weights['inception'] = args.loss_inception
    if args.loss_fid_vel > 0: loss_weights['fid_vel'] = args.loss_fid_vel
    if args.loss_bone > 0: loss_weights['bone'] = args.loss_bone
    if args.loss_foot > 0: loss_weights['foot'] = args.loss_foot
    if args.loss_joint_limit > 0: loss_weights['joint_limit'] = args.loss_joint_limit

    # Print loss weights for debugging
    if rank == 0:
        print("\nUsing loss weights:")
        for k, v in loss_weights.items():
            print(f"  {k}: {v}")
        print()

    # Determine if we're using all data
    use_all_data = train_samples >= 999999 or test_samples >= 999999

    # Set wandb project name
    wandb_project = 'Motion Retargeting'
    if "comprehensive" in args.data_path:
        wandb_project += ' Comprehensive'
        if use_all_data:
            wandb_project += ' All'

    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        train_paired_loader=train_loader,
        val_paired_loader=test_loader,
        num_epochs=num_epochs,
        wandb_project=wandb_project,
        device=device,
        dataset=dataset,
        rank=rank,
        teacher_forcing_ratio=teacher_forcing_ratio,
        teacher_forcing_decay=teacher_forcing_decay,
        loss_weights=loss_weights,
        start_epoch=start_epoch,
        best_loss=best_loss,
        checkpoint_dir=checkpoint_dir,
        save_every=args.save_every,
        mixed_precision=args.mixed_precision,
        scaler=scaler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        scheduler=scheduler
    )

    # Train
    trainer.train()

    # Save model (only do so on rank 0)
    if (not hpc) or (hpc and rank == 0):
        # Determine if we're using all data
        use_all_data = train_samples >= 999999 or test_samples >= 999999

        # Set output model path
        output_model_path = args.output_model_path

        # If using all data and the path doesn't already include "all", add it
        if use_all_data and "all" not in output_model_path.lower():
            # Insert "all" before the file extension
            base, ext = os.path.splitext(output_model_path)
            output_model_path = f"{base}_all{ext}"

        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")

        # Run evaluation if requested
        if args.run_eval:
            print("\nRunning evaluation...")
            import subprocess
            eval_cmd = [
                'python', 'eval_model.py',
                f'--dataset={dataset}',
                f'--setting={setting}',
                f'--model_type=transformer',
                f'--transformer_model_path={output_model_path}',
                f'--eval_model=sgn',
                f'--test_samples={test_samples}'
            ]
            print(f"Executing: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd)

    # Cleanup
    if hpc and world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
