print('in main.py')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import pickle
import os
from model.autoencoder import Model
from train import Trainer
from data import get_cross_data, load_data
from util import init_seed
import argparse

def parse_bool_env(varname, default=False):
    """
    Reads an environment variable (like 'true' / '1') as boolean.
    """
    val = os.environ.get(varname, str(default)).lower()
    return val in ['true', '1', 't', 'y', 'yes']

# Get dataset from argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ntu', help='Dataset to use (ntu or ntu120 or etri)')
parser.add_argument('--hpc', action='store_true', help='Enable HPC distributed mode.')
args = parser.parse_args()

# If not passing --hpc, we can also read from an env var
hpc = args.hpc or parse_bool_env("HPC_MODE", False)

init_seed(42)

# Parameters
cache_samples = True
save_samples = False
batch_size = 128
T = 64          # Number of frames (64)
M = 1           # Number of persons
V = 25          # Number of joints
setting = 'cs'  # 'cs' or 'cv'
dataset = args.dataset
lr = 1e-5
train_samples = 50000
test_samples = 5000

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

        # If world_size > 1, we do init_process_group
        if world_size > 1:
            print('Initializing distributed process group...')
            dist.init_process_group(backend='nccl', init_method='env://')
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

    # Load data
    if cache_samples:
        # Try to load paired data from torch file
        paired_file_path = f'data/{dataset}_{setting}_paired.pt'
        if os.path.exists(paired_file_path):
            paired_data = torch.load(paired_file_path)
            paired_train = paired_data['train']
            paired_test = paired_data['test']
            print('Paired data loaded from torch file')
        else:
            X = load_data(dataset, T)
            paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                       batch_size,
                                                       return_loader=False,
                                                       train_samples=train_samples,
                                                       test_samples=test_samples,
                                                       threads=1)
            if save_samples:
                torch.save({'train': paired_train, 'test': paired_test}, paired_file_path)
                print('Paired data saved to torch file')
    else:
        X = load_data(dataset, T)
        paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                   batch_size,
                                                   return_loader=False,
                                                   train_samples=train_samples,
                                                   test_samples=test_samples,
                                                   threads=1)

    # Trim data to desired length
    paired_train.sampled_data = paired_train.sampled_data[:train_samples]
    paired_test.sampled_data = paired_test.sampled_data[:test_samples]

    if save_samples:
        import pickle
        with open(f'data/{dataset}_{setting}_paired.pkl', 'wb') as f:
            pickle.dump({'train': paired_train, 'test': paired_test}, f)
            print('Paired data saved to pickle file')

    # Create data loaders
    if hpc and world_size > 1:
        # We only use DistributedSampler if we actually have multiple processes
        train_sampler = torch.utils.data.distributed.DistributedSampler(paired_train,
                                                                        num_replicas=world_size,
                                                                        rank=rank,
                                                                        shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(paired_test,
                                                                       num_replicas=world_size,
                                                                       rank=rank,
                                                                       shuffle=False)
        train_loader = DataLoader(paired_train, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(paired_test, batch_size=batch_size, sampler=test_sampler)
    else:
        # Single GPU or CPU
        train_loader = DataLoader(paired_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(paired_test, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Model(num_class=120 if dataset == 'ntu120' else 60,
                  num_point=V, num_person=M, graph='graph.ntu_rgb_d.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  debug=False, dataset=dataset)
    model = model.to(device)

    # Wrap in DDP if multi-GPU
    if hpc and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Number of epochs
    num_epochs = 100

    # Create Trainer instance
    from train import Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_paired_loader=train_loader,
        val_paired_loader=test_loader,
        num_epochs=num_epochs,
        wandb_project='Motion Retargeting',
        device=device,
        dataset=dataset,
        rank=rank
    )

    # Train
    trainer.train()

    # Save model (only do so on rank 0)
    if (not hpc) or (hpc and rank == 0):
        torch.save(model.state_dict(), 'model.pth')
        print("Model saved to model.pth")

    # Cleanup
    if hpc and world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
