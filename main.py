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

# Get dataset from argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ntu120', help='Dataset to use (ntu or ntu120)')
args = parser.parse_args()


init_seed(42)

# Parameters
# TODO: Change to argparse
cache_samples=True
save_samples=False
batch_size = 128
T = 64          # Number of frames (64)
M = 1           # Number of persons
V = 25          # Number of joints
setting = 'cs'  # 'cs' or 'cv'
dataset = args.dataset
lr = 1e-5
train_samples = 50000
test_samples = 5000
hpc = True

def main():
    global hpc
    print('in main')
    try:
        if hpc:
            print('Initializing distributed process group...')
            dist.init_process_group(backend='nccl')
            print('Process group initialized.')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.cuda.set_device(rank)
            device = torch.device('cuda', rank)
            print(f"Running on rank {rank}.")
        else:
            rank = 0
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Running on device {device}.")
        print(f'Rank: {rank}')
    except Exception as e:
        print(f"An error occurred: {e}")

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
            # Load data
            X = load_data(dataset, T)
            # Generate paired data and save to torch file
            paired_train, paired_test = get_cross_data(X, dataset, setting, batch_size, return_loader=False, train_samples=train_samples, test_samples=test_samples, threads=1)
            if save_samples:
                torch.save({'train': paired_train, 'test': paired_test}, paired_file_path)
                print('Paired data saved to torch file')
    else:
        # Load data
        X = load_data(dataset, T)
        paired_train, paired_test = get_cross_data(X, dataset, setting, batch_size, return_loader=False, train_samples=train_samples, test_samples=test_samples, threads=1)

    # Trim data to desired length
    paired_train.sampled_data = paired_train.sampled_data[:train_samples]
    paired_test.sampled_data = paired_test.sampled_data[:test_samples]

    if save_samples:
        with open(f'data/{dataset}_{setting}_paired.pkl', 'wb') as f:
            pickle.dump({'train': paired_train, 'test': paired_test}, f)
            print('Paired data saved to pickle file')

    # Create data loaders
    if hpc:
        train_sampler = torch.utils.data.distributed.DistributedSampler(paired_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(paired_test)
        train_loader = DataLoader(paired_train, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(paired_test, batch_size=batch_size, sampler=test_sampler)
    else:
        train_loader = DataLoader(paired_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(paired_test, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = Model(num_class=120 if dataset=='ntu120' else 60, num_point=V, num_person=M, graph='graph.ntu_rgb_d.Graph',
                graph_args={'labeling_mode': 'spatial'}, debug=False, dataset=dataset)
    model = model.to(device)

    if hpc:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Define optimizer
    # Only optimize parameters that require gradients (unfrozen parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Number of epochs
    num_epochs = 100

    # Create Trainer instance
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

    # Save model
    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')

    if hpc:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()