import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import wandb
from util import init_seed
from data import load_data, process_mlm, Masked_AE_Data
from model.encoder import Encoder, pre_process
from model.mlm_decoder import MLMDecoder, post_process

torch.backends.cudnn.enabled = False

# Parameters
dataset = 'ntu'
setting = 'cs'
T = 64
frame_masking_ratio = 0.5
joint_masking_ratio = 0.5
batch_size = 32
lr = 1e-4
seed = 42
epochs = 100
patience = 10
hpc = False  # Set to True when running on HPC cluster with multiple GPUs

# Initialize seed
init_seed(seed)

def main():
    global hpc

    if hpc:
        # Initialize the process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        print(f"Running on rank {rank}.")
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on device {device}.")

    # Initialize wandb (only on rank 0)
    if rank == 0:
        wandb.init(project=f'MLM_PT_{dataset}')

    # Data loading
    os.makedirs(f'data/{dataset}', exist_ok=True)
    if os.path.exists(f'data/{dataset}/pretraining_data.pt'):
        if rank == 0:
            print(f"Loading dataset from data/{dataset}/pretraining_data.pt")
        saved_data = torch.load(f'data/{dataset}/pretraining_data.pt', map_location='cpu')
        train_dataset = saved_data['train_dataset']
        test_dataset = saved_data['test_dataset']
    else:
        if rank == 0:
            print(f"Generating dataset for {dataset}")
        X = load_data(dataset, T=T)
        train_data, test_data = process_mlm(X, setting, dataset, T)
        train_dataset = Masked_AE_Data(torch.tensor([X[f] for f in train_data], dtype=torch.float32), frame_masking_ratio, joint_masking_ratio)
        test_dataset = Masked_AE_Data(torch.tensor([X[f] for f in test_data], dtype=torch.float32), frame_masking_ratio, joint_masking_ratio)
        if rank == 0:
            torch.save({
                'train_dataset': train_dataset,
                'test_dataset': test_dataset
            }, f'data/{dataset}/pretraining_data.pt')
            print(f"Saved dataset to data/{dataset}/pretraining_data.pt")

    # Data loaders with DistributedSampler for multi-GPU
    if hpc:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model definition
    class AE(nn.Module):
        def __init__(self):
            super(AE, self).__init__()
            self.encoder = Encoder(
                num_class=120 if dataset == 'ntu120' else 60,
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
            self.decoder = MLMDecoder(d_model=320, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1)
            self.output_layer = nn.Linear(320, 3)  # Map d_model to channels

        def forward(self, x):
            x = pre_process(x, x.size(0), T, 25, 3)
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.output_layer(x)  # Shape: (sequence_length, batch_size, channels)
            return post_process(x, T, x.size(1), 1, 25, 3)

    model = AE().to(device)

    # Wrap model with DDP if using HPC
    if hpc:
        model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        if hpc:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # Reshape data to match output
            data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Average train loss over all batches
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(device)
                output = model(data)
                # Reshape data to match output
                data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]
                loss = criterion(output, data)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        # Logging and printing (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})

        # Save model checkpoint
        if rank == 0:
            os.makedirs(f'eval/mixformer/pretrained/{dataset}/epochs', exist_ok=True)
            torch.save(model.encoder.state_dict(), f'eval/mixformer/pretrained/{dataset}/epochs/encoder_{epoch}.pth')

    # Saving model checkpoint (only on rank 0)
    if rank == 0:
        print(f"Saving model to eval/mixformer/pretrained/{dataset}/encoder.pth")
        os.makedirs(f'eval/mixformer/pretrained/{dataset}', exist_ok=True)
        torch.save(model.encoder.state_dict(), f'eval/mixformer/pretrained/{dataset}/encoder.pth')
        print("Model saved successfully")
        print("Exiting training loop")

    # Clean up (only if using distributed training)
    if hpc:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
