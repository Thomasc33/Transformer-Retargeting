from data import load_data, process_trainining_data, PT_Data, datasets
import os
import torch
from torch.utils.data import DataLoader
from model.ske_mixf import Model
from util import init_seed
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import random


# Turn these to argparse later
batch_size = 32
dataset = 'ntu120'
setting = 'cs'
lr = 1e-4
seed = 42
epochs = 100
T = 64
frame_masking_ratio = 0.3 
joint_masking_ratio = 0.3 
patience = 10
hpc = False

if dataset not in datasets:
    raise ValueError(f'Dataset {dataset} not found')

init_seed(seed)

if hpc:
    os.environ['MASTER_ADDR'] = 'localhost'  # or the master node's IP
    os.environ['MASTER_PORT'] = '13131' 
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    rank = 0

num_class = datasets[dataset]['num_class']
num_actor = datasets[dataset]['num_actor']
joints = datasets[dataset]['joints']
num_person = datasets[dataset]['max_actors']
graph = datasets[dataset]['graph']
graph_args = datasets[dataset]['graph_args']
channels = datasets[dataset]['channels']
load_dir = datasets[dataset]['path']

# Load or generate data
dataset_dir = f"data/{dataset}"
dataset_file = os.path.join(dataset_dir, "pretraining_data.pt")
os.makedirs(dataset_dir, exist_ok=True)
if os.path.exists(dataset_file):
    print(f"Loading dataset from {dataset_file}")
    saved_data = torch.load(dataset_file)
    train_dataset = saved_data['train_dataset']
    test_dataset = saved_data['test_dataset']
else:
    print(f"Generating dataset for {dataset}")
    X = load_data(dataset)
    train_x, train_y, test_x, test_y = process_trainining_data(X, setting=setting, dataset=dataset)
    train_dataset = PT_Data(train_x, train_y)
    test_dataset = PT_Data(test_x, test_y)
    if rank == 0:
        torch.save({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset
        }, dataset_file)
        print(f"Dataset saved to {dataset_file}")

if hpc:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"{dataset} loaded")

model = Model(num_class=num_class, num_actors=num_actor, num_point=joints, num_person=num_person, graph=graph, graph_args=graph_args, in_channels=channels)
if hpc:
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
else:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(1, epochs+1):
    if hpc: train_sampler.set_epoch(epoch)
    model.train()
    total_action_loss, total_actor_loss = 0, 0
    correct_action, correct_actor = 0, 0
    total_samples = 0

    # Training phase
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        action, actor = y[:, 0], y[:, 1]

        # Reshape input
        x = x.view(x.size(0), T, num_person, joints, channels).permute(0, 4, 1, 3, 2).contiguous()

        # Apply joint masking
        if joint_masking_ratio > 0:
            num_masked_joints = int(joints * joint_masking_ratio)
            joint_mask = torch.ones(joints, device=rank, dtype=torch.bool)
            mask_indices = torch.randperm(joints, device=rank)[:num_masked_joints]
            joint_mask[mask_indices] = False  # Set a percentage to False (masked)
            x[:, :, :, ~joint_mask, :] = 0  # Set masked joints to zero

        # Apply frame masking
        if frame_masking_ratio > 0:
            num_masked_frames = int(x.size(2) * frame_masking_ratio)
            frame_mask = torch.ones(x.size(2), device=rank, dtype=torch.bool)
            mask_indices = torch.randperm(x.size(2), device=rank)[:num_masked_frames]
            frame_mask[mask_indices] = False  # Set a percentage to False (masked)
            x[:, :, ~frame_mask, :, :] = 0  # Set masked frames to zero

        # Forward pass
        action_pred, actor_pred = model(x)

        # Calculate losses
        action_loss = criterion(action_pred, action)
        actor_loss = criterion(actor_pred, actor)
        loss = action_loss + actor_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_action_loss += action_loss.item()
        total_actor_loss += actor_loss.item()

        # Calculate accuracies
        _, action_pred_labels = torch.max(action_pred, 1)
        _, actor_pred_labels = torch.max(actor_pred, 1)
        
        correct_action += (action_pred_labels == action).sum().item()
        correct_actor += (actor_pred_labels == actor).sum().item()
        total_samples += action.size(0)

    scheduler.step()
    
    # Compute average losses and accuracies for the epoch
    avg_action_loss = total_action_loss / len(train_loader)
    avg_actor_loss = total_actor_loss / len(train_loader)
    action_accuracy = 100 * correct_action / total_samples
    actor_accuracy = 100 * correct_actor / total_samples

    if rank == 0:
        print(f'Epoch {epoch} - Train Action Loss: {avg_action_loss:.4f}, '
              f'Train Actor Loss: {avg_actor_loss:.4f}, '
              f'Train Action Accuracy: {action_accuracy:.2f}%, '
              f'Train Actor Accuracy: {actor_accuracy:.2f}%')

    # Validation phase
    model.eval()
    val_action_loss, val_actor_loss = 0, 0
    val_correct_action, val_correct_actor = 0, 0
    val_total_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            action, actor = y[:, 0], y[:, 1]

            # Reshape input
            x = x.view(x.size(0), T, num_person, joints, channels).permute(0, 4, 1, 3, 2).contiguous()
            action_pred, actor_pred = model(x)

            # Accumulate validation losses
            val_action_loss += criterion(action_pred, action).item()
            val_actor_loss += criterion(actor_pred, actor).item()

            # Calculate accuracies
            _, action_pred_labels = torch.max(action_pred, 1)
            _, actor_pred_labels = torch.max(actor_pred, 1)
            
            val_correct_action += (action_pred_labels == action).sum().item()
            val_correct_actor += (actor_pred_labels == actor).sum().item()
            val_total_samples += action.size(0)

    # Compute average losses and accuracies for validation
    avg_val_action_loss = val_action_loss / len(test_loader)
    avg_val_actor_loss = val_actor_loss / len(test_loader)
    val_action_accuracy = 100 * val_correct_action / val_total_samples
    val_actor_accuracy = 100 * val_correct_actor / val_total_samples

    if rank == 0:
        print(f'Epoch {epoch} - Val Action Loss: {avg_val_action_loss:.4f}, '
              f'Val Actor Loss: {avg_val_actor_loss:.4f}, '
              f'Val Action Accuracy: {val_action_accuracy:.2f}%, '
              f'Val Actor Accuracy: {val_actor_accuracy:.2f}%')

        # Save model if validation loss has improved
        if avg_val_action_loss + avg_val_actor_loss < best_val_loss:
            best_val_loss = avg_val_action_loss + avg_val_actor_loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), f"eval/mixformer/pretrained/{dataset}/ar_ri_best.pth")
            print(f"Model saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break