from data import load_data, process_trainining_data, PT_Data, datasets
import os
import torch
from torch.utils.data import DataLoader
from model.ske_mixf import Model
from util import init_seed
from tqdm import tqdm

# Turn these to argparse later
batch_size = 32
dataset = 'ntu120'
setting = 'cs'
lr = 1e-4
seed = 42
epochs = 100
T = 64

if dataset not in datasets:
    raise ValueError(f'Dataset {dataset} not found')

init_seed(seed)

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
    torch.save({
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }, dataset_file)
    print(f"Dataset saved to {dataset_file}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f"{dataset} loaded")

model = Model(num_class=num_class, num_actors=num_actor, num_point=joints, num_person=num_person, graph=graph, graph_args=graph_args, in_channels=channels).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, epochs+1):
    model.train()
    total_action_loss, total_actor_loss = 0, 0
    correct_action, correct_actor = 0, 0
    total_samples = 0

    # Training phase
    for x, y in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        action, actor = y[:, 0], y[:, 1]

        # Reshape input
        x = x.view(batch_size, T, num_person, joints, channels).permute(0, 4, 1, 3, 2).contiguous()
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
    
    # Compute average losses and accuracies for the epoch
    avg_action_loss = total_action_loss / len(train_loader)
    avg_actor_loss = total_actor_loss / len(train_loader)
    action_accuracy = 100 * correct_action / total_samples
    actor_accuracy = 100 * correct_actor / total_samples

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
        for x, y in tqdm(test_loader, desc=f'Validation Epoch {epoch}'):
            x, y = x.cuda(), y.cuda()
            action, actor = y[:, 0], y[:, 1]

            # Reshape input
            x = x.view(batch_size, T, num_person, joints, channels).permute(0, 4, 1, 3, 2).contiguous()
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

    print(f'Epoch {epoch} - Val Action Loss: {avg_val_action_loss:.4f}, '
          f'Val Actor Loss: {avg_val_actor_loss:.4f}, '
          f'Val Action Accuracy: {val_action_accuracy:.2f}%, '
          f'Val Actor Accuracy: {val_actor_accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), f"eval/mixformer/pretrained/{dataset}/ar_ri.pth")
