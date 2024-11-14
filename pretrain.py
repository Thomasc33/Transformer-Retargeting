import torch
import os
from util import init_seed
from data import load_data, process_mlm, Masked_AE_Data
from model.encoder import Encoder, pre_process
from model.mlm_decoder import MLMDecoder, post_process
import torch.nn as nn

dataset='ntu120'
setting='cs'
T=64
frame_masking_ratio=0.5
joint_masking_ratio=0.5
batch_size=32
lr=1e-4
seed=42
epochs=100
patience=10
hpc=False

init_seed(seed)

os.makedirs(f'data/{dataset}', exist_ok=True)
if os.path.exists(f'data/{dataset}/pretraining_data.pt'):
    print(f"Loading dataset from data/{dataset}/pretraining_data.pt")
    saved_data = torch.load(f'data/{dataset}/pretraining_data.pt', weights_only=False)
    train_dataset = saved_data['train_dataset']
    test_dataset = saved_data['test_dataset']
else:    
    print(f"Generating dataset for {dataset}")
    X = load_data(dataset, T=T)
    train_dataset, test_dataset = process_mlm(X, 'cs', dataset, T)
    train_dataset = Masked_AE_Data(torch.tensor([X[f] for f in train_dataset], dtype=torch.float32), frame_masking_ratio, joint_masking_ratio)
    test_dataset = Masked_AE_Data(torch.tensor([X[f] for f in test_dataset], dtype=torch.float32), frame_masking_ratio, joint_masking_ratio)
    torch.save({
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }, f'data/{dataset}/pretraining_data.pt')
    print(f"Saved dataset to data/{dataset}/pretraining_data.pt")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class AE(torch.nn.Module):
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
        x = pre_process(x, batch_size, T, 25, 3)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)  # Shape: (sequence_length, batch_size, channels)
        return post_process(x, T, batch_size, 1, 25, 3)



model = AE().cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        # Reshape data to match output
        data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.cuda()
            output = model(data)
            # Reshape data to match output
            data = data.view(data.size(0), data.size(1), 25, 3).unsqueeze(2)  # [batch_size, frames, 1, 25, 3]
            loss = criterion(output, data)
            test_loss += loss.item()
        test_loss /= len(test_loader)

    print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    if epoch % patience == 0:
        print(f"Saving model to eval/mixformer/pretrained/{dataset}/ar.pth")
        torch.save(model.state_dict(), f'eval/mixformer/pretrained/{dataset}/ar.pth')
        print("Model saved successfully")
        print("Exiting training loop")
        break