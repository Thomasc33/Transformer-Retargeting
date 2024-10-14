import torch

class Trainer:
    def __init__(self, model, optimizer, criterion,
                 train_paired_loader, val_paired_loader,
                 num_epochs=10, wandb_project=None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_paired_loader = train_paired_loader
        self.val_paired_loader = val_paired_loader
        self.num_epochs = num_epochs
        self.wandb_project = wandb_project
        self.device = device

        if self.wandb_project is not None:
            config = {
                'lr': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.train_paired_loader.batch_size,
                'num_epochs': self.num_epochs,
                'train_samples': len(self.train_paired_loader.dataset),
                'val_samples': len(self.val_paired_loader.dataset)
            }
            import wandb
            wandb.init(project=self.wandb_project, config=config)
            wandb.watch(self.model)

    def train(self):
        print("Starting Training (Autoregressive Motion Retargeting)...")
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.train_paired_loader):
                x1 = x1.float().to(self.device)  # P1 A1
                x2 = x2.float().to(self.device)  # P2 A2
                y1 = y1.float().to(self.device)  # P1 A2
                y2 = y2.float().to(self.device)  # P2 A1

                # Combine inputs and targets using the four combinations
                inputs  = torch.cat([x1, y1, x2, y2], dim=0)
                dummy   = torch.cat([x2, y2, x1, y1], dim=0)
                targets = torch.cat([y2, x2, y1, x1], dim=0)

                N_total = inputs.size(0)
                N = N_total
                T = inputs.size(1)
                D = inputs.size(2)
                M = 1
                V = 25
                C_in = 3

                # Reshape data
                source_motion = inputs.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                dummy_skeleton = dummy.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                target_motion = targets.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Forward pass with teacher forcing
                output = self.model(source_motion, dummy_skeleton, target_motion=target_motion, teacher_forcing_ratio=1.0)

                # Compute loss between predicted frames and ground truth next frames
                # output: (N, C_in, T-1, V, M)
                # target_motion: (N, C_in, T, V, M)
                target = target_motion[:, :, 1:, :, :]  # Ground truth next frames

                loss = self.criterion(output, target)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_paired_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')

            # Evaluate on validation data
            val_loss = self.evaluate()

            if self.wandb_project is not None:
                import wandb
                wandb.log({'Loss': avg_loss, 'Val Loss': val_loss})

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.val_paired_loader):
                x1 = x1.float().to(self.device)  # P1 A1
                x2 = x2.float().to(self.device)  # P2 A2
                y1 = y1.float().to(self.device)  # P1 A2
                y2 = y2.float().to(self.device)  # P2 A1

                # Combine inputs and targets using the four combinations
                inputs  = torch.cat([x1, y1, x2, y2], dim=0)
                dummy   = torch.cat([x2, y2, x1, y1], dim=0)
                targets = torch.cat([y2, x2, y1, x1], dim=0)

                N_total = inputs.size(0)
                N = N_total
                T = inputs.size(1)
                D = inputs.size(2)
                M = 1
                V = 25
                C_in = 3

                # Reshape data
                source_motion = inputs.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                dummy_skeleton = dummy.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                target_motion = targets.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Forward pass without teacher forcing (autoregressive generation)
                output = self.model(source_motion, dummy_skeleton, teacher_forcing_ratio=0.0)

                # Compute loss
                target = target_motion[:, :, 1:, :, :]  # Ground truth next frames

                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_paired_loader)
        print(f'Validation Loss: {avg_loss:.4f}')
        self.model.train()
        return avg_loss
