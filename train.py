import torch
from loss import Loss
import time

class Trainer:
    def __init__(self, model, optimizer,
                 train_paired_loader, val_paired_loader,
                 num_epochs=10, wandb_project=None, device='cuda', dataset='ntu', rank=0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_paired_loader = train_paired_loader
        self.val_paired_loader = val_paired_loader
        self.num_epochs = num_epochs
        self.wandb_project = wandb_project
        self.device = device
        self.dataset = dataset
        self.rank = rank

        # Example of how you might combine the new + old losses:
        # By default we have MSE, end-effector, smoothing, inception. 
        # We also turn ON fid_vel with weight=1, and keep bone & foot at 0 for demonstration.
        losses = {
            'mse': 7.0,
            'ee': 5.0,
            'smoothing': 0.075,
            'inception': 0.05,
            'fid_vel': 1.0,    
            'bone': 10.0,      
            'foot': 3.0,
            'joint_limit': 1.0    
        }

        # If you have a DDP model, the real "encoder" submodule is at model.module.encoder
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            encoder = model.module.encoder
        else:
            encoder = model.encoder

        self.loss = Loss(losses, device=device, dataset=dataset, encoder=encoder)

        # Optional: wandb
        if self.wandb_project is not None and self.rank == 0:
            config = {
                'lr': self.optimizer.param_groups[0]['lr'],
                'num_epochs': self.num_epochs,
                'train_samples': len(self.train_paired_loader),
                'val_samples': len(self.val_paired_loader)
            }
            import wandb
            wandb.init(project=self.wandb_project, config=config)
            wandb.watch(self.model)

    def train(self):
        if self.rank == 0:
            print("Starting Training (Autoregressive Motion Retargeting)...")
        self.model.train()
        for epoch in range(self.num_epochs):
            start = time.time()
            running_loss = 0.0
            running_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

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
                T = inputs.size(1)
                # D = inputs.size(2)  # not always used
                M = 1
                V = 25
                C_in = 3

                # Reshape data => (N, C, T, V, M)
                source_motion = inputs.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                dummy_skeleton = dummy.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                target_motion = targets.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Forward pass with teacher forcing
                output = self.model(source_motion, dummy_skeleton, target_motion=target_motion, teacher_forcing_ratio=1.0)

                # Compare output vs ground truth next frames
                # output => (N, C_in, T-1, V, M)
                # target => target_motion => (N, C_in, T, V, M)
                # so we align: ground truth next frames = target_motion[:, :, 1:, :, :]
                target = target_motion[:, :, 1:, :, :]

                loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])

                # Backprop
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                running_loss += loss_val.item()
                for key, value in losses_dict.items():
                    running_losses[key] += value.item()

            end = time.time()
            avg_loss = running_loss / len(self.train_paired_loader)
            epoch_losses = {key: value / len(self.train_paired_loader) for key, value in running_losses.items()}

            if self.rank == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
                print(f'Time taken: {end - start:.2f} s')
                print(epoch_losses)

            # Validation
            vstart = time.time()
            val_loss, val_losses = self.evaluate()
            vend = time.time()
            if self.rank == 0:
                print(f'Validation Time taken: {vend - vstart:.2f} s')

            # Log
            if self.wandb_project is not None and self.rank == 0:
                import wandb
                loss_info = {key: value for key, value in epoch_losses.items()}
                val_loss_info = {f"Val {key}": value for key, value in val_losses.items()}
                wandb.log({
                    'Loss': avg_loss,
                    'Val Loss': val_loss,
                    **loss_info,
                    **val_loss_info,
                    'Train Time': end - start,
                    'Val Time': vend - vstart
                })

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

        for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.val_paired_loader):
            x1 = x1.float().to(self.device)  # P1 A1
            x2 = x2.float().to(self.device)  # P2 A2
            y1 = y1.float().to(self.device)  # P1 A2
            y2 = y2.float().to(self.device)  # P2 A1

            # Combine inputs and targets
            inputs  = torch.cat([x1, y1, x2, y2], dim=0)
            dummy   = torch.cat([x2, y2, x1, y1], dim=0)
            targets = torch.cat([y2, x2, y1, x1], dim=0)

            N_total = inputs.size(0)
            T = inputs.size(1)
            M = 1
            V = 25
            C_in = 3

            source_motion = inputs.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
            dummy_skeleton = dummy.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
            target_motion = targets.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

            # Forward pass, no teacher forcing
            output = self.model(source_motion, dummy_skeleton, teacher_forcing_ratio=0.0)

            # GT next frames
            target = target_motion[:, :, 1:, :, :]

            loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])
            total_loss += loss_val.item()
            for key, val in losses_dict.items():
                losses[key] += val.item()

        avg_loss = total_loss / len(self.val_paired_loader)
        losses = {key: val / len(self.val_paired_loader) for key, val in losses.items()}

        if self.rank == 0:
            print(f'Validation Loss: {avg_loss:.4f}')
            print(losses)
        self.model.train()
        return avg_loss, losses
