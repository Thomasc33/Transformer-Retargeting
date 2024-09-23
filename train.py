import torch

class Trainer:
    def __init__(self, model, optimizer, criterion,
                 train_loader, val_loader,
                 train_paired_loader, val_paired_loader,
                 num_epochs_stage1=10, num_epochs_stage2=10, 
                 wandb_project = None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader                    # train_x for stage 1
        self.val_loader = val_loader                        # test_x for stage 1
        self.train_paired_loader = train_paired_loader      # train_paired_x for stage 2
        self.val_paired_loader = val_paired_loader          # test_paired_x for stage 2
        self.num_epochs_stage1 = num_epochs_stage1
        self.num_epochs_stage2 = num_epochs_stage2
        self.wandb_project = wandb_project
        self.device = device

        if self.wandb_project is not None:
            config = {
                'lr': self.optimizer.param_groups[0]['lr'],
                'batch_size': self.train_loader.batch_size,
                'num_epochs_stage1': self.num_epochs_stage1,
                'num_epochs_stage2': self.num_epochs_stage2,
                'paired_train_samples': len(self.train_paired_loader.dataset),
                'paired_val_samples': len(self.val_paired_loader.dataset)
            }
            import wandb
            wandb.init(project=self.wandb_project, config=config)
            wandb.watch(self.model)

    def train_stage1(self):
        print("Starting Stage 1 Training (Reconstruction with Masking)...")
        self.model.train()
        for epoch in range(self.num_epochs_stage1):
            running_loss = 0.0
            for batch_idx, (data, actors, actions) in enumerate(self.train_loader):
                data = data.float().to(self.device)
                # Apply masking
                masked_data, target = self.mask_data(data)
                # Reshape data as per the model's requirements
                N = masked_data.shape[0]
                T = masked_data.shape[1]
                D = masked_data.shape[2]
                M = 1           # Number of persons
                V = 25          # Number of joints
                C_in = 3        # Number of input channels

                # Ensure that M * V * C_in == D
                assert M * V * C_in == D, "Mismatch in dimensions"

                # Reshape data
                x = masked_data.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()  # (N, C_in, T, V, M)

                # Forward pass
                output = self.model(x)
                # Reshape output to match target
                output = output.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C_in)
                output = output.view(N, T, D)  # (N, T, D)

                # Compute loss
                loss = self.criterion(output, target)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f'Stage 1 Epoch [{epoch+1}/{self.num_epochs_stage1}], Loss: {avg_loss:.4f}')

            # Evaluate on validation data
            val_loss = self.evaluate_stage1(self.val_loader)

            if self.wandb_project is not None:
                import wandb
                wandb.log({'Stage 1 Loss': avg_loss, 'Stage 1 Val Loss': val_loss})


    def mask_data(self, data):
        # data: (batch_size, T, D)
        N, T, D = data.shape
        M = 1
        V = 25
        C_in = 3

        data = data.view(N, T, M, V, C_in)  # (N, T, M, V, C_in)

        # Create masks
        frame_mask_prob = 0.15  # Probability of masking a frame
        joint_mask_prob = 0.15  # Probability of masking a joint

        # Frame mask: mask entire frames
        frame_mask = (torch.rand(N, T, 1, 1, 1, device=self.device) < frame_mask_prob)  # (N, T, 1, 1, 1)

        # Joint mask: mask individual joints
        joint_mask = (torch.rand(N, T, M, V, 1, device=self.device) < joint_mask_prob)  # (N, T, M, V, 1)

        # Combine masks
        mask = frame_mask | joint_mask  # Logical OR to combine masks

        masked_data = data.masked_fill(mask, 0)  # Apply mask

        # Reshape masked_data back to (N, T, D)
        masked_data = masked_data.view(N, T, D)

        # Target is the original data reshaped to (N, T, D)
        target = data.view(N, T, D)

        return masked_data, target

    def train_stage2(self):
        print("Starting Stage 2 Training (Motion Retargeting)...")
        self.model.train()
        for epoch in range(self.num_epochs_stage2):
            running_loss = 0.0
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.train_paired_loader):
                x1 = x1.float().to(self.device)  # P1 A1
                x2 = x2.float().to(self.device)  # P2 A2
                y1 = y1.float().to(self.device)  # P1 A2
                y2 = y2.float().to(self.device)  # P2 A1

                # Combine inputs and targets
                inputs  = torch.cat([x1, y1, x2, y2], dim=0)    # Inputs:  P1 A1, P1 A2, P2 A2, P2 A1
                dummy   = torch.cat([x2, y2, x1, y1], dim=0)    # Dummy:   P2 A2, P2 A1, P1 A1, P1 A2
                targets = torch.cat([y2, x2, y1, x1], dim=0)    # Targets: P2 A1, P2 A2, P1 A2, P1 A1

                N = inputs.shape[0]
                T = inputs.shape[1]
                D = inputs.shape[2]
                M = 1
                V = 25
                C_in = 3

                # Ensure that M * V * C_in == D
                assert M * V * C_in == D, "Mismatch in dimensions"

                # Reshape inputs
                x = inputs.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()  # (N, C_in, T, V, M)
                x_dummy = dummy.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Forward pass
                output = self.model(x, x_dummy)
                # Reshape output to match targets
                output = output.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C_in)
                output = output.view(N, T, D)  # (N, T, D)

                # Compute loss
                loss = self.criterion(output, targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_paired_loader)
            print(f'Stage 2 Epoch [{epoch+1}/{self.num_epochs_stage2}], Loss: {avg_loss:.4f}')

            # Optionally evaluate on validation data
            val_loss = self.evaluate_stage2(self.val_paired_loader)

            if self.wandb_project is not None:
                import wandb
                wandb.log({'Stage 2 Loss': avg_loss, 'Stage 2 Val Loss': val_loss})

    def evaluate_stage1(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, actors, actions) in enumerate(data_loader):
                data = data.float().to(self.device)
                # No masking during evaluation
                target = data.clone()
                N = data.shape[0]
                T = data.shape[1]
                D = data.shape[2]
                M = 1
                V = 25
                C_in = 3

                # Reshape data
                x = data.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()  # (N, C_in, T, V, M)

                # Forward pass
                output = self.model(x)
                # Reshape output to match target
                output = output.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C_in)
                output = output.view(N, T, D)  # (N, T, D)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Stage 1 Validation Loss: {avg_loss:.4f}')
        self.model.train()
        return avg_loss

    def evaluate_stage2(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(data_loader):
                x1 = x1.float().to(self.device)  # P1 A1
                x2 = x2.float().to(self.device)  # P2 A2
                y1 = y1.float().to(self.device)  # P1 A2
                y2 = y2.float().to(self.device)  # P2 A1

                # Combine inputs and targets
                # input is the action, dummy is the actor
                inputs  = torch.cat([x1, y1, x2, y2], dim=0)    # Inputs:  P1 A1, P1 A2, P2 A2, P2 A1
                dummy   = torch.cat([x2, y2, x1, y1], dim=0)    # Dummy:   P2 A2, P2 A1, P1 A1, P1 A2
                targets = torch.cat([y2, x2, y1, x1], dim=0)    # Targets: P2 A1, P2 A2, P1 A2, P1 A1

                N = inputs.shape[0]
                T = inputs.shape[1]
                D = inputs.shape[2]
                M = 1
                V = 25
                C_in = 3

                # Reshape inputs
                x = inputs.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()  # (N, C_in, T, V, M)
                x_dummy = dummy.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Forward pass
                output = self.model(x, x_dummy)
                # Reshape output to match targets
                output = output.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C_in)
                output = output.view(N, T, D)  # (N, T, D)

                # Compute loss
                loss = self.criterion(output, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Stage 2 Validation Loss: {avg_loss:.4f}')
        self.model.train()
        return avg_loss
