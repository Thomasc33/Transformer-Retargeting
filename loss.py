import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss():
    def __init__(self, loss_weights, device = 'cuda', dataset = 'ntu'):
        # Toggles for losses
        self.mse = 'mse' in loss_weights
        self.l1 = 'l1' in loss_weights
        self.smoothl1 = 'smoothl1' in loss_weights
        self.kl = 'kl' in loss_weights
        self.ce = 'ce' in loss_weights
        self.ee = 'ee' in loss_weights
        self.smoothing = 'smoothing' in loss_weights
        self.latent = 'latent' in loss_weights
        self.triplet = 'triplet' in loss_weights

        # Loss weights
        self.loss_weights = loss_weights

        self.device = device
        self.dataset = dataset

    def mse_loss(self, output, target):
        return F.mse_loss(output, target)
    
    def l1_loss(self, output, target):
        return F.l1_loss(output, target)
    
    def smoothl1_loss(self, output, target):
        return F.smooth_l1_loss(output, target)
    
    def kl_loss(self, output, target):
        return F.kl_div(output, target)
    
    def ce_loss(self, output, target):
        return F.cross_entropy(output, target)
    
    def ee_loss(self, output, target):
        # TODO: Adapt this to the new model
        assert self.dataset in ee_chains, f"Dataset {self.dataset} not supported for end effector loss"
        assert self.dataset in ee_chain_lengths, f"Dataset {self.dataset} not supported for end effector loss"

        # Remove the actors dimension (assuming it's always 1)
        output = output.squeeze(-1)  # Shape is now (channels, frames, joints)
        target = target.squeeze(-1)  # Shape is now (channels, frames, joints)

        # Select the channels for end effectors based on `ee_chains`
        # The new shape requires adjusting the index selection to use the correct axis
        x_ee = output[ee_chains[self.dataset], :, :]  # End effector channels from the output
        y_ee = target[ee_chains[self.dataset], :, :]  # End effector channels from the target

        # Calculate velocities by taking differences along the frames dimension
        x_vel = torch.norm(x_ee[:, 1:] - x_ee[:, :-1], dim=0) / ee_chain_lengths[self.dataset].unsqueeze(0)
        y_vel = torch.norm(y_ee[:, 1:] - y_ee[:, :-1], dim=0) / ee_chain_lengths[self.dataset].unsqueeze(0)

        # Compute MSE loss for each joint, reduce as needed
        losses = F.mse_loss(x_vel, y_vel, reduction='none')
        loss = losses.sum(dim=0).mean()  # Sum over end effectors and average over the batch

        return loss
    
    def smoothing_loss(self, output, target):
        # Remove the actors dimension as it's always 1
        output = output.squeeze(-1)  # Shape becomes (channels, frames, joints)
        target = target.squeeze(-1)  # Shape becomes (channels, frames, joints)

        # Calculate squared sum of differences for target and output along frames
        diff_y = torch.sum((target[:, 1:, :] - target[:, :-1, :]) ** 2, dim=1)  # Shape: (channels, joints)
        diff_y_pred = torch.sum((output[:, 1:, :] - output[:, :-1, :]) ** 2, dim=1)  # Shape: (channels, joints)

        # Calculate absolute difference and sum over channels and joints
        abs_diff = torch.abs(diff_y - diff_y_pred)
        loss = abs_diff.sum()  # Sum over channels and joints

        # Normalize by the total number of joints * frames
        total_loss = torch.sqrt(loss) / (target.size(1) * target.size(2))

        return total_loss


    def latent_loss(self, output, target):
        pass

    def triplet_loss(self, output, target, input):
        return F.triplet_margin_loss(output, target, input)
        
    def loss(self, output, target, input):
        loss = 0
        losses = {key: 0 for key in self.loss_weights}
        if self.mse:
            losses['mse'] = self.mse_loss(output, target)
        if self.l1:
            losses['l1'] = self.l1_loss(output, target)
        if self.smoothl1:
            losses['smoothl1'] = self.smoothl1_loss(output, target)
        if self.kl:
            losses['kl'] = self.kl_loss(output, target)
        if self.ce:
            losses['ce'] = self.ce_loss(output, target)
        if self.ee:
            losses['ee'] = self.ee_loss(output, target)
        if self.smoothing:
            losses['smoothing'] = self.smoothing_loss(output, target)
        if self.latent:
            losses['latent'] = self.latent_loss(output, target)
        if self.triplet:
            losses['triplet'] = self.triplet_loss(output, target, input)
        loss = sum([losses[key] * self.loss_weights[key] for key in self.loss_weights])
        return loss, losses
    

ee_chains = {
    'ntu': torch.tensor([19, 15, 23, 24, 21, 22, 3]) * 3,
    'ntu120': torch.tensor([19, 15, 23, 24, 21, 22, 3]) * 3,
}

ee_chain_lengths = {
    'ntu': torch.tensor([5, 5, 8, 8, 8, 8, 5]),
    'ntu120': torch.tensor([5, 5, 8, 8, 8, 8, 5]),
}