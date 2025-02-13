import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Loss():
    def __init__(self, loss_weights, device='cuda', dataset='ntu', encoder=None):
        """
        This class implements various losses for motion generation and retargeting.

        :param loss_weights: dict specifying which losses are used and their weights.
                             Example: {'mse': 1.0, 'ee': 1.0, 'fid_vel': 0.5, ...}
        :param device:       'cuda' or 'cpu'
        :param dataset:      'ntu', 'ntu120', 'etri'
        :param encoder:      Optional encoder network (for inception-like loss)
        """

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
        self.inception = 'inception' in loss_weights

        # Newly added toggles
        self.fid_vel = 'fid_vel' in loss_weights       # FID on velocities
        self.bone = 'bone' in loss_weights             # Bone-length loss
        self.foot = 'foot' in loss_weights             # Foot-contact loss

        # Loss weights dictionary
        self.loss_weights = loss_weights

        self.device = device
        self.dataset = dataset
        self.encoder = encoder

    # -----------------
    # Standard losses
    # -----------------
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

    # -----------------
    # End-effector loss
    # -----------------
    def ee_loss(self, output, target):
        """
        An end-effector velocity loss for NTU/ETRI style skeletons.
        Normalizes by chain lengths, measuring how different the end-effector 
        velocities are from target velocities.
        Reference: sometimes used in motion-retargeting tasks to ensure the 
        extremities (feet, hands, head) track well.
        """
        assert self.dataset in ee_chains, f"Dataset {self.dataset} not supported for end effector loss"
        assert self.dataset in ee_chain_lengths, f"Dataset {self.dataset} not supported for end effector loss"

        # (N, C, T, V, M) => remove M if M=1
        output = output.squeeze(-1)  # (N, C, T, V)
        target = target.squeeze(-1)  # (N, C, T, V)

        # End-effector indices
        ee_idx = ee_chains[self.dataset]
        chain_lens = ee_chain_lengths[self.dataset].to(self.device)

        # Subselect just the end-effector joints
        x_ee = output[:, :, :, ee_idx]  # (N, C, T, len(ee_idx))
        y_ee = target[:, :, :, ee_idx]

        # Calculate velocities by taking differences along the T dimension
        # => shape becomes (N, C, T-1, len(ee_idx))
        x_vel = x_ee[:, :, 1:, :] - x_ee[:, :, :-1, :]
        y_vel = y_ee[:, :, 1:, :] - y_ee[:, :, :-1, :]

        # Norm across channels dimension, since C=3 usually
        # => shape (N, T-1, len(ee_idx))
        x_vel_norm = torch.norm(x_vel, dim=1) / chain_lens.unsqueeze(0).unsqueeze(1)
        y_vel_norm = torch.norm(y_vel, dim=1) / chain_lens.unsqueeze(0).unsqueeze(1)

        # MSE across those velocities
        losses = (x_vel_norm - y_vel_norm).pow(2)
        # sum over end-effectors, average over batch & time
        loss = losses.mean()

        return loss

    # -----------------
    # Smoothing loss
    # -----------------
    def smoothing_loss(self, output, target):
        """
        Encourages smooth transitions, comparing frame-to-frame differences 
        in the target vs. the predicted output.
        """
        # (N, C, T, V, M) => remove M
        output = output.squeeze(-1)  # (N, C, T, V)
        target = target.squeeze(-1)  # (N, C, T, V)

        # Compute squared differences along time dimension
        diff_y = torch.sum((target[:, :, 1:, :] - target[:, :, :-1, :]) ** 2, dim=1)  # (N, T-1, V)
        diff_y_pred = torch.sum((output[:, :, 1:, :] - output[:, :, :-1, :]) ** 2, dim=1)

        # L1 difference of these sums => shape (N, T-1, V)
        abs_diff = torch.abs(diff_y - diff_y_pred)
        # sum over all, then average
        loss = abs_diff.sum()  
        total_loss = torch.sqrt(loss) / (target.size(1) * target.size(3))  # Normalization

        return total_loss

    # -----------------
    # Latent / Triplet placeholders
    # -----------------
    def latent_loss(self, output, target):
        """
        Placeholder for a style-latent or content-latent loss.
        Could compare latent distributions of the motion. 
        Not implemented.
        """
        return torch.tensor(0.0, device=self.device)

    def triplet_loss(self, output, target, input):
        """
        Placeholder for a standard triplet margin loss.
        Typically used if you have anchor/positive/negative motions 
        to enforce a margin in latent space. 
        Not implemented.
        """
        return torch.tensor(0.0, device=self.device)
    
    # -----------------
    # Inception-like loss
    # -----------------
    @torch.no_grad()
    def inception_loss(self, output, target):
        """
        Compares the features from self.encoder(output) and self.encoder(target).
        This is like a naive version of 'inception distance' for motion, 
        but we rely on the 'encoder' to extract features.
        """
        # Expand time dimension (fake) for alignment if needed
        output = torch.cat([output, torch.zeros_like(output[:, :, :1, :, :])], dim=2)
        target = torch.cat([target, torch.zeros_like(target[:, :, :1, :, :])], dim=2)

        # Compare feature embeddings
        return F.mse_loss(self.encoder(output), self.encoder(target))

    def fid_velocity_loss(self, output, target):
        """
        Approximate 'FID' style distance on joint velocities.
        - We compute velocity = difference along T dimension
        - Flatten across batch*N, frames*(T-1), joints*V => get Nx3 columns
        - Compute mean, covariance for real (target) and generated (output)
        - Compute Frechet Distance
        This is an adaptation for skeletal data, referencing the standard FID formula
        from the image domain but using motion velocities.
        """
        # (N, C=3, T-1, V, M=1) => unify shape
        out = output.squeeze(-1).permute(0, 2, 3, 1)  # (N, T-1, V, 3)
        tgt = target.squeeze(-1).permute(0, 2, 3, 1)

        # If T-1 < 2, can't compute cov meaningfully. Just return 0
        if out.size(1) < 2:
            return torch.tensor(0.0, device=self.device)

        # Compute velocity along time dimension
        # out[:, 1:, ...] - out[:, :-1, ...] => shape (N, T-2, V, 3)
        out_vel = out[:, 1:, :, :] - out[:, :-1, :, :]
        tgt_vel = tgt[:, 1:, :, :] - tgt[:, :-1, :, :]

        # Flatten out => (N*(T-2)*V, 3)
        out_vel = out_vel.reshape(-1, 3)
        tgt_vel = tgt_vel.reshape(-1, 3)

        # Compute statistics
        mu1, sigma1 = self._compute_statistics(out_vel)
        mu2, sigma2 = self._compute_statistics(tgt_vel)

        # Compute Frechet distance
        fid_val = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_val

    def bone_length_loss(self, output, target):
        """
        Enforces consistent bone lengths between generated motion and the target.
        For each dataset, we define a typical bone list. Then we compute
        the difference in bone length for each pair. Summation or average over them.
        
        Inspired by typical skeleton constraints in e.g. [Holden et al. 2016, 2017].
        """
        # (N, C=3, T, V, M=1)
        out = output.squeeze(-1).permute(0, 2, 3, 1)  # (N, T, V, 3)
        tgt = target.squeeze(-1).permute(0, 2, 3, 1)

        # We define adjacency of NTU (or ETRI) for "bones". 
        # You can refine or load from the official graph. 
        # For brevity, let's define a small adjacency for 25-joint skeleton:
        bone_pairs = bone_pairs_dict.get(self.dataset, [])

        if len(bone_pairs) == 0:
            # If no bone pairs defined for this dataset, no penalty
            return torch.tensor(0.0, device=self.device)

        # We'll compute the MSE difference in bone lengths frame by frame
        # across all bones.
        losses = []
        for (j1, j2) in bone_pairs:
            out_vec = out[:, :, j1, :] - out[:, :, j2, :]  # (N, T, 3)
            tgt_vec = tgt[:, :, j1, :] - tgt[:, :, j2, :]
            out_len = torch.norm(out_vec, dim=-1)  # (N, T)
            tgt_len = torch.norm(tgt_vec, dim=-1)
            losses.append((out_len - tgt_len).pow(2))  # (N, T)

        # Stack => shape (num_bones, N, T), mean over all
        bone_loss = torch.mean(torch.stack(losses, dim=0))
        return bone_loss

    def foot_contact_loss(self, output, target):
        """
        A simplified foot-sliding penalty. 
        We check frames in which the target's foot is presumably contacting 
        (velocity < threshold). Then we penalize the predicted foot velocity in those frames.
        
        This approach is naive because it doesn't consider absolute foot height or ground plane,
        but it's a common trick in mocap foot contact constraints (ref. [Holden 2017]).
        """
        # (N, C=3, T, V, M=1)
        out = output.squeeze(-1).permute(0, 2, 3, 1)  # (N, T, V, 3)
        tgt = target.squeeze(-1).permute(0, 2, 3, 1)

        # Velocity along T
        out_vel = out[:, 1:, :, :] - out[:, :-1, :, :]  # (N, T-1, V, 3)
        tgt_vel = tgt[:, 1:, :, :] - tgt[:, :-1, :, :]

        # Foot-joint indices (rough guess for NTU)
        foot_joints = foot_indices_dict.get(self.dataset, [])
        if len(foot_joints) == 0:
            return torch.tensor(0.0, device=self.device)

        # Subselect foot joints => shape (N, T-1, len(feet), 3)
        out_foot_vel = out_vel[:, :, foot_joints, :]
        tgt_foot_vel = tgt_vel[:, :, foot_joints, :]

        # Norm of foot velocity
        out_foot_speed = torch.norm(out_foot_vel, dim=-1)  # (N, T-1, #feet)
        tgt_foot_speed = torch.norm(tgt_foot_vel, dim=-1)

        # Threshold to decide "contact"
        threshold = 0.05
        # Build a mask of shape (N, T-1, #feet). 
        # True => foot contact in target
        contact_mask = (tgt_foot_speed < threshold).float()

        # We want predicted foot speed to be small in those frames
        # => MSE or L2 cost on out_foot_speed in masked frames
        # Weighted by contact_mask
        loss_mat = out_foot_speed * contact_mask  # speed in frames where contact_mask=1
        # Average them
        loss = torch.mean(loss_mat)
        return loss

    # -----------------
    # Helpers
    # -----------------
    def _compute_statistics(self, arr):
        """
        Compute mean and covariance of arr, where arr is shape (N, D).
        Returns:
          mu: (D,)
          sigma: (D, D)
        """
        mu = torch.mean(arr, dim=0, keepdim=True)  # shape (1, D)
        diff = arr - mu
        # Cov => E[xx^T]
        cov = torch.matmul(diff.transpose(0, 1), diff) / (arr.size(0) - 1)
        return mu.view(-1), cov

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Standard Frechet distance (FID) formula:
           d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))
        We'll do a matrix-sqrt using a custom function.
        """
        diff = mu1 - mu2
        diff_sq = diff.dot(diff)

        # Add small eps to diagonals for numerical stability
        sigma1_eps = sigma1 + torch.eye(sigma1.size(0), device=self.device) * eps
        sigma2_eps = sigma2 + torch.eye(sigma2.size(0), device=self.device) * eps

        # sqrt of product
        covmean = self._matrix_sqrt(sigma1_eps.mm(sigma2_eps))

        # Might fail if rank is deficient
        if not torch.isfinite(covmean).all():
            # fallback
            return diff_sq

        fid = diff_sq + torch.trace(sigma1_eps + sigma2_eps - 2*covmean)
        return torch.sqrt(torch.clamp(fid, min=1e-12))

    def _matrix_sqrt(self, x):
        """
        Compute the matrix square root via repeated (Newton-Schulz) method.
        x: (D, D)
        """
        # Some minimal checks
        dim = x.size(0)
        I = torch.eye(dim, device=x.device, dtype=x.dtype)
        norm_x = x.norm()
        # Scale to improve conditioning
        x = x / norm_x

        y = torch.clone(x)
        z = I.clone()
        for _ in range(5):
            y2 = y.matmul(y)
            z2 = z.matmul(z)
            # Newton-Schulz iteration
            numerator = 0.5 * (3*I - z2)
            y = y.matmul(numerator)
            z = numerator.matmul(z)
        # Unscale
        y = y * torch.sqrt(norm_x)
        return y

    # -----------------
    # Master loss aggregator
    # -----------------
    def loss(self, output, target, input):
        """
        Aggregates all requested losses. 
        :param output: (N, C_in, T-1, V, M)
        :param target: (N, C_in, T-1, V, M)
        :param input:  (N, C_in, T-1, V, M) - sometimes used for certain losses
        """
        total_loss = 0
        losses = {key: 0 for key in self.loss_weights}

        # Standard losses
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
        if self.inception:
            losses['inception'] = self.inception_loss(output, target)

        # New losses
        if self.fid_vel:
            losses['fid_vel'] = self.fid_velocity_loss(output, target)
        if self.bone:
            losses['bone'] = self.bone_length_loss(output, target)
        if self.foot:
            losses['foot'] = self.foot_contact_loss(output, target)

        # Weighted sum
        total_loss = sum([losses[k] * self.loss_weights[k] for k in self.loss_weights if k in losses])
        return total_loss, losses


# -----------------------------
# End-effector config for NTU
# -----------------------------
ee_chains = {
    'ntu': torch.tensor([19, 15, 23, 24, 21, 22, 3]),
    'ntu120': torch.tensor([19, 15, 23, 24, 21, 22, 3]),
    'etri': torch.tensor([19, 15, 23, 24, 21, 22, 3]),
}

ee_chain_lengths = {
    'ntu': torch.tensor([5, 5, 8, 8, 8, 8, 5]),
    'ntu120': torch.tensor([5, 5, 8, 8, 8, 8, 5]),
    'etri': torch.tensor([5, 5, 8, 8, 8, 8, 5]),
}

# -----------------------------------
# Bone pairs for bone-length loss
# (You can refine these for your dataset)
# Example pairs for NTU-type skeleton
#  (joint indices) 
#  same as the adjacency from official NTU, but truncated
# -----------------------------------
bone_pairs_dict = {
    'ntu': [
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ],
    'ntu120': [
        # same as 'ntu'
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ],
    'etri': [
        # If ETRI also uses an NTU-like skeleton
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ]
}

# -----------------------------------
# Foot indices for foot-contact loss
# Typically ankles / feet in NTU:
#   left ankle=19, right ankle=15, etc.
#   Some sets also include foot tips or toes if using 30-joint expansions.
# -----------------------------------
foot_indices_dict = {
    'ntu': [15, 19], 
    'ntu120': [15, 19],
    'etri': [15, 19]
}
