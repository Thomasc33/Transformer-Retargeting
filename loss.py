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
        self.fid_vel = 'fid_vel' in loss_weights       # FID on velocities
        self.bone = 'bone' in loss_weights             # Bone-length loss
        self.foot = 'foot' in loss_weights             # Foot-contact loss
        self.joint_limit = 'joint_limit' in loss_weights

        # Loss weights dictionary
        self.loss_weights = loss_weights

        self.device = device
        self.dataset = dataset
        self.encoder = encoder

        # Below is a full set of angle definitions for NTU's 25-joint skeleton
        # Each entry = ((parent, joint, child) : (min_angle_deg, max_angle_deg))
        # You can tweak these ranges as needed.
        self.joint_angle_ranges_ntu = {
            (0, 1, 20):  (-95,  95),   # HipCenter -> Spine -> SpineShoulder
            (1, 20, 2):  (-120, 120), # Spine -> SpineShoulder -> Neck
            (20, 2, 3):  (-45,  45),  # SpineShoulder -> Neck -> Head
            (2, 3, 4):   (-95,  95),  # Neck -> Head -> LeftShoulder
            (20, 8, 9):  (-120, 120), # SpineShoulder -> RightShoulder -> RightElbow
            (8, 9, 10):  (-120, 120), # RightShoulder -> RightElbow -> RightWrist
            (9, 10, 11): (-120, 120), # RightElbow -> RightWrist -> RightHand
            (20, 16, 17):(-120, 120), # SpineShoulder -> RightHip -> RightKnee
            (16, 17, 18):(-120, 120), # RightHip -> RightKnee -> RightAnkle
            (17, 18, 19):(-120, 120), # RightKnee -> RightAnkle -> RightFoot
            (1, 5, 6):   (-120, 120), # Spine -> LeftShoulder -> LeftElbow
            (5, 6, 7):   (-120, 120), # LeftShoulder -> LeftElbow -> LeftWrist
            (1, 12, 13): (-120, 120), # Spine -> LeftHip -> LeftKnee
            (12, 13, 14):(-120, 120), # LeftHip -> LeftKnee -> LeftAnkle
            (13, 14, 15):(-120, 120), # LeftKnee -> LeftAnkle -> LeftFoot
            (8, 20, 21): (-120, 120), # RightShoulder -> SpineShoulder -> LeftHandTip
            (20, 21, 22):(-120, 120), # SpineShoulder -> LeftHandTip -> LeftThumb
            (8, 20, 23): (-120, 120), # RightShoulder -> SpineShoulder -> RightHandTip
            (20, 23, 24):(-120, 120), # SpineShoulder -> RightHandTip -> RightThumb
        }

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

        # Check for degenerate cases
        if target.size(2) <= 1:  # T dimension too small
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute squared differences along time dimension
        diff_y = torch.sum((target[:, :, 1:, :] - target[:, :, :-1, :]) ** 2, dim=1)  # (N, T-1, V)
        diff_y_pred = torch.sum((output[:, :, 1:, :] - output[:, :, :-1, :]) ** 2, dim=1)

        # L1 difference of these sums => shape (N, T-1, V)
        abs_diff = torch.abs(diff_y - diff_y_pred)
        # sum over all, then average
        loss = abs_diff.sum()

        # Safe normalization
        normalizer = target.size(1) * target.size(3)
        if normalizer == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.sqrt(torch.clamp(loss, min=1e-12)) / normalizer

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
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def triplet_loss(self, output, target, input):
        """
        Placeholder for a standard triplet margin loss.
        Typically used if you have anchor/positive/negative motions
        to enforce a margin in latent space.
        Not implemented.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    # -----------------
    # Inception-like loss
    # -----------------
    def inception_loss(self, output, target):
        """
        Compares the features from self.encoder(output) and self.encoder(target).
        This is like a naive version of 'inception distance' for motion,
        but we rely on the 'encoder' to extract features.

        Note: Removed @torch.no_grad() to allow gradient computation.
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
            return torch.tensor(0.0, device=self.device, requires_grad=True)

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
            return torch.tensor(0.0, device=self.device, requires_grad=True)

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
            return torch.tensor(0.0, device=self.device, requires_grad=True)

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

    def _compute_angle_between(self, v1, v2):
        """
        Computes the angle (in degrees) between v1, v2 of shape (..., 3).
        angle = arccos((v1 • v2) / (||v1|| * ||v2||)).
        """
        dot = (v1 * v2).sum(dim=-1)  # (...)
        norm1 = torch.norm(v1, dim=-1)
        norm2 = torch.norm(v2, dim=-1)
        denom = norm1 * norm2 + 1e-6
        cos_val = torch.clamp(dot / denom, -1.0, 1.0)
        return torch.acos(cos_val) * (180.0 / math.pi)

    def joint_limit_loss(self, output):
        """
        For each (parent, joint, child) in self.joint_angle_ranges_ntu,
        compute the angle at 'joint' and penalize angles outside [min_deg, max_deg].
        """
        if self.dataset != 'ntu':
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # (N, C=3, T, V, M=1) => (N, T, V, 3)
        out = output.squeeze(-1).permute(0, 2, 3, 1)

        angle_loss_accumulator = []
        for (p, j, c), (min_deg, max_deg) in self.joint_angle_ranges_ntu.items():
            # vectors: parent->joint, joint->child => (N, T, 3)
            pj_vec = out[:, :, j, :] - out[:, :, p, :]
            jc_vec = out[:, :, c, :] - out[:, :, j, :]
            angles = self._compute_angle_between(pj_vec, jc_vec)  # (N, T)

            # measure violation outside [min_deg, max_deg]
            below_mask = (angles < min_deg).float()
            above_mask = (angles > max_deg).float()

            below_violation = (min_deg - angles) * below_mask
            above_violation = (angles - max_deg) * above_mask
            angle_loss_accumulator.append(below_violation + above_violation)

        if not angle_loss_accumulator:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # mean over all frames, joints, batch
        angle_loss = torch.mean(torch.stack(angle_loss_accumulator, dim=0))
        return angle_loss

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
        # Check for empty or invalid input
        if arr.size(0) <= 1:
            # Return zeros for degenerate case
            D = arr.size(1)
            mu = torch.zeros(D, device=arr.device, dtype=arr.dtype)
            sigma = torch.eye(D, device=arr.device, dtype=arr.dtype) * 1e-6
            return mu, sigma

        mu = torch.mean(arr, dim=0, keepdim=True)  # shape (1, D)
        diff = arr - mu

        # Add numerical stability
        N = arr.size(0)
        if N <= 1:
            # Fallback for edge case
            D = arr.size(1)
            sigma = torch.eye(D, device=arr.device, dtype=arr.dtype) * 1e-6
        else:
            # Cov => E[xx^T] with numerical stability
            cov = torch.matmul(diff.transpose(0, 1), diff) / max(N - 1, 1)
            # Add small regularization to diagonal for numerical stability
            D = cov.size(0)
            sigma = cov + torch.eye(D, device=arr.device, dtype=arr.dtype) * 1e-8

        return mu.view(-1), sigma

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Standard Frechet distance (FID) formula:
           d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))
        We'll do a matrix-sqrt using a custom function.
        """
        # Check for NaN inputs
        if not (torch.isfinite(mu1).all() and torch.isfinite(mu2).all() and
                torch.isfinite(sigma1).all() and torch.isfinite(sigma2).all()):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        diff = mu1 - mu2
        diff_sq = diff.dot(diff)

        # Add small eps to diagonals for numerical stability
        sigma1_eps = sigma1 + torch.eye(sigma1.size(0), device=self.device) * eps
        sigma2_eps = sigma2 + torch.eye(sigma2.size(0), device=self.device) * eps

        try:
            # sqrt of product
            covmean = self._matrix_sqrt(sigma1_eps.mm(sigma2_eps))

            # Check if matrix sqrt failed
            if not torch.isfinite(covmean).all():
                # fallback to simpler distance
                return torch.sqrt(torch.clamp(diff_sq, min=1e-12))

            fid = diff_sq + torch.trace(sigma1_eps + sigma2_eps - 2*covmean)
            result = torch.sqrt(torch.clamp(fid, min=1e-12))

            # Final NaN check
            if not torch.isfinite(result):
                return torch.sqrt(torch.clamp(diff_sq, min=1e-12))

            return result
        except Exception:
            # Fallback to simple distance if anything fails
            return torch.sqrt(torch.clamp(diff_sq, min=1e-12))

    def _matrix_sqrt(self, x):
        """
        Compute the matrix square root via repeated (Newton-Schulz) method.
        x: (D, D)
        """
        # Check for invalid input
        if not torch.isfinite(x).all():
            dim = x.size(0)
            return torch.eye(dim, device=x.device, dtype=x.dtype)

        # Some minimal checks
        dim = x.size(0)
        I = torch.eye(dim, device=x.device, dtype=x.dtype)
        norm_x = x.norm()

        # Handle zero norm case
        if norm_x < 1e-12:
            return torch.eye(dim, device=x.device, dtype=x.dtype) * 1e-6

        # Scale to improve conditioning
        x = x / norm_x

        y = torch.clone(x)
        z = I.clone()
        for i in range(5):
            y2 = y.matmul(y)
            z2 = z.matmul(z)
            # Newton-Schulz iteration
            numerator = 0.5 * (3*I - z2)
            y = y.matmul(numerator)
            z = numerator.matmul(z)

            # Check for convergence issues
            if not torch.isfinite(y).all() or not torch.isfinite(z).all():
                return torch.eye(dim, device=x.device, dtype=x.dtype) * torch.sqrt(norm_x)

        # Unscale
        y = y * torch.sqrt(norm_x)

        # Final check
        if not torch.isfinite(y).all():
            return torch.eye(dim, device=x.device, dtype=x.dtype) * torch.sqrt(norm_x)

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

        # Check for NaN/inf in inputs
        if not (torch.isfinite(output).all() and torch.isfinite(target).all()):
            print("Warning: NaN/inf detected in loss inputs!")
            # Return zero losses to avoid propagating NaN, but maintain gradient tracking
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for key in self.loss_weights:
                losses[key] = zero_loss.clone()
            return zero_loss, losses

        # Helper function to create zero loss with gradients
        def zero_loss_with_grad():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Standard losses with NaN checking
        if self.mse:
            loss_val = self.mse_loss(output, target)
            losses['mse'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.l1:
            loss_val = self.l1_loss(output, target)
            losses['l1'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.smoothl1:
            loss_val = self.smoothl1_loss(output, target)
            losses['smoothl1'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.kl:
            loss_val = self.kl_loss(output, target)
            losses['kl'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.ce:
            loss_val = self.ce_loss(output, target)
            losses['ce'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.ee:
            loss_val = self.ee_loss(output, target)
            losses['ee'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.smoothing:
            loss_val = self.smoothing_loss(output, target)
            losses['smoothing'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.latent:
            loss_val = self.latent_loss(output, target)
            losses['latent'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.triplet:
            loss_val = self.triplet_loss(output, target, input)
            losses['triplet'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.inception:
            loss_val = self.inception_loss(output, target)
            losses['inception'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()

        # New losses with NaN checking
        if self.fid_vel:
            loss_val = self.fid_velocity_loss(output, target)
            losses['fid_vel'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.bone:
            loss_val = self.bone_length_loss(output, target)
            losses['bone'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.foot:
            loss_val = self.foot_contact_loss(output, target)
            losses['foot'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()
        if self.joint_limit:
            loss_val = self.joint_limit_loss(output)
            losses['joint_limit'] = loss_val if torch.isfinite(loss_val) else zero_loss_with_grad()

        # Weighted sum with NaN checking
        total_loss = sum([losses[k] * self.loss_weights[k] for k in self.loss_weights if k in losses])

        # Final NaN check
        if not torch.isfinite(total_loss):
            print("Warning: NaN detected in total loss! Setting to 0.")
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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
