"""
MIRAGE-inspired losses for TMR Stage 3 training.

Adds four key components from MIRAGE to improve TMR's action recognition
on retargeted output while maintaining privacy:

1. DistributionDiscriminator: Ensures retargeted output stays in-distribution
2. OutputActionClassifier: Cooperative action classification on decoder output
3. OutputIdentityAdversary: Adversarial identity classification on decoder output
4. OutputContrastiveLoss: Scatters same-identity outputs in embedding space
5. Enhanced end-effector loss with velocity preservation

All losses operate on TMR's (B, C=3, T, V=25, M=1) format and convert
internally to (B, T, 75) where needed.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def tmr_to_flat(x):
    """Convert TMR format (B, C=3, T, V=25, M=1) to flat (B, T, 75).

    Returns a new tensor without modifying the input.
    """
    if x.dim() == 5:
        B, C, T, V, M = x.shape
        # (B, C, T, V, M) -> (B, T, V, C, M) -> (B, T, V*C)
        return x.permute(0, 2, 3, 1, 4).reshape(B, T, V * C)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# 1. Distribution Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class DistributionDiscriminator(nn.Module):
    """Binary classifier: raw (1) vs retargeted (0) skeleton sequences.

    Uses temporal convolutions + global pooling -> binary logit.
    Kept simple to provide smooth gradient signal without overpowering
    the generator.

    Args:
        input_dim: per-frame dimension (75 = 25 joints * 3 coords)
        hidden_dim: conv hidden channels
        dropout: dropout rate
        use_spectral_norm: apply spectral normalization for stability
    """

    def __init__(self, input_dim=75, hidden_dim=128, dropout=0.3,
                 use_spectral_norm=False):
        super().__init__()
        sn = spectral_norm if use_spectral_norm else (lambda x: x)

        self.conv1 = sn(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2))
        self.conv2 = sn(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        self.conv3 = sn(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1))
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

        self.fc1 = sn(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.fc2 = sn(nn.Linear(hidden_dim // 4, 1))

    def forward(self, x_flat):
        """
        Args:
            x_flat: (B, T, 75) skeleton sequence in flat format

        Returns:
            logits: (B, 1) -- positive means 'raw', negative means 'retargeted'
        """
        x = x_flat.permute(0, 2, 1)  # (B, 75, T)

        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(self.drop(f1)))
        f3 = self.act(self.conv3(self.drop(f2)))

        pooled = f3.mean(dim=2)  # (B, hidden//2)
        logits = self.fc2(self.act(self.fc1(pooled)))  # (B, 1)
        return logits

    def compute_loss(self, raw_tmr, retargeted_tmr):
        """Compute discriminator + generator losses.

        Args:
            raw_tmr: (B, C, T, V, M) raw skeleton (ground truth y2 target)
            retargeted_tmr: (B, C, T, V, M) retargeted output (decoder output)

        Returns:
            disc_loss: loss for updating the discriminator (maximize separation)
            gen_loss: loss for updating the generator (fool the discriminator)
            disc_acc: discriminator accuracy (for logging)
        """
        raw_flat = tmr_to_flat(raw_tmr)
        ret_flat = tmr_to_flat(retargeted_tmr)

        # Discriminator predictions
        real_logits = self.forward(raw_flat)
        fake_logits = self.forward(ret_flat.detach())

        # Discriminator loss: real=1, fake=0
        disc_loss = (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        ) * 0.5

        # Discriminator accuracy
        with torch.no_grad():
            real_correct = (real_logits > 0).float().mean()
            fake_correct = (fake_logits <= 0).float().mean()
            disc_acc = (real_correct + fake_correct).item() * 0.5

        # Generator loss: fool discriminator (retargeted should look real)
        gen_logits = self.forward(ret_flat)
        gen_loss = F.binary_cross_entropy_with_logits(
            gen_logits, torch.ones_like(gen_logits)
        )

        return disc_loss, gen_loss, disc_acc


# ──────────────────────────────────────────────────────────────────────────────
# 2. Output-Level Action Classifier (cooperative)
# ──────────────────────────────────────────────────────────────────────────────

class OutputActionClassifier(nn.Module):
    """Cooperative action classifier on retargeted output.

    Temporal conv -> global avg pool -> MLP -> num_actions.
    The generator MINIMIZES this loss so the retargeted output
    is easily recognizable by action.

    Args:
        input_dim: per-frame dimension (75)
        num_classes: number of action classes
        hidden_dim: hidden channels
        dropout: dropout rate
    """

    def __init__(self, input_dim=75, num_classes=49, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_tmr):
        """
        Args:
            x_tmr: (B, C=3, T, V=25, M=1) retargeted output in TMR format

        Returns:
            logits: (B, num_classes)
        """
        x_flat = tmr_to_flat(x_tmr)  # (B, T, 75)
        x = x_flat.permute(0, 2, 1)  # (B, 75, T)
        x = self.temporal_net(x)  # (B, hidden, T)
        x = x.mean(dim=2)  # (B, hidden) -- global avg pool
        return self.classifier(x)

    def compute_loss(self, retargeted_tmr, action_labels):
        """Cooperative loss: generator minimizes this.

        Args:
            retargeted_tmr: (B, C, T, V, M) decoder output
            action_labels: (B,) action class indices

        Returns:
            loss: cross-entropy loss (generator minimizes)
            accuracy: classification accuracy (for logging)
        """
        logits = self.forward(retargeted_tmr)
        loss = F.cross_entropy(logits, action_labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == action_labels).float().mean().item()

        return loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# 3. Output-Level Identity Adversary
# ──────────────────────────────────────────────────────────────────────────────

class OutputIdentityAdversary(nn.Module):
    """Adversarial identity classifier on retargeted output.

    Same architecture as OutputActionClassifier but for identity.
    The classifier is trained to recognize identity (classifier step),
    and the generator NEGATES this loss to fool it (generator step).

    Args:
        input_dim: per-frame dimension (75)
        num_identities: number of identity classes
        hidden_dim: hidden channels
        dropout: dropout rate
    """

    def __init__(self, input_dim=75, num_identities=40, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_identities),
        )

    def forward(self, x_tmr):
        """
        Args:
            x_tmr: (B, C=3, T, V=25, M=1) retargeted output in TMR format

        Returns:
            logits: (B, num_identities)
        """
        x_flat = tmr_to_flat(x_tmr)  # (B, T, 75)
        x = x_flat.permute(0, 2, 1)  # (B, 75, T)
        x = self.temporal_net(x)  # (B, hidden, T)
        x = x.mean(dim=2)  # (B, hidden)
        return self.classifier(x)

    def compute_loss(self, retargeted_tmr, identity_labels):
        """Compute classifier loss and adversarial generator loss.

        Args:
            retargeted_tmr: (B, C, T, V, M) decoder output
            identity_labels: (B,) identity class indices

        Returns:
            classifier_loss: CE loss for training the classifier (detached output)
            adversarial_loss: negated CE loss for the generator (live gradients)
            accuracy: classification accuracy (for logging)
        """
        # Classifier step: train on detached output
        logits_detached = self.forward(retargeted_tmr.detach())
        classifier_loss = F.cross_entropy(logits_detached, identity_labels)

        # Generator step: fool the classifier (negate the loss)
        logits_live = self.forward(retargeted_tmr)
        adversarial_loss = -F.cross_entropy(logits_live, identity_labels)

        with torch.no_grad():
            preds = logits_detached.argmax(dim=1)
            accuracy = (preds == identity_labels).float().mean().item()

        return classifier_loss, adversarial_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# 4. Output-Level Identity Contrastive Loss (scattering)
# ──────────────────────────────────────────────────────────────────────────────

class OutputContrastiveLoss(nn.Module):
    """Supervised contrastive loss on retargeted output for identity scattering.

    Projects output sequences to a compact embedding, applies SupCon loss
    using identity labels. The generator NEGATES this to scatter same-identity
    outputs in representation space.

    Args:
        input_dim: per-frame dimension (75)
        proj_dim: projection embedding dimension
        temperature: contrastive temperature
    """

    def __init__(self, input_dim=75, proj_dim=128, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, retargeted_tmr, identity_labels):
        """Compute contrastive loss on retargeted output.

        Args:
            retargeted_tmr: (B, C=3, T, V=25, M=1) decoder output
            identity_labels: (B,) identity class indices

        Returns:
            loss: scalar. Positive when same-identity pairs are similar.
                  Generator should NEGATE this.
        """
        x_flat = tmr_to_flat(retargeted_tmr)  # (B, T, 75)
        pooled = x_flat.mean(dim=1)  # (B, 75) -- temporal average

        # Project to unit sphere
        z = self.projector(pooled)  # (B, proj_dim)
        z = F.normalize(z, dim=1)

        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Pairwise cosine similarity / temperature
        sim = torch.matmul(z, z.T) / self.temperature  # (B, B)

        # Same-identity mask (excluding self)
        labels = identity_labels.unsqueeze(0)  # (1, B)
        pos_mask = (labels == labels.T).float()  # (B, B)
        pos_mask.fill_diagonal_(0.0)

        # If no positive pairs in this batch, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Log-sum-exp over all negatives (exclude self)
        self_mask = torch.eye(B, device=z.device)
        neg_mask = 1.0 - self_mask

        # Numerical stability
        sim_max, _ = sim.detach().max(dim=1, keepdim=True)
        sim = sim - sim_max

        exp_sim = torch.exp(sim) * neg_mask
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # SupCon: average log-prob of positive pairs
        log_prob = sim - log_sum_exp
        pos_log_prob = (log_prob * pos_mask).sum(dim=1)
        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_log_prob / num_pos).mean()

        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 5. Enhanced End-Effector Loss
# ──────────────────────────────────────────────────────────────────────────────

# NTU end-effector joints (action-critical)
EE_INDICES = [11, 23, 24, 7, 21, 22, 19, 15, 3]


def enhanced_end_effector_loss(output_tmr, target_tmr):
    """End-effector position + velocity preservation loss.

    Adapted from MIRAGE for TMR's (B, C=3, T, V=25, M=1) format.

    Args:
        output_tmr: (B, C, T, V, M) retargeted output
        target_tmr: (B, C, T, V, M) ground truth target

    Returns:
        loss: scalar combining position and velocity errors at end-effectors
    """
    # Extract end-effector positions: (B, 3, T, num_ee, 1)
    y_ee = output_tmr[:, :, :, EE_INDICES, :]
    x_ee = target_tmr[:, :, :, EE_INDICES, :]

    # Position error
    pos_err = F.mse_loss(y_ee, x_ee)

    # Velocity error (temporal differences)
    T = output_tmr.shape[2]
    if T > 1:
        y_vel = y_ee[:, :, 1:, :, :] - y_ee[:, :, :-1, :, :]
        x_vel = x_ee[:, :, 1:, :, :] - x_ee[:, :, :-1, :, :]
        vel_err = F.mse_loss(y_vel, x_vel)
        return pos_err + vel_err

    return pos_err


# ──────────────────────────────────────────────────────────────────────────────
# Container for all MIRAGE-inspired losses
# ──────────────────────────────────────────────────────────────────────────────

class MirageInspiredLosses(nn.Module):
    """Container managing all MIRAGE-inspired loss modules.

    Provides a single interface for Stage 3 training:
    - Creates and holds all sub-modules
    - Provides separate parameter groups for optimizers
    - Computes all losses in one call

    Args:
        num_classes: number of action classes
        num_identities: number of identity classes
        device: torch device
        lambda_dist_disc: weight for distribution discriminator loss
        lambda_output_act: weight for output action classifier loss
        lambda_output_id: weight for output identity adversary loss
        lambda_output_contrastive: weight for output contrastive loss
        lambda_ee_enhanced: weight for enhanced end-effector loss
    """

    def __init__(self, num_classes=49, num_identities=40, device='cuda',
                 lambda_dist_disc=1.0, lambda_output_act=1.0,
                 lambda_output_id=1.0, lambda_output_contrastive=1.0,
                 lambda_ee_enhanced=1.0, lambda_motion_disc=0.0,
                 lambda_coord_std=0.0, raw_data_dict=None):
        super().__init__()

        self.lambda_dist_disc = lambda_dist_disc
        self.lambda_output_act = lambda_output_act
        self.lambda_output_id = lambda_output_id
        self.lambda_output_contrastive = lambda_output_contrastive
        self.lambda_ee_enhanced = lambda_ee_enhanced
        self.lambda_motion_disc = lambda_motion_disc
        self.lambda_coord_std = lambda_coord_std

        # Sub-modules
        self.dist_disc = DistributionDiscriminator(
            input_dim=75, hidden_dim=128, dropout=0.3, use_spectral_norm=False
        ).to(device)

        self.output_act = OutputActionClassifier(
            input_dim=75, num_classes=num_classes, hidden_dim=128, dropout=0.3
        ).to(device)

        self.output_id = OutputIdentityAdversary(
            input_dim=75, num_identities=num_identities, hidden_dim=128, dropout=0.3
        ).to(device)

        self.output_contrastive = OutputContrastiveLoss(
            input_dim=75, proj_dim=128, temperature=0.1
        ).to(device)

        # Motion-space distribution discriminator (Approach 2)
        self.motion_disc = None
        if lambda_motion_disc > 0:
            self.motion_disc = MotionDistributionDiscriminator(
                input_dim=147, hidden_dim=128, dropout=0.3
            ).to(device)

        # Coordinate standardization loss (Approach 4)
        self.coord_std_loss = None
        if lambda_coord_std > 0 and raw_data_dict is not None:
            self.coord_std_loss = CoordinateStandardizationLoss.from_data(
                raw_data_dict, device=device
            )

    def get_discriminator_params(self):
        """Parameters for the discriminator optimizer (separate from generator)."""
        params = list(self.dist_disc.parameters()) + list(self.output_id.parameters())
        if self.motion_disc is not None:
            params += list(self.motion_disc.parameters())
        return params

    def get_classifier_params(self):
        """Parameters for the cooperative action classifier optimizer."""
        return list(self.output_act.parameters())

    def get_contrastive_params(self):
        """Parameters for the contrastive projector."""
        return list(self.output_contrastive.parameters())

    def compute_all(self, output_tmr, target_tmr, action_labels, identity_labels):
        """Compute all MIRAGE-inspired losses.

        Args:
            output_tmr: (B, C, T, V, M) decoder output (retargeted)
            target_tmr: (B, C, T, V, M) ground truth target (y2)
            action_labels: (B,) action class indices
            identity_labels: (B,) identity class indices (of target identity)

        Returns:
            dict with:
                'gen_total': total generator loss (to add to main loss)
                'disc_total': total discriminator/classifier loss (separate step)
                + individual losses and accuracies for logging
        """
        results = {}

        # 1. Distribution discriminator
        disc_loss, gen_disc_loss, disc_acc = self.dist_disc.compute_loss(
            target_tmr, output_tmr
        )
        results['disc_loss'] = disc_loss
        results['gen_disc_loss'] = gen_disc_loss
        results['disc_acc'] = disc_acc

        # 2. Output action classifier (cooperative)
        act_loss, act_acc = self.output_act.compute_loss(output_tmr, action_labels)
        results['output_act_loss'] = act_loss
        results['output_act_acc'] = act_acc

        # 3. Output identity adversary
        id_cls_loss, id_adv_loss, id_acc = self.output_id.compute_loss(
            output_tmr, identity_labels
        )
        results['output_id_cls_loss'] = id_cls_loss
        results['output_id_adv_loss'] = id_adv_loss
        results['output_id_acc'] = id_acc

        # 4. Output contrastive scattering
        contrastive_loss = self.output_contrastive(output_tmr, identity_labels)
        results['output_contrastive_loss'] = contrastive_loss

        # 5. Enhanced end-effector loss
        ee_loss = enhanced_end_effector_loss(output_tmr, target_tmr)
        results['ee_enhanced_loss'] = ee_loss

        # 6. Motion-space distribution discriminator
        motion_disc_loss = torch.tensor(0.0, device=output_tmr.device)
        gen_motion_disc_loss = torch.tensor(0.0, device=output_tmr.device)
        motion_disc_acc = 0.0
        if self.motion_disc is not None:
            motion_disc_loss, gen_motion_disc_loss, motion_disc_acc = (
                self.motion_disc.compute_loss(target_tmr, output_tmr)
            )
        results['motion_disc_loss'] = motion_disc_loss
        results['gen_motion_disc_loss'] = gen_motion_disc_loss
        results['motion_disc_acc'] = motion_disc_acc

        # 7. Coordinate standardization loss
        coord_std_loss = torch.tensor(0.0, device=output_tmr.device)
        if self.coord_std_loss is not None:
            coord_std_loss = self.coord_std_loss(output_tmr)
        results['coord_std_loss'] = coord_std_loss

        # Aggregate: generator loss (added to main TMR loss)
        gen_total = (
            self.lambda_dist_disc * gen_disc_loss
            + self.lambda_output_act * act_loss
            + self.lambda_output_id * id_adv_loss
            - self.lambda_output_contrastive * contrastive_loss  # negate: scatter same-id
            + self.lambda_ee_enhanced * ee_loss
            + self.lambda_motion_disc * gen_motion_disc_loss
            + self.lambda_coord_std * coord_std_loss
        )
        results['gen_total'] = gen_total

        # Aggregate: discriminator/adversary loss (separate optimizer step)
        disc_total = disc_loss + id_cls_loss
        if self.motion_disc is not None:
            disc_total = disc_total + motion_disc_loss
        results['disc_total'] = disc_total

        return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. Motion-Space Distribution Discriminator
# ──────────────────────────────────────────────────────────────────────────────

# NTU bone pairs for angle computation
_BONE_PAIRS_FOR_ANGLES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def compute_motion_features(x_tmr):
    """Compute body-shape-invariant motion features: velocities + bone angles.

    Args:
        x_tmr: (B, C=3, T, V=25, M=1) skeleton in TMR format

    Returns:
        features: (B, T-1, D) where D = 75 (velocities) + 24*3 (bone directions)
    """
    x_flat = tmr_to_flat(x_tmr)  # (B, T, 75)
    B, T, _ = x_flat.shape

    # Velocities: frame differences (B, T-1, 75)
    velocities = x_flat[:, 1:, :] - x_flat[:, :-1, :]

    # Bone angle features: unit bone direction vectors (shape-invariant)
    joints = x_flat[:, 1:, :].reshape(B, T - 1, 25, 3)  # use frames 1..T
    bone_dirs = []
    for parent, child in _BONE_PAIRS_FOR_ANGLES:
        bone_vec = joints[:, :, child, :] - joints[:, :, parent, :]  # (B, T-1, 3)
        bone_len = bone_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        bone_dirs.append(bone_vec / bone_len)  # unit direction

    bone_features = torch.cat(bone_dirs, dim=-1)  # (B, T-1, 24*3=72)

    return torch.cat([velocities, bone_features], dim=-1)  # (B, T-1, 147)


class MotionDistributionDiscriminator(nn.Module):
    """Discriminator operating on motion-space features (velocities + bone angles).

    Unlike DistributionDiscriminator which sees raw coordinates (and thus body
    shape), this discriminator only sees shape-invariant motion patterns.
    This encourages the retargeted output to have natural motion dynamics
    without penalizing different body proportions.

    Args:
        input_dim: per-frame motion feature dimension (147 = 75 vel + 72 bone dirs)
        hidden_dim: conv hidden channels
        dropout: dropout rate
    """

    def __init__(self, input_dim=147, hidden_dim=128, dropout=0.3):
        super().__init__()
        sn = spectral_norm

        self.conv1 = sn(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2))
        self.conv2 = sn(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        self.conv3 = sn(nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1))
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

        self.fc1 = sn(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.fc2 = sn(nn.Linear(hidden_dim // 4, 1))

    def forward(self, motion_features):
        """
        Args:
            motion_features: (B, T, D) motion features

        Returns:
            logits: (B, 1)
        """
        x = motion_features.permute(0, 2, 1)  # (B, D, T)
        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(self.drop(f1)))
        f3 = self.act(self.conv3(self.drop(f2)))
        pooled = f3.mean(dim=2)
        return self.fc2(self.act(self.fc1(pooled)))

    def compute_loss(self, raw_tmr, retargeted_tmr):
        """Compute discriminator + generator losses on motion features.

        Args:
            raw_tmr: (B, C, T, V, M) raw skeleton
            retargeted_tmr: (B, C, T, V, M) retargeted output

        Returns:
            disc_loss, gen_loss, disc_acc
        """
        raw_feat = compute_motion_features(raw_tmr)
        ret_feat = compute_motion_features(retargeted_tmr)

        real_logits = self.forward(raw_feat)
        fake_logits = self.forward(ret_feat.detach())

        disc_loss = (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        ) * 0.5

        with torch.no_grad():
            real_correct = (real_logits > 0).float().mean()
            fake_correct = (fake_logits <= 0).float().mean()
            disc_acc = (real_correct + fake_correct).item() * 0.5

        gen_logits = self.forward(ret_feat)
        gen_loss = F.binary_cross_entropy_with_logits(
            gen_logits, torch.ones_like(gen_logits)
        )

        return disc_loss, gen_loss, disc_acc


# ──────────────────────────────────────────────────────────────────────────────
# 7. Coordinate Standardization Loss
# ──────────────────────────────────────────────────────────────────────────────

class CoordinateStandardizationLoss(nn.Module):
    """Penalizes per-joint coordinate deviation from raw data statistics.

    Pre-computes mean and std of each joint coordinate across the raw training
    set, then during training penalizes:
        L_std = MSE(mean(output_joints, batch), raw_mean)
              + MSE(std(output_joints, batch), raw_std)

    This encourages the retargeted output to have similar coordinate
    distributions to raw data without requiring a frozen classifier.

    Args:
        raw_mean: (75,) tensor of per-coordinate means
        raw_std: (75,) tensor of per-coordinate stds
    """

    def __init__(self, raw_mean, raw_std):
        super().__init__()
        self.register_buffer('raw_mean', raw_mean)
        self.register_buffer('raw_std', raw_std)

    @staticmethod
    def from_data(raw_data_dict, device='cuda'):
        """Create from raw data dictionary.

        Args:
            raw_data_dict: dict {name: ndarray(T, 75)}
            device: torch device

        Returns:
            CoordinateStandardizationLoss instance
        """
        all_frames = []
        for seq in raw_data_dict.values():
            s = seq[:, :75] if seq.shape[1] > 75 else seq  # Take first person only
            all_frames.append(s)  # (T, 75)
        all_frames = np.concatenate(all_frames, axis=0)  # (N, 75)

        raw_mean = torch.tensor(all_frames.mean(axis=0), dtype=torch.float32, device=device)
        raw_std = torch.tensor(all_frames.std(axis=0), dtype=torch.float32, device=device).clamp(min=1e-6)

        return CoordinateStandardizationLoss(raw_mean, raw_std)

    def forward(self, output_tmr):
        """Compute coordinate standardization loss.

        Args:
            output_tmr: (B, C=3, T, V=25, M=1) decoder output

        Returns:
            loss: scalar
        """
        x_flat = tmr_to_flat(output_tmr)  # (B, T, 75)

        # Pool over time to get per-sample stats, then batch stats
        # (B, T, 75) -> (B*T, 75)
        flat = x_flat.reshape(-1, 75)

        batch_mean = flat.mean(dim=0)  # (75,)
        batch_std = flat.std(dim=0).clamp(min=1e-6)  # (75,)

        loss_mean = F.mse_loss(batch_mean, self.raw_mean)
        loss_std = F.mse_loss(batch_std, self.raw_std)

        return loss_mean + loss_std
