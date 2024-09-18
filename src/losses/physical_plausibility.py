"""
Physical plausibility losses to ensure realistic motion generation

These losses prevent "spider skeleton" artifacts by enforcing:
1. Bone length consistency (bones don't change length)
2. Temporal smoothness (motion is not jittery)
3. Velocity consistency (acceleration is bounded)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalPlausibilityLoss(nn.Module):
    """
    Physical plausibility losses for skeleton motion
    
    Prevents adversarial perturbations by enforcing physical constraints
    """
    
    def __init__(self, dataset='ntu', device='cuda'):
        super().__init__()
        self.dataset = dataset
        self.device = device
        
        # NTU RGB+D skeleton bone connections (25 joints)
        # Format: (parent_joint, child_joint)
        self.ntu_bones = [
            (0, 1),   # base of spine -> middle of spine
            (1, 20),  # middle of spine -> spine
            (20, 2),  # spine -> shoulder center
            (2, 3),   # shoulder center -> head
            (20, 8),  # spine -> left shoulder
            (8, 9),   # left shoulder -> left elbow
            (9, 10),  # left elbow -> left wrist
            (10, 11), # left wrist -> left hand
            (11, 23), # left hand -> left hand tip
            (10, 24), # left wrist -> left thumb
            (20, 4),  # spine -> right shoulder
            (4, 5),   # right shoulder -> right elbow
            (5, 6),   # right elbow -> right wrist
            (6, 7),   # right wrist -> right hand
            (7, 21),  # right hand -> right hand tip
            (6, 22),  # right wrist -> right thumb
            (0, 16),  # base of spine -> left hip
            (16, 17), # left hip -> left knee
            (17, 18), # left knee -> left ankle
            (18, 19), # left ankle -> left foot
            (0, 12),  # base of spine -> right hip
            (12, 13), # right hip -> right knee
            (13, 14), # right knee -> right ankle
            (14, 15), # right ankle -> right foot
        ]
        
        self.bones = torch.tensor(self.ntu_bones, dtype=torch.long).to(device)
    
    def compute_bone_lengths(self, motion):
        """
        Compute bone lengths for all frames
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            bone_lengths: (B, T, num_bones)
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
        
        B, C, T, V = motion.shape
        
        # Reshape to (B, T, V, C)
        motion = motion.permute(0, 2, 3, 1).contiguous()
        
        # Get parent and child joint positions
        parent_joints = motion[:, :, self.bones[:, 0], :]  # (B, T, num_bones, C)
        child_joints = motion[:, :, self.bones[:, 1], :]   # (B, T, num_bones, C)
        
        # Compute bone vectors
        bone_vectors = child_joints - parent_joints  # (B, T, num_bones, C)
        
        # Compute bone lengths
        bone_lengths = torch.norm(bone_vectors, dim=-1)  # (B, T, num_bones)
        
        return bone_lengths
    
    def bone_length_loss(self, generated_motion, target_motion):
        """
        Bone length consistency loss
        
        Ensures generated motion has similar bone lengths to target motion
        This prevents skeleton deformation (spider skeletons)
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M)
            target_motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            loss: scalar
        """
        gen_bone_lengths = self.compute_bone_lengths(generated_motion)
        target_bone_lengths = self.compute_bone_lengths(target_motion)
        
        # MSE loss on bone lengths
        loss = F.mse_loss(gen_bone_lengths, target_bone_lengths)
        
        return loss
    
    def temporal_smoothness_loss(self, motion):
        """
        Temporal smoothness loss
        
        Penalizes jittery motion by enforcing smooth velocity
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            loss: scalar
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
        
        # Compute first-order differences (velocity)
        velocity = motion[:, :, 1:, :] - motion[:, :, :-1, :]  # (B, C, T-1, V)
        
        # Compute second-order differences (acceleration)
        acceleration = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]  # (B, C, T-2, V)
        
        # L2 norm of acceleration (penalize large changes in velocity)
        loss = torch.mean(acceleration ** 2)
        
        return loss
    
    def end_effector_loss(self, generated_motion, target_motion):
        """
        End-effector specific loss
        
        Enforces accurate position/velocity for hands and feet (essential for action meaning)
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M)
            target_motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            loss: scalar
        """
        if len(generated_motion.shape) == 5:
            generated_motion = generated_motion.squeeze(-1)
        if len(target_motion.shape) == 5:
            target_motion = target_motion.squeeze(-1)
            
        # NTU End Effectors:
        # 11: left hand, 23: left hand tip, 10: left thumb
        # 7: right hand, 21: right hand tip, 6: right thumb
        # 15: right foot, 19: left foot
        # Indices (0-based):
        end_effectors = [11, 23, 10, 7, 21, 6, 15, 19]
        
        # Position Loss
        gen_pos = generated_motion[:, :, :, end_effectors]
        tgt_pos = target_motion[:, :, :, end_effectors]
        pos_loss = F.mse_loss(gen_pos, tgt_pos)
        
        # Velocity Loss (Crucial for "drink water" type actions)
        gen_vel = gen_pos[:, :, 1:, :] - gen_pos[:, :, :-1, :]
        tgt_vel = tgt_pos[:, :, 1:, :] - tgt_pos[:, :, :-1, :]
        vel_loss = F.mse_loss(gen_vel, tgt_vel)
        
        return pos_loss + vel_loss

    def velocity_consistency_loss(self, generated_motion, target_motion):
        """
        Velocity consistency loss
        
        Ensures generated motion has similar velocity patterns to target motion
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M)
            target_motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            loss: scalar
        """
        if len(generated_motion.shape) == 5:
            generated_motion = generated_motion.squeeze(-1)
        if len(target_motion.shape) == 5:
            target_motion = target_motion.squeeze(-1)
        
        # Compute velocities
        gen_velocity = generated_motion[:, :, 1:, :] - generated_motion[:, :, :-1, :]
        target_velocity = target_motion[:, :, 1:, :] - target_motion[:, :, :-1, :]
        
        # MSE loss on velocities
        loss = F.mse_loss(gen_velocity, target_velocity)
        
        return loss
    
    def forward(self, generated_motion, target_motion, weights=None):
        """
        Compute all physical plausibility losses
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M)
            target_motion: (B, C, T, V) or (B, C, T, V, M)
            weights: dict with keys 'bone_length', 'temporal_smoothness', 'velocity'
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        if weights is None:
            weights = {
                'bone_length': 0.5,
                'temporal_smoothness': 0.3,
                'velocity': 0.2
            }
        
        # Compute individual losses
        bone_loss = self.bone_length_loss(generated_motion, target_motion)
        smoothness_loss = self.temporal_smoothness_loss(generated_motion)
        velocity_loss = self.velocity_consistency_loss(generated_motion, target_motion)
        
        # New: End-Effector Loss (if weight provided, else 0)
        ee_weight = weights.get('end_effector', 0.0)
        if ee_weight > 0:
            ee_loss = self.end_effector_loss(generated_motion, target_motion)
        else:
            ee_loss = torch.tensor(0.0, device=self.device)

        # Weighted sum
        total_loss = (
            weights['bone_length'] * bone_loss +
            weights['temporal_smoothness'] * smoothness_loss +
            weights['velocity'] * velocity_loss +
            ee_weight * ee_loss
        )
        
        loss_dict = {
            'bone_length': bone_loss.item(),
            'temporal_smoothness': smoothness_loss.item(),
            'velocity': velocity_loss.item(),
            'end_effector': ee_loss.item(),
            'total_physical': total_loss.item()
        }
        
        return total_loss, loss_dict


def test_physical_losses():
    """Test physical plausibility losses"""
    print("Testing Physical Plausibility Losses...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = PhysicalPlausibilityLoss(dataset='ntu', device=device)
    
    # Create dummy data
    B, C, T, V = 4, 3, 64, 25
    generated = torch.randn(B, C, T, V).to(device)
    target = torch.randn(B, C, T, V).to(device)
    
    # Test individual losses
    print("\n1. Bone Length Loss:")
    bone_loss = loss_fn.bone_length_loss(generated, target)
    print(f"   Loss: {bone_loss.item():.6f}")
    
    print("\n2. Temporal Smoothness Loss:")
    smoothness_loss = loss_fn.temporal_smoothness_loss(generated)
    print(f"   Loss: {smoothness_loss.item():.6f}")
    
    print("\n3. Velocity Consistency Loss:")
    velocity_loss = loss_fn.velocity_consistency_loss(generated, target)
    print(f"   Loss: {velocity_loss.item():.6f}")
    
    print("\n4. Combined Loss:")
    total_loss, loss_dict = loss_fn(generated, target)
    print(f"   Total: {total_loss.item():.6f}")
    print(f"   Dict: {loss_dict}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_physical_losses()

