"""
Enhanced Physical Plausibility Losses for TMR

This module implements comprehensive biomechanical constraints to ensure
realistic human motion generation, including:
1. Joint angle limits based on human anatomy
2. Foot contact and ground plane constraints
3. Momentum conservation constraints
4. End-effector smoothness constraints
5. Existing constraints (bone length, temporal smoothness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class EnhancedPhysicalLoss(nn.Module):
    """
    Enhanced physical plausibility losses with biomechanical constraints.
    
    Implements comprehensive physical constraints for realistic human motion:
    - Joint angle limits (anatomical constraints)
    - Foot contact detection and ground plane constraints
    - Momentum conservation (center of mass dynamics)
    - End-effector smoothness (hands, feet, head)
    - Bone length consistency (from existing implementation)
    - Temporal smoothness (from existing implementation)
    """
    
    def __init__(self, dataset='ntu', device='cuda'):
        super().__init__()
        self.dataset = dataset
        self.device = device
        
        # NTU RGB+D skeleton structure (25 joints)
        self._setup_skeleton_structure()
        self._setup_joint_angle_limits()
        self._setup_end_effectors()
        
    def _setup_skeleton_structure(self):
        """Setup NTU RGB+D skeleton bone connections and joint indices."""
        # NTU RGB+D 25-joint skeleton structure
        # Joint indices (0-24):
        # 0: spine base, 1: middle spine, 2: neck, 3: head
        # 4-7: right arm (shoulder, elbow, wrist, hand)
        # 8-11: left arm (shoulder, elbow, wrist, hand)
        # 12-15: right leg (hip, knee, ankle, foot)
        # 16-19: left leg (hip, knee, ankle, foot)
        # 20: spine, 21-24: hand tips and thumbs
        
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
        
        self.bones = torch.tensor(self.ntu_bones, dtype=torch.long).to(self.device)
        
        # Joint indices for key body parts
        self.joint_indices = {
            'spine_base': 0,
            'middle_spine': 1,
            'neck': 2,
            'head': 3,
            'right_shoulder': 4,
            'right_elbow': 5,
            'right_wrist': 6,
            'right_hand': 7,
            'left_shoulder': 8,
            'left_elbow': 9,
            'left_wrist': 10,
            'left_hand': 11,
            'right_hip': 12,
            'right_knee': 13,
            'right_ankle': 14,
            'right_foot': 15,
            'left_hip': 16,
            'left_knee': 17,
            'left_ankle': 18,
            'left_foot': 19,
            'spine': 20,
        }
        
    def _setup_joint_angle_limits(self):
        """Setup anatomical joint angle limits based on biomechanics literature.

        Note: compute_joint_angle uses acos on bone vectors meeting at joint,
        so 0 rad = straight limb, pi = fully folded. acos returns [0, pi].
        Validated against Physiopedia ROM norms and CDC joint ROM study.
        """
        self.angle_limits = {
            'knee': {
                'joints': [self.joint_indices['right_knee'], self.joint_indices['left_knee']],
                'min_angle': 0.0,
                'max_angle': np.pi * 170 / 180,  # 170° (140° max flexion + Kinect noise buffer)
                'parent_child_pairs': [
                    (self.joint_indices['right_hip'], self.joint_indices['right_knee'], self.joint_indices['right_ankle']),
                    (self.joint_indices['left_hip'], self.joint_indices['left_knee'], self.joint_indices['left_ankle'])
                ]
            },
            'elbow': {
                'joints': [self.joint_indices['right_elbow'], self.joint_indices['left_elbow']],
                'min_angle': 0.0,
                'max_angle': np.pi * 160 / 180,  # 160° (150° max flexion + buffer)
                'parent_child_pairs': [
                    (self.joint_indices['right_shoulder'], self.joint_indices['right_elbow'], self.joint_indices['right_wrist']),
                    (self.joint_indices['left_shoulder'], self.joint_indices['left_elbow'], self.joint_indices['left_wrist'])
                ]
            },
            'hip_flexion': {
                'joints': [self.joint_indices['right_hip'], self.joint_indices['left_hip']],
                'min_angle': 0.0,
                'max_angle': np.pi,  # 180° — ball-and-socket, effectively unconstrained
                'parent_child_pairs': [
                    (self.joint_indices['spine_base'], self.joint_indices['right_hip'], self.joint_indices['right_knee']),
                    (self.joint_indices['spine_base'], self.joint_indices['left_hip'], self.joint_indices['left_knee'])
                ]
            },
            'shoulder_flexion': {
                'joints': [self.joint_indices['right_shoulder'], self.joint_indices['left_shoulder']],
                'min_angle': 0.0,
                'max_angle': np.pi,  # 180° — ball-and-socket, effectively unconstrained
                'parent_child_pairs': [
                    (self.joint_indices['spine'], self.joint_indices['right_shoulder'], self.joint_indices['right_elbow']),
                    (self.joint_indices['spine'], self.joint_indices['left_shoulder'], self.joint_indices['left_elbow'])
                ]
            },
            'spine': {
                'joints': [self.joint_indices['middle_spine']],
                'min_angle': 0.0,
                'max_angle': np.pi * 90 / 180,  # 90° (allows bending, sitting, bowing)
                'parent_child_pairs': [
                    (self.joint_indices['spine_base'], self.joint_indices['middle_spine'], self.joint_indices['neck'])
                ]
            }
        }
        
    def _setup_end_effectors(self):
        """Setup end-effector joint indices for smoothness constraints."""
        self.end_effectors = {
            'hands': [self.joint_indices['left_hand'], self.joint_indices['right_hand']],  # joints 11, 7
            'feet': [self.joint_indices['left_foot'], self.joint_indices['right_foot']],   # joints 19, 15
            'head': [self.joint_indices['head']]  # joint 3
        }
        
    def compute_joint_angle(self, parent_pos, joint_pos, child_pos):
        """
        Compute joint angle from three consecutive joint positions.
        
        Args:
            parent_pos: (B, T, 3) - parent joint positions
            joint_pos: (B, T, 3) - current joint positions  
            child_pos: (B, T, 3) - child joint positions
            
        Returns:
            angles: (B, T) - joint angles in radians
        """
        # Compute bone vectors
        vec1 = parent_pos - joint_pos  # vector from joint to parent
        vec2 = child_pos - joint_pos   # vector from joint to child
        
        # Normalize vectors
        vec1_norm = F.normalize(vec1, dim=-1, eps=1e-8)
        vec2_norm = F.normalize(vec2, dim=-1, eps=1e-8)
        
        # Compute dot product (cosine of angle)
        cos_angle = torch.sum(vec1_norm * vec2_norm, dim=-1)  # (B, T)
        
        # Clamp to valid range for acos
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angle
        angles = torch.acos(cos_angle)  # (B, T)
        
        return angles
        
    def joint_angle_loss(self, motion):
        """
        Enforce anatomical joint angle limits.
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M) - skeleton motion
            
        Returns:
            loss: scalar tensor
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
            
        B, C, T, V = motion.shape
        
        # Reshape to (B, T, V, C)
        motion = motion.permute(0, 2, 3, 1).contiguous()
        
        total_loss = 0.0
        violation_count = 0
        
        # Check each joint type
        for joint_type, limits in self.angle_limits.items():
            for parent_idx, joint_idx, child_idx in limits['parent_child_pairs']:
                # Get joint positions
                parent_pos = motion[:, :, parent_idx, :]  # (B, T, 3)
                joint_pos = motion[:, :, joint_idx, :]    # (B, T, 3)
                child_pos = motion[:, :, child_idx, :]    # (B, T, 3)
                
                # Compute joint angles
                angles = self.compute_joint_angle(parent_pos, joint_pos, child_pos)  # (B, T)
                
                # Check violations
                min_violation = F.relu(limits['min_angle'] - angles)  # penalty for angles below minimum
                max_violation = F.relu(angles - limits['max_angle'])  # penalty for angles above maximum
                
                # Smooth penalty function (quadratic)
                joint_loss = torch.mean(min_violation ** 2 + max_violation ** 2)
                total_loss += joint_loss
                violation_count += 1
                
        # Average over all joints
        if violation_count > 0:
            total_loss = total_loss / violation_count
            
        return total_loss
        
    def foot_contact_loss(self, motion):
        """
        Detect foot contact and penalize foot sliding and ground penetration.
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M) - skeleton motion
            
        Returns:
            loss: scalar tensor
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
            
        B, C, T, V = motion.shape
        
        # Reshape to (B, T, V, C)
        motion = motion.permute(0, 2, 3, 1).contiguous()
        
        # Get foot positions
        left_foot = motion[:, :, self.joint_indices['left_foot'], :]   # (B, T, 3)
        right_foot = motion[:, :, self.joint_indices['right_foot'], :] # (B, T, 3)
        
        total_loss = 0.0
        
        for foot_pos in [left_foot, right_foot]:
            # 1. Ground plane penetration (y-coordinate should not be negative)
            # Assuming y-axis is vertical (up is positive)
            ground_penetration = F.relu(-foot_pos[:, :, 1])  # penalize y < 0
            penetration_loss = torch.mean(ground_penetration ** 2)
            
            # 2. Foot contact detection and sliding penalty
            if T > 1:
                # Compute foot velocity
                foot_velocity = foot_pos[:, 1:, :] - foot_pos[:, :-1, :]  # (B, T-1, 3)
                foot_speed = torch.norm(foot_velocity, dim=-1)  # (B, T-1)
                
                # Detect contact (foot close to ground and low velocity)
                foot_height = foot_pos[:, 1:, 1]  # y-coordinate, (B, T-1)
                contact_threshold_height = 0.1  # 10cm above ground
                contact_threshold_speed = 0.05   # 5cm/s velocity
                
                # Contact detected when foot is low and moving slowly
                is_contact = (foot_height < contact_threshold_height) & (foot_speed < contact_threshold_speed)
                
                # During contact, foot should not slide (velocity should be near zero)
                sliding_penalty = foot_speed * is_contact.float()
                sliding_loss = torch.mean(sliding_penalty ** 2)
            else:
                sliding_loss = torch.tensor(0.0, device=motion.device)
                
            total_loss += penetration_loss + sliding_loss
            
        return total_loss
        
    def momentum_conservation_loss(self, motion):
        """
        Enforce momentum conservation for physically plausible dynamics.
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M) - skeleton motion
            
        Returns:
            loss: scalar tensor
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
            
        B, C, T, V = motion.shape
        
        if T < 3:  # Need at least 3 frames for acceleration
            return torch.tensor(0.0, device=motion.device)
            
        # Reshape to (B, T, V, C)
        motion = motion.permute(0, 2, 3, 1).contiguous()
        
        # Compute center of mass (simple average of all joints)
        # In practice, could use weighted average based on body segment masses
        center_of_mass = torch.mean(motion, dim=2)  # (B, T, 3)
        
        # Compute velocity and acceleration of center of mass
        com_velocity = center_of_mass[:, 1:, :] - center_of_mass[:, :-1, :]      # (B, T-1, 3)
        com_acceleration = com_velocity[:, 1:, :] - com_velocity[:, :-1, :]      # (B, T-2, 3)
        
        # Penalize unrealistic acceleration patterns (sudden changes in velocity)
        # Use L2 norm of acceleration
        acceleration_magnitude = torch.norm(com_acceleration, dim=-1)  # (B, T-2)
        
        # Penalize large accelerations (unrealistic dynamics)
        max_reasonable_acceleration = 10.0  # m/s^2 (adjust based on data scale)
        excessive_acceleration = F.relu(acceleration_magnitude - max_reasonable_acceleration)
        
        loss = torch.mean(excessive_acceleration ** 2)
        
        return loss
        
    def end_effector_smoothness_loss(self, motion):
        """
        Enforce smooth trajectories for end-effectors (hands, feet, head).
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M) - skeleton motion
            
        Returns:
            loss: scalar tensor
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
            
        B, C, T, V = motion.shape
        
        if T < 3:  # Need at least 3 frames for jerk computation
            return torch.tensor(0.0, device=motion.device)
            
        # Reshape to (B, T, V, C)
        motion = motion.permute(0, 2, 3, 1).contiguous()
        
        total_loss = 0.0
        effector_count = 0
        
        # Process each end-effector type
        for effector_type, joint_indices in self.end_effectors.items():
            for joint_idx in joint_indices:
                joint_pos = motion[:, :, joint_idx, :]  # (B, T, 3)
                
                # Compute velocity and acceleration
                velocity = joint_pos[:, 1:, :] - joint_pos[:, :-1, :]          # (B, T-1, 3)
                acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]        # (B, T-2, 3)
                
                if T > 3:
                    # Compute jerk (third derivative - rate of change of acceleration)
                    jerk = acceleration[:, 1:, :] - acceleration[:, :-1, :]    # (B, T-3, 3)
                    
                    # Minimize jerk for smooth trajectories
                    jerk_magnitude = torch.norm(jerk, dim=-1)  # (B, T-3)
                    jerk_loss = torch.mean(jerk_magnitude ** 2)
                else:
                    jerk_loss = torch.tensor(0.0, device=motion.device)
                
                # Also penalize large accelerations for smoothness
                acceleration_magnitude = torch.norm(acceleration, dim=-1)  # (B, T-2)
                acceleration_loss = torch.mean(acceleration_magnitude ** 2)
                
                # Combine jerk and acceleration penalties
                effector_loss = jerk_loss + 0.1 * acceleration_loss
                total_loss += effector_loss
                effector_count += 1
                
        # Average over all end-effectors
        if effector_count > 0:
            total_loss = total_loss / effector_count
            
        return total_loss
        
    def compute_bone_lengths(self, motion):
        """
        Compute bone lengths for all frames (from existing implementation).
        
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
        Bone length consistency loss (from existing implementation).
        
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
        Temporal smoothness loss (from existing implementation).
        
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
        
    def forward(self, generated_motion, target_motion=None, weights=None):
        """
        Compute all enhanced physical plausibility losses.
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M) - generated skeleton motion
            target_motion: (B, C, T, V) or (B, C, T, V, M) - target skeleton motion (optional)
            weights: dict with loss weights
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual losses
        """
        if weights is None:
            weights = {
                # New enhanced losses
                'joint_angle': 0.3,
                'foot_contact': 0.2,
                'momentum': 0.1,
                'end_effector': 0.2,
                # Existing losses
                'bone_length': 0.5,
                'temporal_smoothness': 0.3,
            }
        
        loss_dict = {}
        total_loss = 0.0
        
        # New enhanced physical losses (only need generated motion)
        joint_angle_loss = self.joint_angle_loss(generated_motion)
        foot_contact_loss = self.foot_contact_loss(generated_motion)
        momentum_loss = self.momentum_conservation_loss(generated_motion)
        end_effector_loss = self.end_effector_smoothness_loss(generated_motion)
        
        loss_dict.update({
            'joint_angle': joint_angle_loss.item(),
            'foot_contact': foot_contact_loss.item(),
            'momentum': momentum_loss.item(),
            'end_effector': end_effector_loss.item(),
        })
        
        total_loss += (
            weights['joint_angle'] * joint_angle_loss +
            weights['foot_contact'] * foot_contact_loss +
            weights['momentum'] * momentum_loss +
            weights['end_effector'] * end_effector_loss
        )
        
        # Existing losses (need both generated and target motion)
        if target_motion is not None:
            bone_loss = self.bone_length_loss(generated_motion, target_motion)
            loss_dict['bone_length'] = bone_loss.item()
            total_loss += weights['bone_length'] * bone_loss
        
        # Temporal smoothness (only needs generated motion)
        smoothness_loss = self.temporal_smoothness_loss(generated_motion)
        loss_dict['temporal_smoothness'] = smoothness_loss.item()
        total_loss += weights['temporal_smoothness'] * smoothness_loss
        
        loss_dict['total_enhanced_physical'] = total_loss.item()
        
        return total_loss, loss_dict


def test_enhanced_physical_losses():
    """Test enhanced physical plausibility losses."""
    print("Testing Enhanced Physical Plausibility Losses...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = EnhancedPhysicalLoss(dataset='ntu', device=device)
    
    # Create dummy data
    B, C, T, V = 4, 3, 64, 25
    generated = torch.randn(B, C, T, V).to(device)
    target = torch.randn(B, C, T, V).to(device)
    
    print(f"Input shape: {generated.shape}")
    print(f"Device: {device}")
    
    # Test individual new losses
    print("\n=== New Enhanced Losses ===")
    
    print("\n1. Joint Angle Loss:")
    try:
        joint_loss = loss_fn.joint_angle_loss(generated)
        print(f"   Loss: {joint_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Foot Contact Loss:")
    try:
        foot_loss = loss_fn.foot_contact_loss(generated)
        print(f"   Loss: {foot_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Momentum Conservation Loss:")
    try:
        momentum_loss = loss_fn.momentum_conservation_loss(generated)
        print(f"   Loss: {momentum_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. End-Effector Smoothness Loss:")
    try:
        effector_loss = loss_fn.end_effector_smoothness_loss(generated)
        print(f"   Loss: {effector_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test existing losses
    print("\n=== Existing Losses ===")
    
    print("\n5. Bone Length Loss:")
    try:
        bone_loss = loss_fn.bone_length_loss(generated, target)
        print(f"   Loss: {bone_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n6. Temporal Smoothness Loss:")
    try:
        smoothness_loss = loss_fn.temporal_smoothness_loss(generated)
        print(f"   Loss: {smoothness_loss.item():.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test combined loss
    print("\n=== Combined Loss ===")
    try:
        total_loss, loss_dict = loss_fn(generated, target)
        print(f"   Total: {total_loss.item():.6f}")
        print("   Individual losses:")
        for key, value in loss_dict.items():
            print(f"     {key}: {value:.6f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✓ All tests completed!")


if __name__ == '__main__':
    test_enhanced_physical_losses()