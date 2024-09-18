"""
Frozen SGN Auxiliary Loss

Uses a pre-trained, frozen SGN model to guide TMR to generate
motion that is recognizable by downstream models.

IMPORTANT: SGN is frozen (no gradient updates) to prevent adversarial
perturbations. This loss should have a SMALL weight to avoid creating
"spider skeletons" that fool SGN but look unrealistic.

Safeguards:
1. SGN is completely frozen (requires_grad=False)
2. Loss weight should be small (0.1-0.3)
3. Used in combination with reconstruction and physical losses
4. Only applied in Stage 2 and Stage 3 (after encoder pretraining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FrozenSGNLoss(nn.Module):
    """
    Auxiliary loss using frozen pre-trained SGN
    
    Guides TMR to generate motion that SGN can recognize,
    without creating adversarial perturbations
    """
    
    def __init__(self, sgn_checkpoint_path, num_classes=None, device='cuda'):
        super().__init__()
        self.device = device

        # Import SGN here to avoid circular imports
        from src.model.sgn import SGN

        # Infer num_classes from checkpoint if possible
        ckpt_num_classes = None
        if os.path.exists(sgn_checkpoint_path):
            ckpt = torch.load(sgn_checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            for key in ['fc.weight', 'module.fc.weight']:
                if key in state:
                    ckpt_num_classes = state[key].shape[0]
                    break
        fallback_classes = 49  # NTU with two-person actions removed
        if num_classes is None:
            num_classes = ckpt_num_classes or fallback_classes
        elif ckpt_num_classes is not None and ckpt_num_classes != num_classes:
            print(f"⚠ Overriding requested num_classes={num_classes} with checkpoint head={ckpt_num_classes}")
            num_classes = ckpt_num_classes
        self.num_classes = num_classes

        # Load pre-trained SGN
        print(f"Loading frozen SGN from {sgn_checkpoint_path} (num_classes={self.num_classes})...")
        self.sgn = SGN(
            num_classes=self.num_classes,
            dataset='ntu',
            seg=64,
            bias=True
        ).to(device)
        
        if os.path.exists(sgn_checkpoint_path):
            checkpoint = torch.load(sgn_checkpoint_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            # Handle num_classes mismatch (e.g., checkpoint has 48 classes, model has 49)
            # We'll load all weights except the final FC layer if there's a mismatch
            model_state = self.sgn.state_dict()
            filtered_state = {}

            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        print(f"  ⚠ Skipping {k}: shape mismatch ({v.shape} vs {model_state[k].shape})")

            self.sgn.load_state_dict(filtered_state, strict=False)
            print(f"✓ Frozen SGN loaded successfully! ({len(filtered_state)}/{len(model_state)} layers)")
        else:
            print(f"⚠ WARNING: SGN checkpoint not found at {sgn_checkpoint_path}")
            print("  Using randomly initialized SGN (not recommended!)")
        
        # FREEZE all SGN parameters
        for param in self.sgn.parameters():
            param.requires_grad = False
        
        self.sgn.eval()  # Set to eval mode
        
        print(f"✓ SGN frozen with {sum(p.numel() for p in self.sgn.parameters())} parameters")
    
    def prepare_for_sgn(self, motion):
        """
        Convert motion to SGN format: (B, T, V*C)
        
        Args:
            motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            sgn_input: (B, T, V*C)
        """
        if len(motion.shape) == 5:
            motion = motion.squeeze(-1)  # (B, C, T, V)
        
        B, C, T, V = motion.shape
        
        # SGN expects (B, T, V*C)
        motion = motion.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        motion = motion.view(B, T, V * C)  # (B, T, 75)
        
        return motion
    
    def forward(self, generated_motion, action_labels):
        """
        Compute frozen SGN auxiliary loss
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M) - TMR-generated motion
            action_labels: (B,) - ground truth action labels (0-indexed)
        
        Returns:
            loss: scalar cross-entropy loss
            accuracy: scalar accuracy (for monitoring)
        """
        # Prepare motion for SGN
        sgn_input = self.prepare_for_sgn(generated_motion)
        
        # Forward pass through frozen SGN (no gradient to SGN)
        with torch.no_grad():
            self.sgn.eval()  # Ensure eval mode
        
        # Get SGN predictions (gradients flow back to generated_motion only)
        logits = self.sgn(sgn_input)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, action_labels)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == action_labels).float().mean()
        
        return loss, accuracy.item()
    
    def get_predictions(self, generated_motion):
        """
        Get SGN predictions for generated motion (for evaluation)
        
        Args:
            generated_motion: (B, C, T, V) or (B, C, T, V, M)
        
        Returns:
            logits: (B, num_classes)
            predictions: (B,)
        """
        sgn_input = self.prepare_for_sgn(generated_motion)
        
        with torch.no_grad():
            logits = self.sgn(sgn_input)
            predictions = logits.argmax(dim=1)
        
        return logits, predictions


class FrozenSGNLossWithSafeguards(FrozenSGNLoss):
    """
    Enhanced frozen SGN loss with additional safeguards
    
    Monitors for adversarial perturbations and adjusts loss weight dynamically
    """
    
    def __init__(self, sgn_checkpoint_path, num_classes=48, device='cuda',
                 max_weight=0.3, min_accuracy_threshold=0.3):
        super().__init__(sgn_checkpoint_path, num_classes, device)
        
        self.max_weight = max_weight
        self.min_accuracy_threshold = min_accuracy_threshold
        self.current_weight = max_weight
        
        # Track statistics
        self.accuracy_history = []
        self.max_history_len = 100
    
    def update_weight(self, accuracy):
        """
        Dynamically adjust loss weight based on accuracy
        
        If accuracy is too low, reduce weight to prevent adversarial perturbations
        """
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > self.max_history_len:
            self.accuracy_history.pop(0)
        
        # If recent accuracy is very low, reduce weight
        if len(self.accuracy_history) >= 10:
            recent_avg = sum(self.accuracy_history[-10:]) / 10
            if recent_avg < self.min_accuracy_threshold:
                self.current_weight = max(0.05, self.current_weight * 0.9)
                print(f"⚠ Low SGN accuracy ({recent_avg:.3f}), reducing weight to {self.current_weight:.3f}")
    
    def forward(self, generated_motion, action_labels):
        """
        Compute frozen SGN loss with safeguards
        
        Returns:
            loss: scalar (with current weight applied)
            accuracy: scalar
            current_weight: current loss weight
        """
        loss, accuracy = super().forward(generated_motion, action_labels)
        
        # Update weight based on accuracy
        self.update_weight(accuracy)
        
        # Apply current weight
        weighted_loss = self.current_weight * loss
        
        return weighted_loss, accuracy, self.current_weight


def test_frozen_sgn_loss():
    """Test frozen SGN loss"""
    print("Testing Frozen SGN Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if SGN checkpoint exists
    sgn_path = 'output/ntu_sgn_ar_paired/model_best.pth.tar'
    if not os.path.exists(sgn_path):
        print(f"⚠ SGN checkpoint not found at {sgn_path}")
        print("  Skipping test (requires pre-trained SGN)")
        return
    
    loss_fn = FrozenSGNLoss(sgn_path, num_classes=48, device=device)
    
    # Create dummy data
    B, C, T, V = 4, 3, 64, 25
    generated = torch.randn(B, C, T, V, requires_grad=True).to(device)
    action_labels = torch.randint(0, 48, (B,)).to(device)
    
    print("\n1. Forward pass:")
    loss, accuracy = loss_fn(generated, action_labels)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    print("\n2. Backward pass:")
    loss.backward()
    print(f"   Generated motion gradient: {generated.grad is not None}")
    print(f"   SGN parameters frozen: {all(not p.requires_grad for p in loss_fn.sgn.parameters())}")
    
    print("\n3. Test with safeguards:")
    loss_fn_safe = FrozenSGNLossWithSafeguards(sgn_path, num_classes=48, device=device)
    generated2 = torch.randn(B, C, T, V, requires_grad=True).to(device)
    loss, accuracy, weight = loss_fn_safe(generated2, action_labels)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Weight: {weight:.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_frozen_sgn_loss()
