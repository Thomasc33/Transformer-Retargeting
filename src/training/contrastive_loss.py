"""
Contrastive Learning Loss for TMR

InfoNCE (Normalized Temperature-scaled Cross Entropy) loss to cluster
same actions in embedding space while separating different actions.

Based on recommendations from ChatGPT Deep Research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(encoder, anchor, positive, negatives, temperature=0.07):
    """
    InfoNCE contrastive loss.
    
    Encourages:
    - anchor and positive (same action, different person) to be close
    - anchor and negatives (different actions) to be far apart
    
    Args:
        encoder: Model with get_embedding() method (e.g., CooperativeARClassifier)
        anchor: (B, C, T, V, M) motion of action A by person P1
        positive: (B, C, T, V, M) motion of action A by person P2
        negatives: (B, K, C, T, V, M) motions of K different actions
        temperature: Temperature parameter for scaling (default 0.07)
    
    Returns:
        loss: InfoNCE loss value
    """
    # Get embeddings
    anchor_emb = encoder.get_embedding(anchor)  # (B, D)
    positive_emb = encoder.get_embedding(positive)  # (B, D)
    
    # Handle negatives
    if negatives.dim() == 6:
        # negatives is (B, K, C, T, V, M)
        B, K, C, T, V, M = negatives.shape
        negatives_flat = negatives.view(B * K, C, T, V, M)
        negative_embs = encoder.get_embedding(negatives_flat)  # (B*K, D)
        negative_embs = negative_embs.view(B, K, -1)  # (B, K, D)
    else:
        # negatives is (B, C, T, V, M) - single negative per anchor
        negative_embs = encoder.get_embedding(negatives).unsqueeze(1)  # (B, 1, D)
    
    # Normalize embeddings
    anchor_emb = F.normalize(anchor_emb, dim=1)  # (B, D)
    positive_emb = F.normalize(positive_emb, dim=1)  # (B, D)
    negative_embs = F.normalize(negative_embs, dim=2)  # (B, K, D)
    
    # Compute similarities
    pos_sim = (anchor_emb * positive_emb).sum(dim=1) / temperature  # (B,)
    neg_sim = torch.bmm(negative_embs, anchor_emb.unsqueeze(2)).squeeze(2) / temperature  # (B, K)
    
    # Concatenate positive and negative similarities
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+K)
    
    # Labels: positive is always at index 0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def simple_contrastive_loss(encoder, motion, action_labels, temperature=0.07):
    """
    Simplified contrastive loss using batch samples.
    
    For each sample in the batch:
    - Positive: another sample with the same action
    - Negatives: samples with different actions
    
    Args:
        encoder: Model with get_embedding() method
        motion: (B, C, T, V, M) batch of motions
        action_labels: (B,) action labels for each motion
        temperature: Temperature parameter
    
    Returns:
        loss: Contrastive loss value
    """
    B = motion.size(0)
    
    # Get embeddings for all samples
    embeddings = encoder.get_embedding(motion)  # (B, D)
    embeddings = F.normalize(embeddings, dim=1)  # Normalize
    
    # Compute similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature  # (B, B)
    
    # Create mask for positive pairs (same action, different sample)
    action_labels = action_labels.unsqueeze(1)  # (B, 1)
    pos_mask = (action_labels == action_labels.t()).float()  # (B, B)
    
    # Remove self-similarity
    pos_mask.fill_diagonal_(0)
    
    # Create mask for negative pairs (different action)
    neg_mask = (action_labels != action_labels.t()).float()  # (B, B)
    
    # Check if we have valid positive pairs
    if pos_mask.sum() == 0:
        # No positive pairs in batch, return zero loss
        return torch.tensor(0.0, device=motion.device)
    
    # Compute loss for each anchor
    losses = []
    for i in range(B):
        # Get positive and negative similarities for anchor i
        pos_sims = sim_matrix[i][pos_mask[i] == 1]
        neg_sims = sim_matrix[i][neg_mask[i] == 1]
        
        if len(pos_sims) == 0 or len(neg_sims) == 0:
            continue
        
        # For each positive, compute loss against all negatives
        for pos_sim in pos_sims:
            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            
            # Label: positive is at index 0
            label = torch.zeros(1, dtype=torch.long, device=motion.device)
            
            # Cross-entropy loss
            loss_i = F.cross_entropy(logits.unsqueeze(0), label)
            losses.append(loss_i)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=motion.device)
    
    # Average loss
    loss = torch.stack(losses).mean()
    
    return loss


def test_contrastive_loss():
    """
    Test function to verify contrastive loss works correctly.
    """
    print("Testing Contrastive Loss...")
    
    # Create dummy encoder
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 64 * 25, 256)
        
        def get_embedding(self, motion):
            # motion: (B, C, T, V, M)
            B = motion.size(0)
            motion_flat = motion.view(B, -1)
            # Take mean to reduce dimension
            motion_flat = motion_flat[:, :3*64*25]
            return self.fc(motion_flat)
    
    encoder = DummyEncoder()
    encoder.eval()
    
    # Create dummy data
    batch_size = 8
    anchor = torch.randn(batch_size, 3, 64, 25, 1)
    positive = torch.randn(batch_size, 3, 64, 25, 1)
    negatives = torch.randn(batch_size, 4, 3, 64, 25, 1)  # 4 negatives per anchor
    
    # Test InfoNCE loss
    with torch.no_grad():
        loss_infonce = infonce_loss(encoder, anchor, positive, negatives)
    
    assert not torch.isnan(loss_infonce), "InfoNCE loss is NaN"
    assert not torch.isinf(loss_infonce), "InfoNCE loss is Inf"
    assert loss_infonce >= 0, "InfoNCE loss is negative"
    
    print(f"✅ InfoNCE loss: {loss_infonce.item():.6f}")
    
    # Test simple contrastive loss
    motion = torch.randn(batch_size, 3, 64, 25, 1)
    action_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4 actions, 2 samples each
    
    with torch.no_grad():
        loss_simple = simple_contrastive_loss(encoder, motion, action_labels)
    
    assert not torch.isnan(loss_simple), "Simple contrastive loss is NaN"
    assert not torch.isinf(loss_simple), "Simple contrastive loss is Inf"
    assert loss_simple >= 0, "Simple contrastive loss is negative"
    
    print(f"✅ Simple contrastive loss: {loss_simple.item():.6f}")
    
    print("\n✅ All contrastive loss tests passed!")


if __name__ == "__main__":
    test_contrastive_loss()

