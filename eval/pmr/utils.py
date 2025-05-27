import torch

def create_onehot(indices, num_classes, device=None):
    """
    Create one-hot encoding tensors, ensuring device compatibility.
    
    Args:
        indices: The class indices (zero-based)
        num_classes: Number of classes for one-hot encoding
        device: Device to place tensor on (defaults to indices' device)
    
    Returns:
        One-hot encoding tensor on the appropriate device
    """
    if device is None and isinstance(indices, torch.Tensor):
        device = indices.device
        
    # Create tensor on the correct device
    eye_tensor = torch.eye(num_classes, device=device)
    
    # Make sure indices are long type on correct device
    if isinstance(indices, torch.Tensor):
        indices = indices.long().to(device)
    else:
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        
    # Create one-hot encoding
    return eye_tensor[indices]
