import torch
import numpy as np
import random
import os
import os.path as osp

#------------------------------------------------------------------------------
# Directory Management Functions
#------------------------------------------------------------------------------

def make_dir(dataset, tag):
    """
    Create output directory for saving model checkpoints and logs.
    
    Args:
        dataset: str - Dataset name (NTU, NTU120, ETRI)
        tag: str - Task tag (ar or ri)
        
    Returns:
        str - Path to output directory
    """
    output_dir = osp.join('eval/sgn/pretrained', dataset.lower())
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

#------------------------------------------------------------------------------
# Data Processing Functions
#------------------------------------------------------------------------------

def format_data(x):
    """
    Format the input data to match the input shape of the model.
    
    Args:
        x: torch.Tensor - Input data tensor
        
    Returns:
        torch.Tensor - Formatted input data with shape (N, C, T, V, M)
    """
    x = x.float().cuda()
    x = x[:, :64]
    N = x.shape[0]
    T = x.shape[1]
    M = 2  # Number of people
    V = 25  # Number of joints
    C = 3  # Number of input channels
    assert M * V * C == x.shape[2], "Mismatch in the number of joints, channels, and people"
    x = x.view(N, T, M, V, C).permute(0, 4, 1, 3, 2)  # N, C, T, V, M
    return x

#------------------------------------------------------------------------------
# Random Seed Functions
#------------------------------------------------------------------------------

def init_seed(seed):
    """
    Initialize random seeds for reproducibility.
    
    Args:
        seed: int - Seed value for random number generators
    """
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False