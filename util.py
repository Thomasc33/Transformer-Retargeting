import torch
import numpy as np
import random

def format_data(x):
    '''
    Format the input data to match the input shape of the model
    :param x: Input data
    :return: Formatted input data
    '''
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

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False