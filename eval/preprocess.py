"""
Preprocessing utilities for skeleton-based action recognition evaluation.

This module provides standalone implementations of test-time preprocessing functions
used by SGN and Skeleton-MixFormer models, ensuring consistent evaluation.

Main entry points:
  - sgn_preprocess_single_skeleton: Process skeletons for SGN model evaluation
  - mixformer_preprocess_single_skeleton: Process skeletons for MixFormer model evaluation
"""

import numpy as np
import torch
import math

#------------------------------------------------------------------------------
# Skeleton Structure Constants
#------------------------------------------------------------------------------

# NTU RGB+D skeleton joint pairs for bone representation
ntu_pairs = (
    (2,1), (3,2), (4,3), (5,4), (6,5),
    (7,6), (8,7), (9,8), (10,9), (11,10),
    (12,11), (13,12), (14,13), (15,14), (16,15),
    (17,16), (18,17), (19,18), (20,19), (21,2),
    (22,21), (23,22), (24,23), (25,24)
)

#------------------------------------------------------------------------------
# MixFormer Preprocessing Functions
#------------------------------------------------------------------------------

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    """
    Crop and resize a sequence to a target length.
    
    Args:
        data_numpy: np.ndarray - Input array with shape (C, T, V, M)
        valid_frame_num: int - Number of valid frames
        p_interval: float or list - Random proportion or range for cropping
        window: int - Target length after resizing
        
    Returns:
        np.ndarray - Cropped and resized data with shape (C, window, V, M)
    """
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # Handle different p_interval types
    if isinstance(p_interval, (list, tuple)):
        if len(p_interval) == 1:
            p = p_interval[0]
            bias = int((1 - p) * valid_size / 2)
            data = data_numpy[:, begin + bias:end - bias, :, :]
            cropped_length = data.shape[1]
        else:
            p = np.random.rand() * (p_interval[1] - p_interval[0]) + p_interval[0]
            cropped_length = np.minimum(
                np.maximum(int(np.floor(valid_size * p)), 64),
                valid_size
            )
            bias = np.random.randint(0, valid_size - cropped_length + 1)
            data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
            if data.shape[1] == 0:
                print(cropped_length, bias, valid_size)
    else:
        p = p_interval
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]

    # Resize to 'window' frames with interpolation
    import torch.nn.functional as F
    data_torch = torch.tensor(data, dtype=torch.float)   # (C, cropped_length, V, M)
    data_torch = data_torch.permute(0, 2, 3, 1).contiguous()  # => (C, V, M, cropped_length)
    c, v, m, t = data_torch.shape
    data_torch = data_torch.view(1, 1, c * v * m, t)     # (1,1,(C*V*M),cropped_length)
    data_torch = F.interpolate(data_torch, size=(c * v * m, window),
                               mode='bilinear', align_corners=False)
    data_torch = data_torch.squeeze(0).squeeze(0)        # => shape (C*V*M, window)
    data_torch = data_torch.view(c, v, m, window).permute(0, 3, 1, 2).contiguous()
    out = data_torch.numpy()  # => shape (C, window, V, M)
    return out

def downsample(data_numpy, step, random_sample=True):
    """
    Downsample data along temporal dimension.
    
    Args:
        data_numpy: np.ndarray - Input data
        step: int - Downsampling step
        random_sample: bool - Whether to use random starting point
        
    Returns:
        np.ndarray - Downsampled data
    """
    import random
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]

def mean_subtractor(data_numpy, mean):
    """
    Subtract mean from non-zero frames.
    
    Args:
        data_numpy: np.ndarray - Input data
        mean: float - Mean value to subtract
        
    Returns:
        np.ndarray - Mean-normalized data
    """
    if mean == 0:
        return data_numpy
        
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy

def auto_pading(data_numpy, size, random_pad=False):
    """
    Pad data to target length.
    
    Args:
        data_numpy: np.ndarray - Input data
        size: int - Target size
        random_pad: bool - Whether to use random padding position
        
    Returns:
        np.ndarray - Padded data
    """
    import random
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy

def _rot(rot):
    """
    Create rotation matrices from rotation angles.
    
    Args:
        rot: torch.Tensor - Rotation angles with shape (N, 3)
        
    Returns:
        torch.Tensor - 3D rotation matrices
    """
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[0], 1).zero_()
    ones = rot.new(rot.size()[0], 1).fill_(1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=1)

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    out = rz.matmul(ry).matmul(rx)
    return out

def random_rot(data_numpy, theta=0.3):
    """
    Apply random 3D rotation to data.
    
    Args:
        data_numpy: np.ndarray - Input data with shape (C, T, V, M)
        theta: float - Maximum rotation angle in radians
        
    Returns:
        np.ndarray - Rotated data
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)
    rot = torch.zeros(T, 3).uniform_(-theta, theta)
    rot_mat = _rot(rot)
    data_torch = torch.bmm(rot_mat, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()
    return data_torch.numpy()

#------------------------------------------------------------------------------
# SGN Preprocessing Functions
#------------------------------------------------------------------------------

def turn_two_to_one(seq):
    """
    Merge 2-person skeleton into 1-person format.
    
    Args:
        seq: np.ndarray - Input data with shape (T, 150)
        
    Returns:
        np.ndarray - Processed data with merged skeletons
    """
    new_seq = []
    for ske in seq:
        if (ske[0:75] == 0).all():
            new_seq.append(ske[75:])
        elif (ske[75:] == 0).all():
            new_seq.append(ske[0:75])
        else:
            new_seq.append(ske[0:75])
            new_seq.append(ske[75:])
    return np.array(new_seq)

def sgn_sub_seq(seqs, seq, seg=20, train=1, dataset='NTU'):
    """
    Generate sub-sequences for SGN model.
    
    Args:
        seqs: list - List to store results
        seq: np.ndarray - Input sequence
        seg: int - Segment length
        train: int - 1 for training mode, 2 for test mode
        dataset: str - Dataset name
        
    Returns:
        list - Updated list with sub-sequences
    """
    group = seg
    if dataset in ['SYSU', 'SYSU_same']:
        seq = seq[::2, :]

    # Pad if shorter than segment length
    if seq.shape[0] < seg:
        pad = np.zeros((seg - seq.shape[0], seq.shape[1])).astype(np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    ave_duration = max(seq.shape[0] // group, 1)  # Ensure ave_duration is at least 1

    if train == 1:
        # Single set of random offsets for training
        offsets = np.multiply(list(range(group)), ave_duration) + \
                  np.random.randint(ave_duration, size=group)
        offsets = np.clip(offsets, 0, seq.shape[0] - 1)
        seq_ = seq[offsets]
        seqs.append(seq_)
    elif train == 2:
        # Multiple sets of random offsets for test (5 crops)
        for i in range(5):
            np.random.seed(i + 1000)  # Use different seeds for more diversity
            offsets = np.multiply(list(range(group)), ave_duration) + \
                     np.random.randint(ave_duration, size=group)
            offsets = np.clip(offsets, 0, seq.shape[0] - 1)
            seqs.append(seq[offsets])
        
        # Reset random seed
        np.random.seed(None)
        
    return seqs

def sgn_Tolist_fix(x_list, seg=20, dataset='NTU', is_test=True):
    """
    Process list of skeletons for SGN model.
    
    Args:
        x_list: list - List of raw skeleton arrays
        seg: int - Segment length
        dataset: str - Dataset name
        is_test: bool - Whether in test mode
        
    Returns:
        np.ndarray - Processed data array 
    """
    all_subseq = []
    for x_ in x_list:
        # Merge 2-person skeletons
        x_merged = turn_two_to_one(x_)

        # Generate sub-sequences
        sub_seqs = sgn_sub_seq([], x_merged, seg=seg, train=2 if is_test else 1, dataset=dataset)
        
        if len(sub_seqs) > 0:
            try:
                sub_stack = np.stack(sub_seqs, axis=0)  # shape (5, seg, 75)
                all_subseq.append(sub_stack)
            except ValueError as e:
                print(f"Warning: Could not stack sub-sequences: {e}")
                # Create placeholder with correct shape
                tmp = np.zeros((5, seg, 75), dtype=np.float32)
                all_subseq.append(tmp)
        else:
            # Edge case handling
            tmp = np.zeros((5, seg, 75), dtype=np.float32)
            all_subseq.append(tmp)

    if len(all_subseq) > 0:
        try:
            out = np.concatenate(all_subseq, axis=0)  # (5*N, seg, 75)
        except ValueError as e:
            print(f"Warning: Could not concatenate sub-sequences: {e}")
            # Create placeholder with expected shape
            out = np.zeros((5 * len(all_subseq), seg, 75), dtype=np.float32)
            # Fill with available data
            for i, subseq in enumerate(all_subseq):
                if i*5 + 5 <= out.shape[0]:  # Stay in bounds
                    out[i*5:i*5+5] = subseq
    else:
        out = np.zeros((0, seg, 75), dtype=np.float32)
    
    return out

#------------------------------------------------------------------------------
# Main Entry Points
#------------------------------------------------------------------------------

def sgn_preprocess_single_skeleton(
    skeleton_array,
    seg=20,
    dataset='NTU'
):
    """
    Process a single skeleton for SGN model evaluation.
    
    Replicates the test-time pipeline from SGN: collate_fn_fix_test -> Tolist_fix -> 
    sub_seq(train=2) -> 5-crops.
    
    Args:
        skeleton_array: np.ndarray - Input skeleton with shape (T, 75) or (T, 150)
        seg: int - Segment length for SGN model
        dataset: str - Dataset name in uppercase ('NTU', 'NTU120', 'ETRI')
        
    Returns:
        np.ndarray - Processed skeleton crops with shape (5, seg, 75)
    """
    # Handle NaN values
    skeleton_array = np.nan_to_num(skeleton_array)
    
    # Expand to 150 format if input is only 75
    if skeleton_array.shape[1] == 75:
        pad = np.zeros_like(skeleton_array)
        skeleton_array = np.concatenate([skeleton_array, pad], axis=1)  # => (T,150)

    # Use only first actor (consistent with training)
    skeleton_array = skeleton_array[:, :75]
    
    # Remove zero frames (frames where all values are zero)
    non_zero_mask = ~np.all(skeleton_array == 0, axis=1)
    non_zero_frames = skeleton_array[non_zero_mask]
    
    if len(non_zero_frames) == 0:
        # Edge case: all frames are zero
        non_zero_frames = np.zeros((1, skeleton_array.shape[1]), dtype=np.float32)
    
    # Process all 5 crops for test-time evaluation
    crops = []
    
    # If sequence is shorter than segment length, repeat last frame instead of zero-padding
    if len(non_zero_frames) < seg:
        last_frame = non_zero_frames[-1:]
        num_repeats = seg - len(non_zero_frames)
        processed_seq = np.vstack([non_zero_frames] + [last_frame] * num_repeats)
        # Just use one copy of this padded sequence 5 times
        for _ in range(5):
            crops.append(processed_seq)
    else:
        # Create 5 different random samplings
        num_frames = len(non_zero_frames)
        ave_duration = max(num_frames // seg, 1)
        
        for seed in range(5):
            np.random.seed(seed)  # Use fixed seeds for reproducibility
            offsets = np.multiply(list(range(seg)), ave_duration) + \
                     np.random.randint(ave_duration, size=seg)
            offsets = np.clip(offsets, 0, num_frames-1)
            crops.append(non_zero_frames[offsets])
        
        # Reset random seed
        np.random.seed(None)
    
    # Stack the crops
    out = np.stack(crops, axis=0)
    return out.astype(np.float32)  # shape => (5, seg, 75)

def mixformer_preprocess_single_skeleton(
    skeleton_array,
    split='test',
    p_interval=1,
    window_size=-1,
    random_rot=False,
    bone=False,
    vel=False
):
    """
    Process a single skeleton for MixFormer model evaluation.
    
    Replicates the test-time pipeline from MixFormer feeder_ntu.py.
    
    Args:
        skeleton_array: np.ndarray - Input skeleton with shape (T, 75)
        split: str - 'train' or 'test'
        p_interval: float or list - Sampling proportion or range
        window_size: int - Target frame count (-1 for default 64)
        random_rot: bool - Whether to apply random rotation
        bone: bool - Whether to convert to bone representation
        vel: bool - Whether to convert to velocity representation
        
    Returns:
        np.ndarray - Processed skeleton with shape (3, T, 25, 1)
    """
    # Handle NaN values
    skeleton_array = np.nan_to_num(skeleton_array)
    
    # Ensure we're only using first 75 dimensions (first actor)
    if skeleton_array.shape[1] > 75:
        skeleton_array = skeleton_array[:, :75]
    
    # Reshape to (C, T, V, M) format
    T = skeleton_array.shape[0]
    c = 3
    v = 25
    m = 1

    data_numpy = skeleton_array.reshape(T, v, c).transpose(2, 0, 1)  # => shape (3,T,25)
    data_numpy = data_numpy[..., np.newaxis]  # => (3, T, 25, 1)

    # Remove zero frames (frames where all values are zero)
    sum_over = data_numpy.sum(axis=0).sum(axis=-1).sum(axis=-1)  # shape (T,)
    valid = (sum_over != 0)
    valid_frame_num = valid.sum()

    if valid_frame_num == 0:
        # Edge case: all frames are zero
        valid_frame_num = 1
        
    # Crop and resize
    out = valid_crop_resize(
        data_numpy, 
        valid_frame_num, 
        p_interval=(p_interval if isinstance(p_interval, list) else [p_interval]), 
        window=64 if window_size < 0 else window_size
    )

    # Apply random rotation if specified
    if random_rot and split == 'train':
        out = random_rot(out, theta=0.3)

    # Convert to bone representation if specified
    if bone:
        C2, T2, V2, M2 = out.shape
        bone_data = np.zeros_like(out)
        for (v1, v2) in ntu_pairs:
            bone_data[:, :, v1-1, :] = out[:, :, v1-1, :] - out[:, :, v2-1, :]
        out = bone_data

    # Convert to velocity representation if specified
    if vel:
        C2, T2, V2, M2 = out.shape
        out_t = torch.from_numpy(out)
        out_t[:, :-1] = out_t[:, 1:] - out_t[:, :-1]
        out_t[:, -1] = 0
        out = out_t.numpy()

    return out  # shape => (3, T', 25, 1)
