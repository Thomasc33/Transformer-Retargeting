###############################################################################
# eval/preprocessing.py
#
# A standalone file replicating the test-time preprocessing logic from:
#   - SGN (main.py + data.py "collate_fn_fix_test" + sub_seq, etc.)
#   - Skeleton-MixFormer (feeder_ntu.py + tools.py, with bone/vel logic, etc.)
#
# Provides two main entry points:
#   sgn_preprocess_single_skeleton(...)
#   mixformer_preprocess_single_skeleton(...)
#
# so that your evaluation code can transform each sample exactly as the original
# code's test pipeline does, ensuring consistent evaluation results.
###############################################################################

import numpy as np
import torch
import math

###############################################################################
# 1) bone_pairs.py (referenced by feeder_ntu.py if bone=True)
###############################################################################
# Typically, we have "from .bone_pairs import ntu_pairs" for bone modality.
# The code typically uses the standard 25-joint NTU skeleton pairs.
# We replicate it here exactly.

ntu_pairs = (
    (2,1), (3,2), (4,3), (5,4), (6,5),
    (7,6), (8,7), (9,8), (10,9), (11,10),
    (12,11), (13,12), (14,13), (15,14), (16,15),
    (17,16), (18,17), (19,18), (20,19), (21,2),
    (22,21), (23,22), (24,23), (25,24)
)

###############################################################################
# 2) tools.py logic (MixFormer uses these to do valid_crop_resize, random_rot, etc.)
###############################################################################

def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    """
    Crops or resizes the sequence to 'window' length. The logic used in feeder_ntu.py.
    - data_numpy: shape (C, T, V, M)
    - valid_frame_num: scalar # of valid frames
    - p_interval: random proportion or a range
    - window: the final #frames to resize to
    Returns shape (C, window, V, M).
    """
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

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
        # single p_interval value
        p = p_interval
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]
        cropped_length = data.shape[1]

    # Resize to 'window' frames with interpolation
    import torch.nn.functional as F
    import torch

    data_torch = torch.tensor(data, dtype=torch.float)   # (C, cropped_length, V, M)
    # Permute to (1,1,C*V*M,cropped_length) for F.interpolate usage
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
    Not typically used in test, but part of tools.py
    """
    import random
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]

def mean_subtractor(data_numpy, mean):
    """
    Not typically used for test, naive version, from tools.py
    """
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy

def auto_pading(data_numpy, size, random_pad=False):
    """
    tools.py
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

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    """
    Not typically used at test time, from tools.py
    """
    import random
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    import torch
    theta = torch.tensor(np.array([a, s, t_x, t_y])).transpose(0,1) # shape (T,4)
    # We'll omit full random_move detail or keep it as is:
    # left in the code for completeness only
    return data_numpy

def _rot(rot):
    # from tools.py for random_rot
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
    data_numpy: shape (C,T,V,M). Rotate around x,y,z by up to theta rad.
    """
    import torch
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    # shape => (T, C, V*M)
    data_torch = data_torch.permute(1,0,2,3).contiguous().view(T,C,V*M)
    rot = torch.zeros(T,3).uniform_(-theta, theta)
    rot_mat = _rot(rot)
    data_torch = torch.bmm(rot_mat, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1,0,2,3).contiguous()
    return data_torch.numpy()

def openpose_match(data_numpy):
    """
    Not used by your code, but included for completeness from tools.py
    """
    C, T, V, M = data_numpy.shape
    score = data_numpy[2, :, :, :].sum(axis=1)
    rank = (-score[0:T-1]).argsort(axis=1).reshape(T-1, M)

    xy1 = data_numpy[0:2, 0:T-1, :, :].reshape(2, T-1, V, M, 1)
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T-1, V, 1, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T-1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert(np.all(forward_map >= 0))

    for t in range(T-1):
        forward_map[t+1] = forward_map[t+1][forward_map[t]]

    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]]
    data_numpy = new_data_numpy

    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]
    return data_numpy

###############################################################################
# 3) SGN data preprocessing logic for test-time (like collate_fn_fix_test).
#
# In SGN's official code, test time calls:
#   test_loader = ntu_loaders.get_test_loader(batch_size=32, ...)
#   which uses collate_fn_fix_test -> Tolist_fix(..., train=2) -> sub_seq(...).
#
# That produces *5 sub-sequences* per sample. The model then averages the output.
###############################################################################

def turn_two_to_one(seq):
    """
    From sgn data.py => merges 2-person skeleton into 1-person if possible,
    but if both present, it doubles the frames. We replicate as-is.
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
    The 'sub_seq' method from data.py used inside Tolist_fix for SGN.
    - If T < seg => we pad with zeros
    - If T >= seg => we pick offset frames
    - If train=2 => we produce 5 distinct crops (like test-time augmentation)
    """
    import math

    group = seg
    if dataset in ['SYSU','SYSU_same']:
        seq = seq[::2, :]

    if seq.shape[0] < seg:
        pad = np.zeros((seg - seq.shape[0], seq.shape[1])).astype(np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    ave_duration = seq.shape[0] // group

    if train == 1:
        offsets = np.multiply(list(range(group)), ave_duration) + \
                  np.random.randint(ave_duration, size=group)
        seq_ = seq[offsets]
        seqs.append(seq_)
    elif train == 2:
        # We produce 5 versions
        offsets1 = np.multiply(list(range(group)), ave_duration) + \
                   np.random.randint(ave_duration, size=group)
        offsets2 = np.multiply(list(range(group)), ave_duration) + \
                   np.random.randint(ave_duration, size=group)
        offsets3 = np.multiply(list(range(group)), ave_duration) + \
                   np.random.randint(ave_duration, size=group)
        offsets4 = np.multiply(list(range(group)), ave_duration) + \
                   np.random.randint(ave_duration, size=group)
        offsets5 = np.multiply(list(range(group)), ave_duration) + \
                   np.random.randint(ave_duration, size=group)

        seqs.append(seq[offsets1])
        seqs.append(seq[offsets2])
        seqs.append(seq[offsets3])
        seqs.append(seq[offsets4])
        seqs.append(seq[offsets5])
    return seqs

def sgn_Tolist_fix(x_list, seg=20, dataset='NTU', is_test=True):
    """
    This replicates the 'Tolist_fix' portion for *test* usage in SGN.
    We'll do:
      1) turn_two_to_one
      2) sub_seq(..., train=2) => produce 5 sub-seqs
    Then we return shape (5*N, seg, D).
    If is_test=True, we do the train=2 logic. If is_test=False, we do train=1.
    x_list: a list of raw skeleton arrays, each shape (T, D=150) or (T, 150).
    Returns: a single NumPy array shape (5*N, seg, 75) after merging two-to-one.
    """
    all_subseq = []
    for x_ in x_list:
        # "turn_two_to_one" => merges or duplicates frames if two-person
        x_merged = turn_two_to_one(x_)

        # Now for each merged skeleton, we do sub_seq
        # But note that 'turn_two_to_one' can produce multiple frames
        # We'll feed the entire merged to sub_seq:
        # Actually, in data.py, it calls sub_seq once per sample, not per frame.
        # So we do it once:
        sub_seqs = sgn_sub_seq([], x_merged, seg=seg, train=2 if is_test else 1, dataset=dataset)
        # sub_seqs is a list of 5 arrays each shape (seg, 75)
        # We'll cat them
        if len(sub_seqs) > 0:
            # shape => (5, seg, 75)
            sub_stack = np.stack(sub_seqs, axis=0)  # shape (5, seg, 75)
            all_subseq.append(sub_stack)
        else:
            # edge-case
            tmp = np.zeros((5, seg, 75), dtype=np.float32)
            all_subseq.append(tmp)

    if len(all_subseq) > 0:
        out = np.concatenate(all_subseq, axis=0)  # (5*N, seg, 75)
    else:
        out = np.zeros((0, seg, 75), dtype=np.float32)
    return out

def sgn_preprocess_single_skeleton(
    skeleton_array,
    seg=20,
    dataset='NTU'
):
    """
    **Primary function** for test-time:
      - Takes a single skeleton of shape (T, 150) if 2-person, or (T,75) if 1-person, 
        but we unify to (T,150) with second person = 0 if needed.
      - Replicates the EXACT logic from SGN's test-time pipeline: 
        "collate_fn_fix_test -> Tolist_fix -> sub_seq(train=2) -> 5-crops"
      - Returns a NumPy array of shape (5, seg, 75), meaning 5 test crops.

    SGN code eventually does "output = model(...)" => shape (5, #classes)
    and then they average it. So you can do the same in your eval script:
      For each sample, run sgn_preprocess_single_skeleton, feed to SGN, average.

    Usage:
      arr_5 = sgn_preprocess_single_skeleton(my_skel, seg=20, dataset='NTU')
      # shape => (5, 20, 75)
    """
    # If the input is (T,75), we can stack zeros to get (T,150) or we just pass it in as is
    # Actually in original code, "turn_two_to_one" expects (T, 150).
    # So if we only have 1-person skeleton (T,75), we'll replicate the approach:
    if skeleton_array.shape[1] == 75:
        # Expand to 150 by padding second person
        pad = np.zeros_like(skeleton_array)
        skeleton_array = np.concatenate([skeleton_array, pad], axis=1)  # => (T,150)

    # We'll process as if we have a single sample in a batch
    x_list = [skeleton_array]
    # This returns shape (5*N, seg, 75) => (5, seg, 75)
    out = sgn_Tolist_fix(x_list, seg=seg, dataset=dataset, is_test=True)
    return out  # shape => (5, seg, 75)

###############################################################################
# 4) Skeleton MixFormer feeder_ntu.py logic for test-time
###############################################################################
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
    Replicates the feeder_ntu.Feeder code for test-time usage on a single skeleton.
    - skeleton_array shape => (T, 25*3) = (T, 75), single-person
    - We'll do the same "valid_crop_resize" logic:
        1) compute valid_frame_num
        2) valid_crop_resize(..., p_interval, window_size)
        3) if random_rot and split=='train', do random_rot
        4) if bone => convert to bone
        5) if vel => convert to velocity
    - Then produce shape (C, T', V, M). Usually M=1, V=25, C=3 => so final shape (3, T', 25, 1).

    This matches the test pipeline in feeder_ntu.py => __getitem__(self) when split=='test'.

    Usage:
      out = mixformer_preprocess_single_skeleton(my_skel, split='test', bone=False, vel=False)
      # out => shape (C, T', 25, 1)
    """
    # 1) shape => (T,75). We'll interpret "C=3, V=25, M=1" => so let's rearrange to (3,T,25,1).
    # But first we must find valid_frame_num = # of non-zero frames:
    #  (the official code does: valid_frame_num = sum( data_numpy.sum(0).sum(-1).sum(-1) != 0 )
    # but they store data in shape N,C,T,V,M. We'll replicate that approach in a local manner.

    # We'll treat this single skeleton as if shape => (C,T,V,M). So let's do:
    T = skeleton_array.shape[0]
    # Create a (C,T,V,M) => (3,T,25,1)
    c = 3
    v = 25
    m = 1

    data_numpy = skeleton_array.reshape(T, v, c).transpose(2,0,1)  # => shape (3,T,25)
    data_numpy = data_numpy[..., np.newaxis]                       # => (3, T, 25, 1)

    # valid_frame_num is how many frames are non-zero. i.e. a frame is zero if all = 0
    # Summation shape => data_numpy.sum(axis=0).sum(axis=-1).sum(axis=-1) => shape (T,)
    sum_over = data_numpy.sum(axis=0).sum(axis=-1).sum(axis=-1)  # shape (T,)
    # A frame is valid if sum != 0
    valid = (sum_over != 0)
    valid_frame_num = valid.sum()

    # 2) valid_crop_resize
    # The code does: data = valid_crop_resize(data_numpy, valid_frame_num, p_interval, window_size)
    out = valid_crop_resize(data_numpy, valid_frame_num, p_interval=(p_interval if isinstance(p_interval, list) else [p_interval]), window=64 if window_size<0 else window_size)
    # Now out shape => (3, outT, 25, 1)

    # 3) if random_rot and self.split=='train', then do random rotation. 
    # For test, typically random_rot=False. But if you pass random_rot=True manually, we do it:
    if random_rot and split=='train':
        out = random_rot(out, theta=0.3)

    # 4) bone transform if bone=True
    if bone:
        # from .bone_pairs import ntu_pairs
        # shape => (C, T, V, M)
        # We'll do bone_data[v1-1] = data[v1-1] - data[v2-1]
        # Note that joints are 1-based in the pairs, so we do (v1-1).
        # We must be careful with indexing. We'll do it in place.
        C2, T2, V2, M2 = out.shape
        bone_data = np.zeros_like(out)
        # replicate the official code:
        for (v1, v2) in ntu_pairs:
            bone_data[:, :, v1-1, :] = out[:, :, v1-1, :] - out[:, :, v2-1, :]
        out = bone_data

    # 5) velocity transform if vel=True
    if vel:
        # out[:, :-1] = out[:, 1:] - out[:, :-1]
        # out[:, -1] = 0
        # shape => (C,T,V,M)
        C2, T2, V2, M2 = out.shape
        # We'll convert to a PyTorch Tensor for convenience:
        import torch
        out_t = torch.from_numpy(out)
        out_t[:, :-1] = out_t[:, 1:] - out_t[:, :-1]
        out_t[:, -1] = 0
        out = out_t.numpy()

    return out  # shape => (3, T', 25, 1)

###############################################################################
# End of file: eval/preprocessing.py
###############################################################################
