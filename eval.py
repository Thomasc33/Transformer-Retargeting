#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import os
import pickle
import numpy as np
import sys
import traceback
import torch.nn.functional as F
from tqdm import tqdm
from data import datasets, load_data, get_cross_data


###############################################################################
# Utility function: dynamic import by string
###############################################################################
def import_class(import_str):
    """
    For dynamic importing: e.g., 'model.sgn.SGN'
    """
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exc()})')


###############################################################################
# Load the chosen anonymizer (Transformer / PMR / DMR)
###############################################################################
def load_anonymizer(model_type, model_path, device, args, ds=None):
    """
    Dynamically load and return an anonymization model.

    model_type : str
        'transformer', 'pmr', or 'dmr'
    model_path : str
        Path to the model weights (transformer_model_path or ./eval/pmr/model.pth, etc.)
    ds : dict
        The dataset config, used only for the transformer (e.g. ds= datasets[args.dataset])
    """
    if model_type == 'transformer':
        # -------------------------------------------------
        # Import your original autoencoder/transformer code
        # -------------------------------------------------
        from model.autoencoder import Model  # Adjust if needed

        # Build the transformer model
        # (We assume you have something like ds = datasets['ntu120'] to get needed config)
        autoenc_model = Model(
            num_class=ds['num_class'],
            num_point=ds['joints'],
            num_person=ds['max_actors'],
            graph=ds['graph'],
            graph_args=ds['graph_args'],
            debug=False
        ).to(device)

        # Load weights
        weights = torch.load(model_path, map_location='cpu')
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        autoenc_model.load_state_dict(weights)
        autoenc_model.eval()
        return autoenc_model

    elif model_type == 'pmr':
        # -------------------------------------------
        # Example: load PMR from ./eval/pmr/pmr.py
        # -------------------------------------------
        sys.path.append('./eval/pmr')
        from pmr import PMR  # or from .pmr import PMR if your structure is different

        model = PMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)
        weights = torch.load(model_path, map_location='cpu')
        model.load_state_dict(weights)
        model.set_eval(True)
        return model

    elif model_type == 'dmr':
        # -------------------------------------------
        # Example: load DMR from ./eval/dmr/dmr.py
        # -------------------------------------------
        sys.path.append('./eval/dmr')
        from dmr import DMR  # or from .dmr import DMR if your structure is different

        model = DMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)
        weights = torch.load(model_path, map_location='cpu')
        model.load_state_dict(weights)
        model.set_eval(True)
        return model

    else:
        raise ValueError(f"Unrecognized model_type={model_type}")


###############################################################################
# Data reshaping function used by the Transformer (64 frames)
###############################################################################
def prep_data(x):
    """
    For the Transformer-based model. 
    Input shape: (N, T, D) => output shape (N, C_in, T, V, M).
    We assume single-person skeleton => M=1, V=25, C_in=3, so D=75.
    """
    N = x.shape[0]
    T = x.shape[1]
    D = x.shape[2]
    M = 1
    V = 25
    C_in = 3

    # Ensure that M * V * C_in == D
    assert M * V * C_in == D, f"Mismatch in dimensions. D={D}, but expected {M*V*C_in}"

    # Reshape inputs: (N, T, D) => (N, T, M, V, C_in) => permute => (N, C_in, T, V, M)
    x = x.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
    return x


###############################################################################
# get_anonymized_paired for Transformer (teacher-forcing approach, 64 frames)
###############################################################################
@torch.no_grad()
def get_anonymized_paired_transformer(batch, model, prep_data_fn):
    """
    The 'batch' has (x1, x2, y1, y2, actors, actions).
    Uses teacher-forcing and returns the anonymized results in the same 
    4-dict format. This is your original logic for the autoencoder/transformer.
    """
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]   # batch size
    T = x1.shape[1]   # original seq length

    # Prepare data for the model
    x1_ = prep_data_fn(x1).cuda()
    x2_ = prep_data_fn(x2).cuda()
    y1_ = prep_data_fn(y1).cuda()
    y2_ = prep_data_fn(y2).cuda()

    # Reorder concatenation
    inputs  = torch.cat([x1_, x2_, y1_, y2_], dim=0)
    dummy   = torch.cat([x2_, x1_, y2_, y1_], dim=0)
    targets = torch.cat([y2_, y1_, x2_, x1_], dim=0)

    # Generate outputs (autoregressive inference)
    outputs = model(inputs, dummy, teacher_forcing_ratio=0.0)
    # outputs shape: (4N, C_in, T-1, V, M)

    # Reconstruct full sequence by prepending the first frame
    initial_frames = inputs[:, :, 0:1, :, :]  # (4N, C_in, 1, V, M)
    full_outputs = torch.cat([initial_frames, outputs], dim=2)  # => (4N, C_in, T, V, M)

    # Split
    x1_hat, x2_hat, y1_hat, y2_hat = torch.split(full_outputs, N, dim=0)

    # Reshape back to (N, T, D)
    D_out = x1.shape[2]
    def unpermute_and_reshape(x_):
        x_ = x_.permute(0, 2, 4, 3, 1).contiguous().view(N, T, D_out)
        return x_

    x1_hat = unpermute_and_reshape(x1_hat)  # => should be P2,A1
    x2_hat = unpermute_and_reshape(x2_hat)  # => should be P1,A2
    y1_hat = unpermute_and_reshape(y1_hat)  # => should be P2,A2
    y2_hat = unpermute_and_reshape(y2_hat)  # => should be P1,A1

    # Combine in the dictionary, 4 items per sample
    anonymized = []
    for i in range(N):
        anonymized.append({
            'skeleton': x1_hat[i].cpu(),
            'gt_skeleton': y2[i].cpu(),
            'retargeted_actor': actors[i, 1], 
            'original_actor': actors[i, 0],    
            'action': actions[i, 0]
        })
        anonymized.append({
            'skeleton': x2_hat[i].cpu(),
            'gt_skeleton': y1[i].cpu(),
            'retargeted_actor': actors[i, 0],
            'original_actor': actors[i, 1],
            'action': actions[i, 1]
        })
        anonymized.append({
            'skeleton': y1_hat[i].cpu(),
            'gt_skeleton': x2[i].cpu(),
            'retargeted_actor': actors[i, 1],
            'original_actor': actors[i, 0],
            'action': actions[i, 0]
        })
        anonymized.append({
            'skeleton': y2_hat[i].cpu(),
            'gt_skeleton': x1[i].cpu(),
            'retargeted_actor': actors[i, 0],
            'original_actor': actors[i, 1],
            'action': actions[i, 1]
        })

    # Cleanup
    del inputs, dummy, targets, outputs
    del x1_, x2_, y1_, y2_
    del initial_frames, full_outputs, x1_hat, x2_hat, y1_hat, y2_hat
    torch.cuda.empty_cache()

    return anonymized


###############################################################################
# get_anonymized_paired for DMR / PMR (no teacher-forcing, produces 75 frames)
###############################################################################
@torch.no_grad()
def get_anonymized_paired_dmr_pmr(batch, model, T=75):
    """
    For DMR/PMR which each produce 75 frames by default, and do not
    rely on teacher forcing.

    We'll create the same 4 dictionary items per sample:
       x1 => x1_hat => GT=y2
       x2 => x2_hat => GT=y1
       y1 => y1_hat => GT=x2
       y2 => y2_hat => GT=x1

    The model call is typically:  model.eval(x1, x2)
    which returns shape (T_out, D_out). We'll do that per sample.
    """
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]

    # Pad to T=75
    old_T = False
    if x1.shape[1] < T:
        old_T = x1.shape[1]
        pad_size = T - x1.shape[1]
        padding = torch.zeros(N, pad_size, x1.shape[2])
        x1 = torch.cat([x1, padding], dim=1)
        x2 = torch.cat([x2, padding], dim=1)
        y1 = torch.cat([y1, padding], dim=1)
        y2 = torch.cat([y2, padding], dim=1)

    anonymized = []

    for i in range(N):
        # shape is (T_in, 75). We add batch dim => (1, T_in, 75)
        x1_in = x1[i].unsqueeze(0).float().cuda()
        x2_in = x2[i].unsqueeze(0).float().cuda()
        y1_in = y1[i].unsqueeze(0).float().cuda()
        y2_in = y2[i].unsqueeze(0).float().cuda()
        
        # Reshape to (1, T_in, 25, 3) for DMR/PMR
        x1_in = x1_in.view(1, T, 25, 3)
        x2_in = x2_in.view(1, T, 25, 3)
        y1_in = y1_in.view(1, T, 25, 3)
        y2_in = y2_in.view(1, T, 25, 3)

        # Now call model.eval(...) as usual
        x1_hat = model.eval(x1_in, x2_in).squeeze(0).cpu()
        x2_hat = model.eval(x2_in, x1_in).squeeze(0).cpu()
        y1_hat = model.eval(y1_in, y2_in).squeeze(0).cpu()
        y2_hat = model.eval(y2_in, y1_in).squeeze(0).cpu()

        # Remove padded frames
        if old_T:
            x1_hat = x1_hat[:old_T, :]
            x2_hat = x2_hat[:old_T, :]
            y1_hat = y1_hat[:old_T, :]
            y2_hat = y2_hat[:old_T, :]

        anonymized.append({
            'skeleton': x1_hat,  
            'gt_skeleton': y2[i].cpu(),
            'retargeted_actor': actors[i, 1],
            'original_actor': actors[i, 0],
            'action': actions[i, 0]
        })
        anonymized.append({
            'skeleton': x2_hat,
            'gt_skeleton': y1[i].cpu(),
            'retargeted_actor': actors[i, 0],
            'original_actor': actors[i, 1],
            'action': actions[i, 1]
        })
        anonymized.append({
            'skeleton': y1_hat,
            'gt_skeleton': x2[i].cpu(),
            'retargeted_actor': actors[i, 1],
            'original_actor': actors[i, 0],
            'action': actions[i, 0]
        })
        anonymized.append({
            'skeleton': y2_hat,
            'gt_skeleton': x1[i].cpu(),
            'retargeted_actor': actors[i, 0],
            'original_actor': actors[i, 1],
            'action': actions[i, 1]
        })

    torch.cuda.empty_cache()
    return anonymized



###############################################################################
# MixFormer Evaluation (Action Recognition, Re-identification, MSE)
###############################################################################
def eval_skeleton_mixformer(paired_test, model, prep_data_fn, dataset_name, 
                            ar_model_weights, ri_model_weights, args):
    """
    Evaluate with:
      1) Action Recognition model (MixFormer-based)
      2) Re-identification model (MixFormer-based)
      3) Compute MSE between anonymized and ground-truth
    """
    # 1) AR model
    ar_model_class = 'model.ske_mixf.Model'
    ar_model_args = {
        'num_class': datasets[dataset_name]['num_class'],
        'num_point': 25,
        'num_person': 2,
        'graph': datasets[dataset_name]['graph'],
    }
    AR_Model = import_class(ar_model_class)
    ar_model = AR_Model(**ar_model_args)

    # Load AR weights
    ar_state_dict = torch.load(ar_model_weights, map_location='cpu')
    ar_state_dict = {k.replace('module.', ''): v for k, v in ar_state_dict.items()}
    ar_model.load_state_dict(ar_state_dict)
    ar_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar_model = ar_model.to(ar_device)
    ar_model.eval()

    # 2) RI model
    ri_model_class = 'model.ske_mixf.Model'
    ri_model_args = {
        'num_class': datasets[dataset_name]['num_actor'],
        'num_point': 25,
        'num_person': 2,
        'graph': datasets[dataset_name]['graph'],
    }
    RI_Model = import_class(ri_model_class)
    ri_model = RI_Model(**ri_model_args)

    # Load RI weights
    ri_state_dict = torch.load(ri_model_weights, map_location='cpu')
    ri_state_dict = {k.replace('module.', ''): v for k, v in ri_state_dict.items()}
    ri_model.load_state_dict(ri_state_dict)
    ri_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ri_model = ri_model.to(ri_device)
    ri_model.eval()

    total_samples = 0
    correct_action = 0
    correct_actor_ret = 0
    correct_actor_orig = 0
    mse_sum = 0.0
    mse_count = 0

    # Go through each paired batch
    for batch in tqdm(paired_test, desc='Evaluating with MixFormer'):
        # Depending on model_type, choose the anonymizing function
        if args.model_type == 'transformer':
            anonymized_data = get_anonymized_paired_transformer(batch, model, prep_data_fn)
        else:
            anonymized_data = get_anonymized_paired_dmr_pmr(batch, model, T=args.T)

        # For each anonymized skeleton, do MSE + AR + Re-ID
        ar_batch_data = []
        ar_labels = []
        ri_batch_data = []
        ri_ret_labels = []
        ri_orig_labels = []

        for item in anonymized_data:
            anonymized_skel = item['skeleton']   # shape (T_out, D_out)
            gt_skel = item['gt_skeleton']        # shape (T_in, D_in)
            action_label = item['action']
            retargeted_actor = item['retargeted_actor']
            original_actor = item['original_actor']

            # We must handle if T_out != T_in
            # For MSE, let's just truncate to the minimum length:
            t_out = anonymized_skel.shape[0]
            t_gt = gt_skel.shape[0]
            T_m = min(t_out, t_gt)
            anonymized_skel_ = anonymized_skel[:T_m]
            gt_skel_ = gt_skel[:T_m]

            # MSE
            mse = F.mse_loss(anonymized_skel_, gt_skel_)
            mse_sum += mse.item()
            mse_count += 1

            # Next, for AR with MixFormer, we need shape (C, T, V, M).
            # Typically C=3, V=25, M=2. If D_out=75 => single person => 25*3=75.
            # We'll do the same "expand with second person = 0" approach.
            T_cur = anonymized_skel.shape[0]  # actual length of anonymized
            D_cur = anonymized_skel.shape[1]
            V = 25
            C = 3
            M = 2

            # If D_cur != 75, there's a mismatch. Usually it should be 75 for single-person
            # so let's assert:
            assert D_cur == V*C, f"Dimension mismatch: D={D_cur}, expected {V*C}"

            # We'll forcibly also truncate/pad to a certain T if your AR model 
            # needs 64 frames. For MixFormer code, it might accept variable T, 
            # but you might want to keep them the same. 
            # For simplicity, let's just use T_cur as-is if the model can handle it.
            # Reshape: (T_cur, 75) => (T_cur, 25, 3)
            an_skel_reshape = anonymized_skel.view(T_cur, V, C)
            # zero for second person
            an_skel_reshape = an_skel_reshape.unsqueeze(1) # => (T_cur,1,V,C)
            zeros_ = torch.zeros_like(an_skel_reshape)
            an_skel_reshape = torch.cat([an_skel_reshape, zeros_], dim=1) # => (T_cur,2,V,C)
            # permute to (C,T,V,M)
            an_skel_reshape = an_skel_reshape.permute(3,0,2,1).contiguous()

            ar_batch_data.append(an_skel_reshape.detach().cpu().numpy())
            ar_labels.append(action_label)

            # Re-ID data
            ri_batch_data.append(an_skel_reshape.detach().cpu().numpy())
            ri_ret_labels.append(retargeted_actor)
            ri_orig_labels.append(original_actor)

        # Convert to Tensors for AR/RI
        ar_batch_data = torch.tensor(np.stack(ar_batch_data), dtype=torch.float32).to(ar_device)
        ar_labels = torch.tensor(ar_labels, dtype=torch.long).to(ar_device)

        ri_batch_data = torch.tensor(np.stack(ri_batch_data), dtype=torch.float32).to(ri_device)
        ri_ret_labels = torch.tensor(ri_ret_labels, dtype=torch.long).to(ri_device)
        ri_orig_labels = torch.tensor(ri_orig_labels, dtype=torch.long).to(ri_device)

        batch_size = ar_batch_data.size(0)
        total_samples += batch_size

        # Action Recognition
        with torch.no_grad():
            ar_output = ar_model(ar_batch_data)
            _, ar_predicted = torch.max(ar_output.data, 1)
            correct_action += (ar_predicted == ar_labels).sum().item()

        # Re-identification
        with torch.no_grad():
            ri_output = ri_model(ri_batch_data)
            _, ri_predicted = torch.max(ri_output.data, 1)
            correct_actor_ret += (ri_predicted == ri_ret_labels).sum().item()
            correct_actor_orig += (ri_predicted == ri_orig_labels).sum().item()

    action_accuracy = 0
    if total_samples > 0:
        action_accuracy = (correct_action / total_samples) * 100

    print(f'[MixFormer] Action Recognition Accuracy: {action_accuracy:.2f}%')

    print('[MixFormer] Re-identification Results:')
    ret_pct = (correct_actor_ret/total_samples)*100 if total_samples>0 else 0
    orig_pct = (correct_actor_orig/total_samples)*100 if total_samples>0 else 0
    print(f'  Predicted Retargeted Actor: {correct_actor_ret} / {total_samples} ({ret_pct:.2f}%)')
    print(f'  Predicted Original Actor:  {correct_actor_orig} / {total_samples} ({orig_pct:.2f}%)')
    neither = total_samples - correct_actor_ret - correct_actor_orig
    neither_pct = (neither / total_samples)*100 if total_samples>0 else 0
    print(f'  Predicted Neither Actor:   {neither} / {total_samples} ({neither_pct:.2f}%)')

    avg_mse = (mse_sum / mse_count) if mse_count > 0 else 0
    print(f'[MixFormer] Average MSE (anonymized vs. ground-truth): {avg_mse:.6f}')


###############################################################################
# SGN Evaluation (Action Recognition, Re-identification, MSE)
###############################################################################
def eval_skeleton_sgn(paired_test, model, prep_data_fn, dataset_name, 
                      ar_model_weights, ri_model_weights, args):
    """
    Evaluate with:
      1) Action Recognition model (SGN-based)
      2) Re-identification model (SGN-based)
      3) Compute MSE between anonymized and ground-truth
    """
    # AR model
    ar_model_class = 'model.sgn.SGN'
    ar_model_args = {
        'num_classes': datasets[dataset_name]['num_class'],
        'dataset': dataset_name,
        'seg': 20,  # segmentation hyperparam for SGN
    }
    AR_Model = import_class(ar_model_class)
    ar_model = AR_Model(**ar_model_args)
    ar_state_dict = torch.load(ar_model_weights, map_location='cpu')
    ar_state_dict = ar_state_dict['state_dict'] if 'state_dict' in ar_state_dict else ar_state_dict
    ar_state_dict = {k.replace('module.', ''): v for k, v in ar_state_dict.items()}
    ar_model.load_state_dict(ar_state_dict)
    ar_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar_model = ar_model.to(ar_device)
    ar_model.eval()

    # RI model
    ri_model_class = 'model.sgn.SGN'
    ri_model_args = {
        'num_classes': datasets[dataset_name]['num_actor'],
        'dataset': dataset_name,
        'seg': 20,
    }
    RI_Model = import_class(ri_model_class)
    ri_model = RI_Model(**ri_model_args)
    ri_state_dict = torch.load(ri_model_weights, map_location='cpu')
    ri_state_dict = ri_state_dict['state_dict'] if 'state_dict' in ri_state_dict else ri_state_dict
    ri_state_dict = {k.replace('module.', ''): v for k, v in ri_state_dict.items()}
    ri_model.load_state_dict(ri_state_dict)
    ri_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ri_model = ri_model.to(ri_device)
    ri_model.eval()

    total_samples = 0
    correct_action = 0
    correct_actor_ret = 0
    correct_actor_orig = 0
    mse_sum = 0.0
    mse_count = 0

    for batch in tqdm(paired_test, desc='Evaluating with SGN'):
        # Switch anonymized function
        if args.model_type == 'transformer':
            anonymized_data = get_anonymized_paired_transformer(batch, model, prep_data_fn)
        else:
            anonymized_data = get_anonymized_paired_dmr_pmr(batch, model, T=args.T)

        ar_batch_data = []
        ar_labels = []
        ri_batch_data = []
        ri_ret_labels = []
        ri_orig_labels = []

        for item in anonymized_data:
            anonymized_skel = item['skeleton']  # shape (T_out, D_out)
            gt_skel = item['gt_skeleton']       # shape (T_in, D_in)
            action_label = item['action']
            retargeted_actor = item['retargeted_actor']
            original_actor = item['original_actor']

            # MSE (truncate/pad if needed)
            T_out = anonymized_skel.shape[0]
            T_in = gt_skel.shape[0]
            T_m = min(T_out, T_in)
            anonymized_skel_ = anonymized_skel[:T_m]
            gt_skel_ = gt_skel[:T_m]

            mse = F.mse_loss(anonymized_skel_, gt_skel_)
            mse_sum += mse.item()
            mse_count += 1

            # Now the SGN logic: we have single-person skeleton => (T, 75) if 25*3
            # We segment to 'seg' frames, e.g. 20.
            num_joint = 25
            in_channel = 3
            seg = ar_model_args['seg']  # 20
            D_cur = anonymized_skel.shape[1]

            assert D_cur == num_joint*in_channel, f"Dimension mismatch: D={D_cur}, expected {num_joint*in_channel}"

            # Step: if T_out < seg => pad, if T_out > seg => sample
            # Reshape: (T_out, 75)
            anonymized_skel_resh = anonymized_skel.view(T_out, -1)
            if T_out < seg:
                pad_size = seg - T_out
                padding = torch.zeros(pad_size, anonymized_skel_resh.shape[1])
                anonymized_skel_resh = torch.cat([anonymized_skel_resh, padding], dim=0)
            else:
                indices = np.linspace(0, T_out - 1, seg).astype(int)
                anonymized_skel_resh = anonymized_skel_resh[indices, :]

            anonymized_skel_resh = anonymized_skel_resh.unsqueeze(0)  # => (1, seg, 75)
            ar_batch_data.append(anonymized_skel_resh)
            ar_labels.append(int(action_label))

            ri_batch_data.append(anonymized_skel_resh)
            ri_ret_labels.append(int(retargeted_actor))
            ri_orig_labels.append(int(original_actor))

        # Stack them
        ar_batch_data = torch.cat(ar_batch_data, dim=0)  # shape (batch_size, seg, 75)
        ar_labels = torch.tensor(ar_labels, dtype=torch.long)

        ri_batch_data = torch.cat(ri_batch_data, dim=0)  # shape (batch_size, seg, 75)
        ri_ret_labels = torch.tensor(ri_ret_labels, dtype=torch.long)
        ri_orig_labels = torch.tensor(ri_orig_labels, dtype=torch.long)

        batch_size = ar_batch_data.size(0)
        total_samples += batch_size

        # AR
        with torch.no_grad():
            ar_output = ar_model(ar_batch_data.to(ar_device))
            if ar_output.dim() == 3 and ar_output.size(0) == 1:
                ar_output = ar_output.squeeze(0)
            _, ar_predicted = torch.max(ar_output.cpu(), 1)
            correct_action += (ar_predicted == ar_labels).sum().item()

        # RI
        with torch.no_grad():
            ri_output = ri_model(ri_batch_data.to(ri_device))
            if ri_output.dim() == 3 and ri_output.size(0) == 1:
                ri_output = ri_output.squeeze(0)
            _, ri_predicted = torch.max(ri_output.cpu(), 1)
            correct_actor_ret += (ri_predicted == ri_ret_labels).sum().item()
            correct_actor_orig += (ri_predicted == ri_orig_labels).sum().item()

    action_accuracy = (correct_action / total_samples * 100) if total_samples > 0 else 0
    print(f'[SGN] Action Recognition Accuracy: {action_accuracy:.2f}%')

    print('[SGN] Re-identification Results:')
    ret_pct = (correct_actor_ret/total_samples)*100 if total_samples>0 else 0
    orig_pct = (correct_actor_orig/total_samples)*100 if total_samples>0 else 0
    print(f'  Predicted Retargeted Actor: {correct_actor_ret} / {total_samples} ({ret_pct:.2f}%)')
    print(f'  Predicted Original Actor:  {correct_actor_orig} / {total_samples} ({orig_pct:.2f}%)')
    neither = total_samples - correct_actor_ret - correct_actor_orig
    neither_pct = (neither / total_samples)*100 if total_samples>0 else 0
    print(f'  Predicted Neither Actor:   {neither} / {total_samples} ({neither_pct:.2f}%)')

    avg_mse = mse_sum / mse_count if mse_count > 0 else 0
    print(f'[SGN] Average MSE (anonymized vs. ground-truth): {avg_mse:.6f}')


###############################################################################
# Main script entry point
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate anonymized skeleton data with AR/RI models.")

    # Dataset / path arguments
    parser.add_argument('--dataset', default='ntu120', choices=['ntu120', 'ntu', 'etri'],
                        help='Which dataset to use (default: ntu120).')
    parser.add_argument('--setting', default='cs', type=str,
                        help='Cross-subject (cs) or cross-view (cv) for dataset splitting (default: cs).')

    # Model selection
    parser.add_argument('--model_type', default='transformer',
                        choices=['transformer', 'pmr', 'dmr'],
                        help='Which anonymization model to evaluate.')
    parser.add_argument('--transformer_model_path', default='model.pth', type=str,
                        help='Path to the trained transformer model weights (if using --model_type=transformer).')
    parser.add_argument('--ar_model_weights', default='', type=str,
                        help='Path to the action recognition model weights.')
    parser.add_argument('--ri_model_weights', default='', type=str,
                        help='Path to the re-identification model weights.')
    parser.add_argument('--eval_model', default='mixformer', choices=['mixformer', 'sgn'],
                        help='Which AR/RI architecture to use (mixformer or sgn).')

    # Batch / caching / sampling
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Unpaired batch size (default: 32).')
    parser.add_argument('--paired_batch_size', default=8, type=int,
                        help='Paired batch size (default: 8).')
    parser.add_argument('--test_samples', default=5000, type=int,
                        help='Number of test samples for get_cross_data (default: 5000).')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached data if available.')
    parser.add_argument('--save_cache', action='store_true',
                        help='Save generated data to cache pkl files.')
    parser.add_argument('--T', default=64, type=int,
                        help='Number of frames: 64 for transformer, 75 for PMR/DMR, etc.')

    args = parser.parse_args()
    print("Arguments:", args)

    # Load the data
    X = load_data(args.dataset)  # e.g., loads data from data/ntu120/ntu120.pkl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Possibly load paired data from cache
    paired_path = f'data/{args.dataset}_{args.setting}_paired.pkl'
    if args.use_cache and os.path.exists(paired_path):
        with open(paired_path, 'rb') as f:
            paired_data = pickle.load(f)
            paired_train = paired_data['train']
            paired_test = paired_data['test']
            print('Paired data loaded from pickle file:', paired_path)
    else:
        # Generate paired data
        paired_train, paired_test = get_cross_data(
            X,
            args.dataset,
            args.setting,
            args.paired_batch_size,
            return_loader=True,
            train_samples=args.batch_size,
            test_samples=args.test_samples
        )
        if args.save_cache:
            with open(paired_path, 'wb') as f:
                pickle.dump({'train': paired_train, 'test': paired_test}, f)
                print('Paired data saved to pickle file:', paired_path)

    # Prepare to load the anonymizer
    ds = datasets[args.dataset]  # e.g., ds = {'num_class':120, 'joints':25, ...}

    if args.model_type == 'transformer':
        # We'll use the user-specified path
        anonymizer_path = args.transformer_model_path
    else:
        # We assume your PMR or DMR model is at ./eval/<model_type>/model.pth
        anonymizer_path = f'./eval/{args.model_type}/{args.dataset}.pt'

    # Load the anonymizer
    anonymizer_model = load_anonymizer(args.model_type, anonymizer_path, device, args, ds=ds)

    # Evaluate with MixFormer or SGN
    if args.eval_model == 'mixformer':
        print('Evaluating with MixFormer')
        eval_skeleton_mixformer(
            paired_test,
            anonymizer_model,
            prep_data,           # only used if transformer, harmless if not
            args.dataset,
            args.ar_model_weights,
            args.ri_model_weights,
            args
        )
    else:
        print('Evaluating with SGN')
        eval_skeleton_sgn(
            paired_test,
            anonymizer_model,
            prep_data,           # only used if transformer
            args.dataset,
            args.ar_model_weights,
            args.ri_model_weights,
            args
        )


if __name__ == '__main__':
    main()
