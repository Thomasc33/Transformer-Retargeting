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
import pandas as pd

# Your dataset / data loader code
from data import datasets, load_data, get_cross_data

# Conditionally import the new preprocessing functions only if they exist
original_sys_path = sys.path.copy()
try:
    sys.path.append(os.path.abspath('eval'))
    from preprocess import (
        sgn_preprocess_single_skeleton,
        mixformer_preprocess_single_skeleton
    )
    from sgn_data import (
        NTUDataLoaders, 
        AverageMeter
    )
finally:
    sys.path = original_sys_path

###############################################################################
# Utility function: dynamic import by string
###############################################################################
def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exc()})')

###############################################################################
# Load the chosen anonymizer (Transformer / PMR / DMR / RAW)
###############################################################################
def load_anonymizer(model_type, model_path, device, args, ds=None):
    """
    Dynamically load and return an anonymization model or None (for raw).
      model_type : str
          'transformer', 'pmr', 'dmr', or 'raw'
      model_path : str
          Path to the model weights (if not raw)
      ds : dict
          The dataset config, used only for transformer
    """
    if model_type == 'raw':
        return None

    elif model_type == 'transformer':
        from model.autoencoder import Model
        autoenc_model = Model(
            num_class=ds['num_class'],
            num_point=ds['joints'],
            num_person=ds['max_actors'],
            graph=ds['graph'],
            graph_args=ds['graph_args'],
            debug=False
        ).to(device)

        weights = torch.load(model_path, map_location='cpu')
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        autoenc_model.load_state_dict(weights)
        autoenc_model.eval()
        return autoenc_model

    elif model_type == 'pmr':
        sys.path.append('./eval/pmr')
        from pmr import PMR
        model = PMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)
        weights = torch.load(model_path, map_location='cpu')
        model.load_state_dict(weights)
        model.set_eval(True)
        return model

    elif model_type == 'dmr':
        sys.path.append('./eval/dmr')
        from dmr import DMR
        model = DMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)
        weights = torch.load(model_path, map_location='cpu')
        model.load_state_dict(weights)
        model.set_eval(True)
        return model

    else:
        raise ValueError(f"Unrecognized model_type={model_type}")

###############################################################################
# If needed: "prep_data" for the Transformer
###############################################################################
def prep_data(x):
    N, T, D = x.shape
    M = 1
    V = 25
    C_in = 3
    assert D == (V*C_in), f"Unexpected D={D}. Should be 75 for single-person with 25 joints."
    x = x.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
    return x

###############################################################################
# Identity / raw
###############################################################################
@torch.no_grad()
def get_anonymized_paired_raw(batch):
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    out = []
    for i in range(N):
        out.append({
            'skeleton': x1[i].cpu(), # p1 a1
            'gt_skeleton': y2[i].cpu(), # p2 a1
            'retargeted_actor': actors[i, 1] - 1, # p2
            'original_actor': actors[i, 0] - 1, # p1
            'action': actions[i, 0] - 1 # a1
        })
        out.append({
            'skeleton': x2[i].cpu(), # p2 a2
            'gt_skeleton': y1[i].cpu(), # p1 a2
            'retargeted_actor': actors[i, 0] - 1, # p1
            'original_actor': actors[i, 1] - 1, # p2
            'action': actions[i, 1] - 1 # a2
        })
        out.append({
            'skeleton': y1[i].cpu(),
            'gt_skeleton': x2[i].cpu(),
            'retargeted_actor': actors[i, 1] - 1,
            'original_actor': actors[i, 0] - 1,
            'action': actions[i, 0] - 1
        })
        out.append({
            'skeleton': y2[i].cpu(),
            'gt_skeleton': x1[i].cpu(),
            'retargeted_actor': actors[i, 0] - 1,
            'original_actor': actors[i, 1] - 1,
            'action': actions[i, 1] - 1
        })
    return out

###############################################################################
# Transformer
###############################################################################
@torch.no_grad()
def get_anonymized_paired_transformer(batch, model, prep_data_fn):
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    T = x1.shape[1]

    x1_ = prep_data_fn(x1).cuda()
    x2_ = prep_data_fn(x2).cuda()
    y1_ = prep_data_fn(y1).cuda()
    y2_ = prep_data_fn(y2).cuda()

    inputs  = torch.cat([x1_, x2_, y1_, y2_], dim=0)
    dummy   = torch.cat([x2_, x1_, y2_, y1_], dim=0)
    targets = torch.cat([y2_, y1_, x2_, x1_], dim=0)

    outputs = model(inputs, dummy, teacher_forcing_ratio=0.0)
    first_frames = inputs[:, :, 0:1, :, :]
    full_out = torch.cat([first_frames, outputs], dim=2)

    x1_hat, x2_hat, y1_hat, y2_hat = torch.split(full_out, N, dim=0)
    D = x1.shape[2]
    def unreshape(x_):
        return x_.permute(0, 2, 4, 3, 1).contiguous().view(N, T, D)

    x1_hat = unreshape(x1_hat)
    x2_hat = unreshape(x2_hat)
    y1_hat = unreshape(y1_hat)
    y2_hat = unreshape(y2_hat)

    out = []
    for i in range(N):
        out.append({
            'skeleton': x1_hat[i].cpu(),
            'gt_skeleton': y2[i].cpu(),
            'retargeted_actor': actors[i, 1] - 1,
            'original_actor': actors[i, 0] - 1,
            'action': actions[i, 0] - 1
        })
        out.append({
            'skeleton': x2_hat[i].cpu(),
            'gt_skeleton': y1[i].cpu(),
            'retargeted_actor': actors[i, 0] - 1,
            'original_actor': actors[i, 1] - 1,
            'action': actions[i, 1] - 1
        })
        out.append({
            'skeleton': y1_hat[i].cpu(),
            'gt_skeleton': x2[i].cpu(),
            'retargeted_actor': actors[i, 1] - 1,
            'original_actor': actors[i, 0] - 1,
            'action': actions[i, 0] - 1
        })
        out.append({
            'skeleton': y2_hat[i].cpu(),
            'gt_skeleton': x1[i].cpu(),
            'retargeted_actor': actors[i, 0] - 1,
            'original_actor': actors[i, 1] - 1,
            'action': actions[i, 1] - 1
        })
    return out

###############################################################################
# PMR / DMR
###############################################################################
@torch.no_grad()
def get_anonymized_paired_dmr_pmr(batch, model, T=75):
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    old_T = x1.shape[1]
    if old_T < T:
        pad_size = T - old_T
        padding = torch.zeros(N, pad_size, x1.shape[2])
        x1 = torch.cat([x1, padding], dim=1)
        x2 = torch.cat([x2, padding], dim=1)
        y1 = torch.cat([y1, padding], dim=1)
        y2 = torch.cat([y2, padding], dim=1)

    out = []
    for i in range(N):
        x1_in = x1[i].unsqueeze(0).float().cuda().view(1, T, 25, 3)
        x2_in = x2[i].unsqueeze(0).float().cuda().view(1, T, 25, 3)
        y1_in = y1[i].unsqueeze(0).float().cuda().view(1, T, 25, 3)
        y2_in = y2[i].unsqueeze(0).float().cuda().view(1, T, 25, 3)

        x1_hat = model.eval(x1_in, x2_in).squeeze(0).cpu()
        x2_hat = model.eval(x2_in, x1_in).squeeze(0).cpu()
        y1_hat = model.eval(y1_in, y2_in).squeeze(0).cpu()
        y2_hat = model.eval(y2_in, y1_in).squeeze(0).cpu()

        if old_T < T:
            x1_hat = x1_hat[:old_T]
            x2_hat = x2_hat[:old_T]
            y1_hat = y1_hat[:old_T]
            y2_hat = y2_hat[:old_T]

        out.append({
            'skeleton': x1_hat,
            'gt_skeleton': y2[i].cpu(),
            'retargeted_actor': actors[i, 1] - 1,
            'original_actor': actors[i, 0] - 1,
            'action': actions[i, 0] - 1
        })
        out.append({
            'skeleton': x2_hat,
            'gt_skeleton': y1[i].cpu(),
            'retargeted_actor': actors[i, 0] - 1,
            'original_actor': actors[i, 1] - 1,
            'action': actions[i, 1] - 1
        })
        out.append({
            'skeleton': y1_hat,
            'gt_skeleton': x2[i].cpu(),
            'retargeted_actor': actors[i, 1] - 1,
            'original_actor': actors[i, 0] - 1,
            'action': actions[i, 0] - 1
        })
        out.append({
            'skeleton': y2_hat,
            'gt_skeleton': x1[i].cpu(),
            'retargeted_actor': actors[i, 0] - 1,
            'original_actor': actors[i, 1] - 1,
            'action': actions[i, 1] - 1
        })
    return out

###############################################################################
# MixFormer-based AR/RI
###############################################################################
def eval_skeleton_mixformer(
    paired_test, model, dataset_name, 
    ar_model_weights, ri_model_weights, args
):
    try:
        from preprocess import mixformer_preprocess_single_skeleton
    except ImportError:
        print("Could not import 'mixformer_preprocess_single_skeleton'.")
        def mixformer_preprocess_single_skeleton(x):
            return x

    AR_Model = import_class('model.ske_mixf.Model')
    ar_model = AR_Model(
        num_class=datasets[dataset_name]['num_class'],
        num_point=25,
        num_person=2,
        graph=datasets[dataset_name]['graph']
    )
    ar_sd = torch.load(ar_model_weights, map_location='cpu')
    ar_sd = {k.replace('module.', ''): v for k, v in ar_sd.items()}
    ar_model.load_state_dict(ar_sd)
    ar_model = ar_model.to(args.device)
    ar_model.eval()

    RI_Model = import_class('model.ske_mixf.Model')
    ri_model = RI_Model(
        num_class=datasets[dataset_name]['num_class'], # override
        num_point=25,
        num_person=2,
        graph=datasets[dataset_name]['graph']
    )
    ri_sd = torch.load(ri_model_weights, map_location='cpu')
    ri_sd = {k.replace('module.', ''): v for k, v in ri_sd.items()}
    ri_model.load_state_dict(ri_sd)
    ri_model = ri_model.to(args.device)
    ri_model.eval()

    total_samples = 0
    correct_action = 0
    correct_actor_ret = 0
    correct_actor_orig = 0
    mse_sum = 0.0
    mse_count = 0

    def anonymize_batch(batch):
        if args.model_type == 'raw':
            return get_anonymized_paired_raw(batch)
        elif args.model_type == 'transformer':
            return get_anonymized_paired_transformer(batch, model, prep_data)
        else:
            return get_anonymized_paired_dmr_pmr(batch, model, T=args.T)

    for batch in tqdm(paired_test, desc='Evaluating with MixFormer'):
        anonymized_data = anonymize_batch(batch)

        for item in anonymized_data:
            anonymized_skel = item['skeleton']
            gt_skel = item['gt_skeleton']
            act_label = item['action']
            ret_actor = item['retargeted_actor']
            orig_actor = item['original_actor']

            t_out = anonymized_skel.shape[0]
            t_in = gt_skel.shape[0]
            Tm = min(t_out, t_in)
            mse_ = F.mse_loss(anonymized_skel[:Tm], gt_skel[:Tm])
            mse_sum += mse_.item()
            mse_count += 1

            skel_np = anonymized_skel.detach().cpu().numpy()
            prepped = mixformer_preprocess_single_skeleton(skel_np)

            zeros_ = np.zeros_like(prepped)
            prepped_2p = np.concatenate([prepped, zeros_], axis=3)
            ar_input = torch.tensor(prepped_2p, dtype=torch.float32).unsqueeze(0).to(args.device)

            with torch.no_grad():
                ar_out = ar_model(ar_input)
                _, ar_pred = torch.max(ar_out, 1)
            correct_action += (ar_pred.item() == act_label)

            with torch.no_grad():
                ri_out = ri_model(ar_input)
                _, ri_pred = torch.max(ri_out, 1)
            if ri_pred.item() == ret_actor:
                correct_actor_ret += 1
            if ri_pred.item() == orig_actor:
                correct_actor_orig += 1

            total_samples += 1

    act_acc = 100.0 * correct_action / total_samples if total_samples else 0
    ret_acc = 100.0 * correct_actor_ret / total_samples if total_samples else 0
    orig_acc = 100.0 * correct_actor_orig / total_samples if total_samples else 0
    neither = total_samples - correct_actor_ret - correct_actor_orig
    neither_acc = 100.0 * neither / total_samples if total_samples else 0
    avg_mse = mse_sum / mse_count if mse_count else 0

    print(f'[MixFormer] Action Recognition Accuracy: {act_acc:.2f}%')
    print('[MixFormer] Re-identification Results:')
    print(f'  Retargeted Actor: {correct_actor_ret}/{total_samples} ({ret_acc:.2f}%)')
    print(f'  Original  Actor: {correct_actor_orig}/{total_samples} ({orig_acc:.2f}%)')
    print(f'  Neither:         {neither}/{total_samples} ({neither_acc:.2f}%)')
    print(f'[MixFormer] Average MSE: {avg_mse:.6f}')

###############################################################################
# The known working snippet logic for SGN (test, run_sgn_eval, etc.)
###############################################################################
from sklearn.metrics import f1_score, precision_score, recall_score

def accuracy_snippet(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)

def top_k_accuracy_snippet(output, target, k=3):
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

def test_snippet(test_loader, model, k=3):
    acces = AverageMeter()
    topk_acces = AverageMeter()
    model.eval()
    label_output = []
    pred_output = []

    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1))
            )
            output = output.mean(1)

        label_output.append(target.numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy_snippet(output.cpu(), target)
        acces.update(acc[0], inputs.size(0))

        topk_acc = top_k_accuracy_snippet(output.cpu(), target, k=k)
        topk_acces.update(topk_acc[0], inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    pred_output = np.concatenate(pred_output, axis=0)

    label_index = np.argmax(label_output, axis=1)
    pred_index = np.argmax(pred_output, axis=1)

    f1 = f1_score(label_index, pred_index, average='macro', zero_division=0)
    precision = precision_score(label_index, pred_index, average='macro', zero_division=0)
    recall = recall_score(label_index, pred_index, average='macro', zero_division=0)

    return acces.avg, f1, precision, recall, topk_acces.avg

def run_sgn_eval_snippet(train_x, train_y, test_x, test_y, val_x, val_y, dataset_name, model, k=3, batch_size=16):
    ntu_loaders = NTUDataLoaders(
        dataset=dataset_name, case=0, seg=20,
        train_X=train_x, train_Y=train_y,
        test_X=test_x,  test_Y=test_y,
        val_X=val_x,    val_Y=val_y,
        aug=0
    )
    test_loader = ntu_loaders.get_test_loader(batch_size, 0)
    return test_snippet(test_loader, model, k=k)

def anonymizer_to_sgn(t, max_frames=300):
    # Pad t to max_frames and add duplicate actor
    T = t.shape[0]
    if T < max_frames:
        pad_size = max_frames - T
        padding = torch.zeros(pad_size, t.shape[1])
        t = torch.cat([t, padding], dim=0)
    if t.shape[1] == 75:
        buffer = torch.zeros_like(t)
        t = torch.cat([t, buffer], dim=1)
    return t

###############################################################################
# The integrated SGN approach
###############################################################################
def eval_skeleton_sgn(paired_test, anonymizer_model, dataset_name, ar_model_weights, ri_model_weights, args):
    """Modified SGN evaluation function"""
    from model.sgn import SGN
    from data import get_num_classes
    
    # 1. Initialize models properly
    num_classes_ar = get_num_classes(dataset_name, 'ar')  # Action recognition classes
    num_classes_ri = get_num_classes(dataset_name, 'ri')  # Person ID classes
    
    # Create model with proper args structure
    model_args = type('', (), {})()  # Create empty object to mimic args
    model_args.num_classes = num_classes_ar
    model_args.dataset = dataset_name
    model_args.seg = 20
    
    sgn_ar = SGN(num_classes_ar, dataset_name, 20).cuda()
    sgn_ri = SGN(num_classes_ri, dataset_name, 20).cuda()
    
    # 2. Load weights properly
    ar_checkpoint = torch.load(ar_model_weights, map_location='cpu')
    if 'state_dict' in ar_checkpoint:
        sgn_ar.load_state_dict(ar_checkpoint['state_dict'])
    else:
        sgn_ar.load_state_dict(ar_checkpoint)
        
    ri_checkpoint = torch.load(ri_model_weights, map_location='cpu')
    if 'state_dict' in ri_checkpoint:
        sgn_ri.load_state_dict(ri_checkpoint['state_dict'])
    else:
        sgn_ri.load_state_dict(ri_checkpoint)
    
    sgn_ar.eval()
    sgn_ri.eval()
    
    # 3. Process data through anonymizer
    anonymized_data = []
    for batch in tqdm(paired_test, desc='Processing through anonymizer'):
        if args.model_type == 'raw':
            batch_data = get_anonymized_paired_raw(batch)
        elif args.model_type == 'transformer':
            batch_data = get_anonymized_paired_transformer(batch, anonymizer_model, prep_data)
        else:
            batch_data = get_anonymized_paired_dmr_pmr(batch, anonymizer_model, T=args.T)
        anonymized_data.extend(batch_data)
    
    # 4. Prepare data and evaluate
    test_size = len(anonymized_data)
    ar_correct = 0
    ri_ret_correct = 0
    ri_orig_correct = 0
    mse_values = []
    
    for item in tqdm(anonymized_data, desc="Evaluating SGN"):
        skel = item['skeleton']
        gt_skel = item['gt_skeleton']
        action_label = item['action'] 
        ret_actor = item['retargeted_actor']
        orig_actor = item['original_actor']
        
        # Calculate MSE between skeletons
        t_out = skel.shape[0]
        t_in = gt_skel.shape[0]
        Tm = min(t_out, t_in)
        mse_ = F.mse_loss(skel[:Tm], gt_skel[:Tm])
        mse_values.append(mse_.item())
        
        # Format for SGN - crucial step!
        try:
            # Convert to numpy for preprocessing
            skel_np = skel.cpu().numpy()
            
            # This is the key - properly segment and format the skeleton
            processed = sgn_preprocess_single_skeleton(skel_np, seg=20)
            
            # Now create proper tensor with exactly 3 dimensions [batch, seg, features]
            input_tensor = torch.from_numpy(processed).float().to(args.device)
            
            # Verify shape matches SGN expectations
            if len(input_tensor.shape) != 3 or input_tensor.shape[1] != 20:
                input_tensor = input_tensor.reshape(1, 20, -1)
            
            # Perform predictions
            with torch.no_grad():
                ar_output = sgn_ar(input_tensor)
                ar_pred = ar_output.argmax(dim=1).item()
                
                ri_output = sgn_ri(input_tensor) 
                ri_pred = ri_output.argmax(dim=1).item()
            
            # Record results
            if ar_pred == action_label:
                ar_correct += 1
                
            if ri_pred == ret_actor:
                ri_ret_correct += 1
            elif ri_pred == orig_actor:
                ri_orig_correct += 1
                
        except Exception as e:
            print(f"Error processing skeleton: {e}")
            continue
    
    # 5. Calculate and report metrics
    ar_acc = 100 * ar_correct / test_size
    ri_ret_acc = 100 * ri_ret_correct / test_size
    ri_orig_acc = 100 * ri_orig_correct / test_size
    ri_neither = test_size - ri_ret_correct - ri_orig_correct
    ri_neither_acc = 100 * ri_neither / test_size
    avg_mse = sum(mse_values) / len(mse_values) if mse_values else 0
    
    # Calculate more comprehensive metrics
    print(f"[SGN] Action Recognition Results:")
    print(f"  Accuracy: {ar_acc:.2f}%")
    
    print(f"[SGN] Re-identification => ret_actor as label:")
    print(f"  Accuracy: {ri_ret_acc:.2f}%")
    print(f"  Retargeted Actor: {ri_ret_correct}/{test_size} ({ri_ret_acc:.2f}%)")
    print(f"  Original  Actor: {ri_orig_correct}/{test_size} ({ri_orig_acc:.2f}%)")
    print(f"  Neither:         {ri_neither}/{test_size} ({ri_neither_acc:.2f}%)")
    print(f"[SGN] Average MSE: {avg_mse:.6f}")

###############################################################################
# Main script entry point
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate anonymized skeleton data with AR/RI models.")
    parser.add_argument('--dataset', default='ntu120', choices=['ntu120','ntu','etri'])
    parser.add_argument('--setting', default='cs', choices=['cs','cv'])
    parser.add_argument('--model_type', default='transformer', choices=['raw','transformer','pmr','dmr'],
                        help="raw => evaluate the unmodified skeleton data. Others => anonymizers.")
    parser.add_argument('--transformer_model_path', default='model.pth', type=str)
    parser.add_argument('--ar_model_weights', default='', type=str)
    parser.add_argument('--ri_model_weights', default='', type=str)
    parser.add_argument('--eval_model', default='mixformer', choices=['mixformer','sgn'])

    # Data / sampling
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--paired_batch_size', default=8, type=int)
    parser.add_argument('--test_samples', default=5000, type=int)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--save_cache', action='store_true')
    parser.add_argument('--T', default=64, type=int,
                        help='Transformer=64 frames, PMR/DMR=75, etc.')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Is CUDA Available?")
    print(torch.cuda.is_available())
    print("Arguments:", args)

    # Load data
    X = load_data(args.dataset)

    # Possibly load from cache
    paired_path = f'data/{args.dataset}_{args.setting}_paired.pkl'
    if args.use_cache and os.path.exists(paired_path):
        with open(paired_path,'rb') as f:
            paired_data = pickle.load(f)
            paired_train = paired_data['train']
            paired_test  = paired_data['test']
        print("Loaded paired data from:", paired_path)
    else:
        paired_train, paired_test = get_cross_data(
            X, args.dataset, args.setting,
            batch_size=args.paired_batch_size,
            return_loader=True,
            train_samples=args.batch_size,
            test_samples=args.test_samples
        )
        if args.save_cache:
            with open(paired_path, 'wb') as f:
                pickle.dump({'train':paired_train,'test':paired_test}, f)
            print("Saved paired data to:", paired_path)

    ds = datasets[args.dataset]
    anonymizer_model = None
    if args.model_type != 'raw':
        if args.model_type == 'transformer':
            anonymizer_path = args.transformer_model_path
        else:
            anonymizer_path = f'./eval/{args.model_type}/{args.dataset}.pt'
        anonymizer_model = load_anonymizer(
            args.model_type, anonymizer_path, args.device, args, ds=ds
        )

    if args.eval_model == 'mixformer':
        print("[MAIN] Evaluate with MixFormer AR/RI...")
        eval_skeleton_mixformer(
            paired_test,
            anonymizer_model,
            dataset_name=args.dataset,
            ar_model_weights=args.ar_model_weights,
            ri_model_weights=args.ar_model_weights,
            args=args
        )
    else:
        print("[MAIN] Evaluate with SGN-based approach (snippet logic)...")
        eval_skeleton_sgn(
            paired_test,
            anonymizer_model,
            dataset_name=args.dataset,
            ar_model_weights=args.ar_model_weights,
            ri_model_weights=args.ri_model_weights,
            args=args
        )

if __name__ == '__main__':
    main()
