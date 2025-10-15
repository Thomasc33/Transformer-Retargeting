#!/usr/bin/env python3
"""
Anonymization Evaluation Script - Version 2
Evaluates raw data and anonymization models (TMR, DMR, PMR) with proper privacy metrics.
"""

import argparse
import torch
import numpy as np
import os
import sys
import json
from pathlib import Path
import time

# Force unbuffered output
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# Import preprocessing
from eval.preprocess import mixformer_preprocess_single_skeleton, sgn_preprocess_single_skeleton


def load_state_dict_with_module_fix(model_path, device='cuda'):
    """Load state dict and handle 'module.' prefix from DataParallel/DistributedDataParallel."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict if checkpoint is a dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict


def load_evaluation_models(eval_model_type, dataset, setting, device='cuda'):
    """
    Load AR and RI evaluation models ONCE.
    
    Returns:
        tuple: (ar_model, ri_model)
    """
    print(f"\n🤖 Loading {eval_model_type.upper()} evaluation models...")
    
    # Determine setting suffix
    if setting == 'cv':
        setting_suffix = 'cview'
    elif setting == 'cs':
        setting_suffix = 'csub'
    else:
        setting_suffix = setting
    
    if eval_model_type == 'mixformer':
        from src.model.ske_mixf import Model as MixformerModel
        
        # AR model
        ar_model_path = f'data/models_output/output/{dataset}_mixformer_ar_{setting_suffix}/NTU_mixformer_ar_{setting_suffix}/model_best.pth.tar'
        if os.path.exists(ar_model_path):
            ar_model = MixformerModel(
                num_class=60 if dataset == 'ntu' else 120,
                num_point=25,
                num_person=2,
                graph='src.graph.ntu_rgb_d.Graph',
                graph_args={'labeling_mode': 'spatial'},
                in_channels=3
            ).to(device)
            
            state_dict = load_state_dict_with_module_fix(ar_model_path, device)
            ar_model.load_state_dict(state_dict, strict=False)
            ar_model.eval()
            print(f"✓ Loaded AR model from {ar_model_path}")
        else:
            print(f"❌ AR model not found: {ar_model_path}")
            return None, None
        
        # RI model
        ri_model_path = f'data/models_output/output/{dataset}_mixformer_ri_{setting_suffix}/NTU_mixformer_ri_{setting_suffix}/model_best.pth.tar'
        if os.path.exists(ri_model_path):
            num_actors = 40 if dataset == 'ntu' else 106
            
            ri_model = MixformerModel(
                num_class=num_actors,
                num_point=25,
                num_person=2,
                graph='src.graph.ntu_rgb_d.Graph',
                graph_args={'labeling_mode': 'spatial'},
                in_channels=3
            ).to(device)
            
            state_dict = load_state_dict_with_module_fix(ri_model_path, device)
            ri_model.load_state_dict(state_dict, strict=False)
            ri_model.eval()
            print(f"✓ Loaded RI model from {ri_model_path}")
        else:
            print(f"❌ RI model not found: {ri_model_path}")
            return None, None
            
    else:  # sgn
        from src.model.sgn import SGN
        
        # AR model
        ar_model_path = f'output/{dataset}_ar_{setting_suffix}/model_best.pth.tar'
        if os.path.exists(ar_model_path):
            ar_model = SGN(
                num_classes=60 if dataset == 'ntu' else 120,
                dataset=dataset,
                seg=64
            ).to(device)
            
            state_dict = load_state_dict_with_module_fix(ar_model_path, device)
            ar_model.load_state_dict(state_dict, strict=False)
            ar_model.eval()
            print(f"✓ Loaded AR model from {ar_model_path}")
        else:
            print(f"❌ AR model not found: {ar_model_path}")
            return None, None

        # RI model
        ri_model_path = f'output/{dataset}_ri_{setting_suffix}/model_best.pth.tar'
        if os.path.exists(ri_model_path):
            num_actors = 40 if dataset == 'ntu' else 106

            ri_model = SGN(
                num_classes=num_actors,
                dataset=dataset,
                seg=64
            ).to(device)

            state_dict = load_state_dict_with_module_fix(ri_model_path, device)
            ri_model.load_state_dict(state_dict, strict=False)
            ri_model.eval()
            print(f"✓ Loaded RI model from {ri_model_path}")
        else:
            print(f"❌ RI model not found: {ri_model_path}")
            return None, None
    
    return ar_model, ri_model


def evaluate_skeletons(skeletons, labels_ar, labels_ri_target, labels_ri_original,
                       ar_model, ri_model, eval_model_type, device='cuda'):
    """
    Evaluate a list of skeletons with AR and RI models.
    
    Args:
        skeletons: List of skeleton tensors (T, 75)
        labels_ar: List of action labels
        labels_ri_target: List of target actor labels
        labels_ri_original: List of original actor labels
        ar_model: Action recognition model
        ri_model: Re-identification model
        eval_model_type: 'mixformer' or 'sgn'
        device: Device to run on
    
    Returns:
        dict: Results with AR and RI metrics
    """
    ar_correct = 0
    ri_target_correct = 0
    ri_original_correct = 0
    total = len(skeletons)
    
    for i, skeleton in enumerate(skeletons):
        try:
            # Preprocess
            if eval_model_type == 'mixformer':
                processed = mixformer_preprocess_single_skeleton(skeleton.numpy())
            else:  # sgn
                processed = sgn_preprocess_single_skeleton(skeleton.numpy())
            
            input_tensor = torch.from_numpy(processed).float().unsqueeze(0).to(device)
            
            # AR evaluation
            if ar_model is not None and labels_ar is not None:
                output = ar_model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                prediction = prediction + 1  # Fix for label indexing
                
                if prediction == labels_ar[i]:
                    ar_correct += 1
            
            # RI evaluation
            if ri_model is not None and labels_ri_target is not None:
                output = ri_model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                prediction = prediction + 1  # Fix for label indexing
                
                if prediction == labels_ri_target[i]:
                    ri_target_correct += 1
                if labels_ri_original is not None and prediction == labels_ri_original[i]:
                    ri_original_correct += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{total} samples...")
                
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue
    
    # Calculate metrics
    ar_accuracy = 100 * ar_correct / total if total > 0 else 0
    ri_target_accuracy = 100 * ri_target_correct / total if total > 0 else 0
    ri_original_accuracy = 100 * ri_original_correct / total if total > 0 else 0
    privacy_score = 100 - ri_original_accuracy
    
    return {
        'ar_accuracy': ar_accuracy,
        'ri_target_accuracy': ri_target_accuracy,
        'ri_original_accuracy': ri_original_accuracy,
        'privacy_score': privacy_score,
        'total_samples': total,
        'ar_correct': ar_correct,
        'ri_target_correct': ri_target_correct,
        'ri_original_correct': ri_original_correct
    }


def load_tmr_model(model_path, device='cuda'):
    """Load TMR (Transformer Motion Retargeting) model."""
    print(f"\n🤖 Loading TMR model from {model_path}...")
    print(f"  Step 1: Importing Model class...")

    from src.model.autoencoder import Model

    print(f"  Step 2: Creating model instance...")

    # Create model
    model = Model(
        num_class=60,
        num_point=25,
        num_person=1,
        graph='src.graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
        debug=False,
        dataset='ntu',
        device=device
    )

    print(f"  Step 3: Moving model to {device}...")
    model = model.to(device)

    print(f"  Step 4: Loading weights...")
    # Load weights with module prefix fix
    state_dict = load_state_dict_with_module_fix(model_path, device)

    print(f"  Step 5: Loading state dict into model...")
    model.load_state_dict(state_dict, strict=False)

    print(f"  Step 6: Setting model to eval mode...")
    model.eval()

    print(f"✓ TMR model loaded successfully")
    return model


def apply_tmr_retargeting(tmr_model, source_skeleton, target_skeleton, device='cuda', debug=False):
    """
    Apply TMR retargeting to convert source motion to target person.

    Args:
        tmr_model: TMR model
        source_skeleton: Source motion (T, 75) - person doing action
        target_skeleton: Target skeleton (T, 75) - person to retarget to
        device: Device to run on
        debug: Print debug information

    Returns:
        Retargeted skeleton (T, 75) - source action performed by target person
    """
    with torch.no_grad():
        # TMR expects (N, C, T, V, M) format
        # Convert from (T, 75) to (1, 3, T, 25, 1)
        T = source_skeleton.shape[0]

        if debug:
            print(f"\n[DEBUG] TMR Retargeting:")
            print(f"  Source shape: {source_skeleton.shape}")
            print(f"  Target shape: {target_skeleton.shape}")
            print(f"  Source range: [{source_skeleton.min():.3f}, {source_skeleton.max():.3f}]")
            print(f"  Target range: [{target_skeleton.min():.3f}, {target_skeleton.max():.3f}]")

        # Reshape: (T, 75) -> (T, 25, 3) -> (3, T, 25) -> (1, 3, T, 25, 1)
        source = source_skeleton.view(T, 25, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        target = target_skeleton.view(T, 25, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)

        if debug:
            print(f"  Source tensor shape: {source.shape}")
            print(f"  Target tensor shape: {target.shape}")

        source = source.to(device)
        target = target.to(device)

        # Run TMR model (outputs T-1 frames)
        output = tmr_model(source, target, target_motion=None, teacher_forcing_ratio=0.0)

        if debug:
            print(f"  TMR output shape: {output.shape}")
            print(f"  TMR output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  TMR output has NaN: {torch.isnan(output).any()}")
            print(f"  TMR output has Inf: {torch.isinf(output).any()}")

        # Convert back: (1, 3, T-1, 25, 1) -> (T-1, 75)
        output = output.squeeze(-1).squeeze(0).permute(1, 2, 0).contiguous().view(T-1, 75)

        # Repeat last frame to get back to T frames (T-1 -> T)
        last_frame = output[-1:, :]  # (1, 75)
        output = torch.cat([output, last_frame], dim=0)  # (T, 75)

        if debug:
            print(f"  Final output shape: {output.shape}")
            print(f"  Final output range: [{output.min():.3f}, {output.max():.3f}]")

        return output.cpu()


def evaluate_raw_data(test_dataset, ar_model, ri_model, eval_model_type, num_samples, device='cuda'):
    """
    Evaluate raw data (baseline).

    Uses y2 from Cross_Data: P2 doing A1 (ground truth retargeting)
    For raw data, target and original are the same (no anonymization)
    """
    print(f"\n🔍 Evaluating RAW DATA (baseline)...")

    skeletons = []
    labels_ar = []
    labels_ri = []

    num_samples = min(num_samples, len(test_dataset))

    for i in range(num_samples):
        x1, x2, y1, y2, actors, actions = test_dataset[i]

        # Use y2: P2 doing A1
        skeletons.append(y2)
        labels_ar.append(int(actions[0]))  # Action A1
        labels_ri.append(int(actors[1]))  # Actor P2

    # For raw data, target and original are the same (no anonymization yet)
    results = evaluate_skeletons(
        skeletons, labels_ar, labels_ri, labels_ri,  # target = original for raw
        ar_model, ri_model, eval_model_type, device
    )

    print(f"\n✓ RAW DATA Results:")
    print(f"  AR Accuracy: {results['ar_accuracy']:.1f}% (baseline utility)")
    print(f"  RI Accuracy: {results['ri_target_accuracy']:.1f}% (baseline)")
    print(f"  Total samples: {results['total_samples']}")

    return results


def evaluate_tmr(test_dataset, tmr_model, ar_model, ri_model, eval_model_type, num_samples, device='cuda'):
    """
    Evaluate TMR model by generating retargeted skeletons.

    For each sample:
    - x1 = P1 doing A1 (source)
    - x2 = P2 doing A2 (target skeleton)
    - Generate: TMR(x1, x2) = A1 retargeted to P2
    - AR should recognize A1
    - RI should predict P2 (target), not P1 (original)
    """
    print(f"\n🔍 Evaluating TMR MODEL...")
    print(f"  Generating retargeted skeletons with TMR...")

    skeletons = []
    labels_ar = []
    labels_ri_target = []
    labels_ri_original = []

    num_samples = min(num_samples, len(test_dataset))

    # Debug first sample
    debug_first = True

    for i in range(num_samples):
        x1, x2, y1, y2, actors, actions = test_dataset[i]

        # Generate retargeted skeleton: TMR(x1, x2) = A1 retargeted to P2
        retargeted = apply_tmr_retargeting(tmr_model, x1, x2, device, debug=debug_first)

        if debug_first:
            print(f"\n[DEBUG] First sample labels:")
            print(f"  Action A1: {actions[0]} (should be recognized)")
            print(f"  Actor P1 (original): {actors[0]} (should NOT be predicted)")
            print(f"  Actor P2 (target): {actors[1]} (should be predicted)")
            debug_first = False

        skeletons.append(retargeted)
        labels_ar.append(int(actions[0]))  # Action A1 (from P1)
        labels_ri_target.append(int(actors[1]))  # Target actor P2 (GOOD if predicted)
        labels_ri_original.append(int(actors[0]))  # Original actor P1 (BAD if predicted)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_samples} retargeted skeletons...")

    print(f"  Evaluating {len(skeletons)} retargeted skeletons...")

    # Evaluate retargeted skeletons
    results = evaluate_skeletons(
        skeletons, labels_ar, labels_ri_target, labels_ri_original,
        ar_model, ri_model, eval_model_type, device
    )

    print(f"\n✓ TMR Results:")
    print(f"  AR Accuracy: {results['ar_accuracy']:.1f}% (utility preservation)")
    print(f"  RI Target Accuracy: {results['ri_target_accuracy']:.1f}% (predicts target - GOOD)")
    print(f"  RI Original Accuracy: {results['ri_original_accuracy']:.1f}% (predicts original - BAD)")
    print(f"  Privacy Score: {results['privacy_score']:.1f}% (higher is better)")
    print(f"  Total samples: {results['total_samples']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate anonymization models')
    parser.add_argument('--eval-model', type=str, default='mixformer', choices=['mixformer', 'sgn'])
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'])
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'])
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='test_results/anonymization_v2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tmr-model-path', type=str, default='data/models_output/model.pth')
    parser.add_argument('--skip-raw', action='store_true', help='Skip raw data evaluation')
    parser.add_argument('--only-tmr', action='store_true', help='Only evaluate TMR')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"ANONYMIZATION EVALUATION V2")
    print(f"{'='*80}")
    print(f"Eval Model: {args.eval_model}")
    print(f"Dataset: {args.dataset} ({args.setting})")
    print(f"Test Samples: {args.test_samples}")
    print(f"Device: {args.device}")
    print(f"TMR Model: {args.tmr_model_path}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Load test data
    print("📁 Loading test data...")
    data_file = f'data/{args.dataset}_{args.setting}_paired_10000_2000.pt'
    if not os.path.exists(data_file):
        data_file = f'data/{args.dataset}_{args.setting}_paired.pt'

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    data = torch.load(data_file, map_location='cpu')
    test_dataset = data['test']
    print(f"✓ Loaded {len(test_dataset)} test samples\n")

    # Load evaluation models ONCE
    ar_model, ri_model = load_evaluation_models(args.eval_model, args.dataset, args.setting, args.device)

    if ar_model is None or ri_model is None:
        print("❌ Failed to load evaluation models")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate raw data (baseline)
    if not args.skip_raw and not args.only_tmr:
        raw_results = evaluate_raw_data(test_dataset, ar_model, ri_model, args.eval_model, args.test_samples, args.device)

        output_file = os.path.join(args.output_dir, 'raw_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'model_type': 'raw',
                'eval_model': args.eval_model,
                'dataset': args.dataset,
                'setting': args.setting,
                'test_samples': args.test_samples,
                'results': raw_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

        print(f"\n✅ Raw results saved to {output_file}")

    # Evaluate TMR
    if not os.path.exists(args.tmr_model_path):
        print(f"\n⚠️  TMR model not found: {args.tmr_model_path}")
        print(f"Skipping TMR evaluation")
    else:
        # Load TMR model
        tmr_model = load_tmr_model(args.tmr_model_path, args.device)

        # Evaluate TMR
        tmr_results = evaluate_tmr(test_dataset, tmr_model, ar_model, ri_model, args.eval_model, args.test_samples, args.device)

        # Save TMR results
        output_file = os.path.join(args.output_dir, 'tmr_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'model_type': 'tmr',
                'eval_model': args.eval_model,
                'dataset': args.dataset,
                'setting': args.setting,
                'test_samples': args.test_samples,
                'results': tmr_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

        print(f"\n✅ TMR results saved to {output_file}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")

    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

