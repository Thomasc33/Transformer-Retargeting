#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Raw vs MLM Evaluation System

This script provides comprehensive evaluation of both raw skeleton data and MLM-processed data
through SGN and Mixformer models for Action Recognition (AR) and Re-identification (RI) tasks.

The system addresses the critical need to compare performance between:
1. Raw skeleton data -> SGN/Mixformer -> AR/RI performance
2. MLM processed data -> SGN/Mixformer -> AR/RI performance

This evaluation was requested by the advisor to understand if MLM preprocessing
improves or degrades downstream task performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrain import SkeletonAutoEncoder
from src.data import load_data, get_cross_data
# Import from the correct model locations
sys.path.insert(0, os.path.join(project_root, 'src', 'model'))
from src.model.sgn import SGN as SGNModel
from src.model.ske_mixf import Model as MixformerModel
from scripts.model_weight_manager import ModelWeightManager

# Import evaluation utilities - define locally to avoid import issues
def import_class(import_str):
    """Dynamically import a class from a string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found')

def safe_load_model(model_path, device='cpu'):
    """Safely load model weights with proper device mapping."""
    print(f"Loading model from {model_path} (device: {device})")
    try:
        # Check if the model path contains .tar extension
        is_tar_file = '.tar' in model_path.lower()

        # Load the model with proper device mapping
        checkpoint = torch.load(model_path, map_location=device)

        # If it's a .tar file, extract state_dict
        if is_tar_file:
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            # For .pt or .pth files
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

        # Clean up state dict keys
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        return state_dict
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

def get_anonymized_paired_raw(batch, gender_map=None):
    """Process raw (unanonymized) skeleton data - exact copy from eval_model.py."""
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    out = []

    for i in range(N):
        # Ensure actions are 0-indexed for models but 1-indexed in data
        action1 = int(actions[i, 0]) - 1
        action2 = int(actions[i, 1]) - 1
        actor1 = int(actors[i, 0]) - 1
        actor2 = int(actors[i, 1]) - 1

        # Add gender information (1-indexed in CSV, so add 1 to actor ID)
        gender1 = None
        gender2 = None
        if gender_map:
            # Convert 0-indexed actor ID to 1-indexed for gender map lookup
            gender1 = 1 if gender_map.get(actor1 + 1, 'M') == 'M' else 0
            gender2 = 1 if gender_map.get(actor2 + 1, 'M') == 'M' else 0

        # Output four combinations of data
        item1 = {
            'skeleton': x1[i].cpu().to(torch.float32),  # p1 a1
            'gt_skeleton': y2[i].cpu().to(torch.float32),  # p2 a1
            'reference_skeleton': x1[i].cpu().to(torch.float32),  # p1 a1 (reference = skeleton for raw)
            'retargeted_actor': actor2,  # p2
            'original_actor': actor1,  # p1
            'action': action1  # a1
        }
        if gender1 is not None:
            item1['gender'] = gender1
        out.append(item1)

        item2 = {
            'skeleton': x2[i].cpu().to(torch.float32),  # p2 a2
            'gt_skeleton': y1[i].cpu().to(torch.float32),  # p1 a2
            'reference_skeleton': x2[i].cpu().to(torch.float32),  # p2 a2 (reference = skeleton for raw)
            'retargeted_actor': actor1,  # p1
            'original_actor': actor2,  # p2
            'action': action2  # a2
        }
        if gender2 is not None:
            item2['gender'] = gender2
        out.append(item2)

        item3 = {
            'skeleton': y1[i].cpu().to(torch.float32),  # p1 a2
            'gt_skeleton': x2[i].cpu().to(torch.float32),  # p2 a2
            'reference_skeleton': y1[i].cpu().to(torch.float32),  # p1 a2 (reference = skeleton for raw)
            'retargeted_actor': actor2,  # p2
            'original_actor': actor1,  # p1
            'action': action2
        }
        if gender1 is not None:
            item3['gender'] = gender1
        out.append(item3)

        item4 = {
            'skeleton': y2[i].cpu().to(torch.float32),  # p2 a1
            'gt_skeleton': x1[i].cpu().to(torch.float32),  # p1 a1
            'reference_skeleton': y2[i].cpu().to(torch.float32),  # p2 a1 (reference = skeleton for raw)
            'retargeted_actor': actor1,  # p1
            'original_actor': actor2,  # p2
            'action': action1
        }
        if gender2 is not None:
            item4['gender'] = gender2
        out.append(item4)
    return out

def test_snippet(test_loader, model, k=3):
    """Test model using snippet-based evaluation like eval_model.py."""
    model.eval()

    total_samples = 0
    correct_predictions = 0
    top_k_correct = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.cuda() if torch.cuda.is_available() else data

            # Forward pass
            outputs = model(data)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Convert one-hot labels to class indices
            if len(labels.shape) > 1:
                label_indices = torch.argmax(labels, dim=1)
            else:
                label_indices = labels

            # Calculate accuracy
            correct_predictions += (predicted == label_indices).sum().item()

            # Calculate top-k accuracy
            _, top_k_pred = torch.topk(outputs, k, dim=1)
            for i in range(label_indices.size(0)):
                if label_indices[i] in top_k_pred[i]:
                    top_k_correct += 1

            total_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    top_k_accuracy = (top_k_correct / total_samples) * 100 if total_samples > 0 else 0

    # Calculate F1, precision, recall (simplified)
    from sklearn.metrics import f1_score, precision_score, recall_score
    try:
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
    except:
        f1 = precision = recall = 0.0

    return accuracy, f1, precision, recall, top_k_accuracy

def test_snippet_sgn_5crop(test_loader, model, k=3):
    """Test SGN model using 5-crop evaluation with averaging."""
    model.eval()

    total_samples = 0
    correct_predictions = 0
    top_k_correct = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        # Group data by original samples (every 5 consecutive items are crops of the same sample)
        batch_data = []
        batch_labels = []

        for batch_idx, (data, labels) in enumerate(test_loader):
            batch_data.append(data)
            batch_labels.append(labels)

        # Concatenate all batches
        all_data = torch.cat(batch_data, dim=0)
        all_labels_tensor = torch.cat(batch_labels, dim=0)

        # Process in groups of 5 (5 crops per sample)
        num_samples = len(all_data) // 5

        for i in range(num_samples):
            start_idx = i * 5
            end_idx = start_idx + 5

            # Get 5 crops for this sample
            crops = all_data[start_idx:end_idx]  # (5, 20, 75)
            sample_labels = all_labels_tensor[start_idx:end_idx]  # (5, num_classes)

            # Get the true label (should be same for all 5 crops)
            if len(sample_labels.shape) > 1:
                true_label = torch.argmax(sample_labels[0], dim=0)
            else:
                true_label = sample_labels[0]

            crops = crops.cuda() if torch.cuda.is_available() else crops

            # Forward pass for all 5 crops
            crop_outputs = []
            for crop in crops:
                output = model(crop.unsqueeze(0))  # Add batch dimension
                crop_outputs.append(output)

            # Average the outputs across crops
            avg_output = torch.mean(torch.stack(crop_outputs), dim=0)

            # Get prediction
            _, predicted = torch.max(avg_output, 1)

            # Calculate accuracy
            if predicted.item() == true_label.item():
                correct_predictions += 1

            # Calculate top-k accuracy
            _, top_k_pred = torch.topk(avg_output, k, dim=1)
            if true_label.item() in top_k_pred[0]:
                top_k_correct += 1

            total_samples += 1

            all_predictions.append(predicted.item())
            all_labels.append(true_label.item())

    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    top_k_accuracy = (top_k_correct / total_samples) * 100 if total_samples > 0 else 0

    # Calculate F1, precision, recall (simplified)
    from sklearn.metrics import f1_score, precision_score, recall_score
    try:
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
    except:
        f1 = precision = recall = 0.0

    return accuracy, f1, precision, recall, top_k_accuracy

# Import preprocessing functions
original_sys_path = sys.path.copy()
PREPROCESSING_AVAILABLE = False

try:
    # Add the evaluation legacy path to import the correct preprocessing functions
    eval_legacy_path = os.path.join(project_root, 'eval', 'suite', 'experiments', 'eval_legacy')
    if eval_legacy_path not in sys.path:
        sys.path.insert(0, eval_legacy_path)

    # Import with explicit module name to avoid conflicts
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_preprocess",
                                                  os.path.join(eval_legacy_path, "preprocess.py"))
    eval_preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_preprocess)

    sgn_preprocess_single_skeleton = eval_preprocess.sgn_preprocess_single_skeleton
    mixformer_preprocess_single_skeleton = eval_preprocess.mixformer_preprocess_single_skeleton

    # Test the functions to make sure they work
    test_skeleton = np.random.rand(10, 75).astype(np.float32)
    test_result = sgn_preprocess_single_skeleton(test_skeleton, seg=20, dataset='NTU')

    print("✓ Successfully imported and tested preprocessing functions")
    PREPROCESSING_AVAILABLE = True

except Exception as e:
    print(f"Warning: Could not import preprocessing functions: {e}")
    print("⚠️  Using fallback preprocessing - results will be significantly less accurate!")
    PREPROCESSING_AVAILABLE = False

    # Define fallback functions (these are simplified and may not give accurate results)
    def sgn_preprocess_single_skeleton(skeleton_array, seg=20, dataset='NTU'):
        """Fallback SGN preprocessing - SIMPLIFIED VERSION."""
        # Simple fallback - just reshape and downsample
        if len(skeleton_array.shape) == 2 and skeleton_array.shape[1] == 75:
            # Downsample to seg frames
            if skeleton_array.shape[0] > seg:
                indices = np.linspace(0, skeleton_array.shape[0] - 1, seg).astype(int)
                skeleton_array = skeleton_array[indices]
            elif skeleton_array.shape[0] < seg:
                # Repeat last frame
                last_frame = skeleton_array[-1:]
                num_repeats = seg - skeleton_array.shape[0]
                skeleton_array = np.vstack([skeleton_array] + [last_frame] * num_repeats)

            # Return 5 crops (same data repeated)
            crops = np.stack([skeleton_array] * 5, axis=0)
            return crops.astype(np.float32)
        return skeleton_array

    def mixformer_preprocess_single_skeleton(skeleton_array, **kwargs):
        """Fallback Mixformer preprocessing - SIMPLIFIED VERSION."""
        if len(skeleton_array.shape) == 2 and skeleton_array.shape[1] == 75:
            T = skeleton_array.shape[0]
            data_numpy = skeleton_array.reshape(T, 25, 3).transpose(2, 0, 1)
            data_numpy = data_numpy[..., np.newaxis]  # (3, T, 25, 1)
            return data_numpy
        return skeleton_array
finally:
    sys.path = original_sys_path


class MLMDataProcessor:
    """Process data through MLM autoencoder for comparison."""
    
    def __init__(self, mlm_model_dir, dataset, seq_len, device):
        self.mlm_model_dir = mlm_model_dir
        self.dataset = dataset
        self.seq_len = seq_len
        self.device = device
        self.mlm_model = None
        self._load_mlm_model()
    
    def _load_mlm_model(self):
        """Load the MLM autoencoder model."""
        try:
            self.mlm_model = SkeletonAutoEncoder(dataset=self.dataset, seq_len=self.seq_len).to(self.device)
            
            # Load model components
            encoder_path = os.path.join(self.mlm_model_dir, 'encoder_best.pth')
            decoder_path = os.path.join(self.mlm_model_dir, 'decoder_best.pth')
            output_layer_path = os.path.join(self.mlm_model_dir, 'output_layer_best.pth')
            
            if all(os.path.exists(p) for p in [encoder_path, decoder_path, output_layer_path]):
                self.mlm_model.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
                self.mlm_model.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
                self.mlm_model.output_layer.load_state_dict(torch.load(output_layer_path, map_location=self.device))
                self.mlm_model.eval()
                print(f"✓ MLM model loaded from {self.mlm_model_dir}")
            else:
                raise FileNotFoundError("Missing MLM model components")
                
        except Exception as e:
            print(f"✗ Failed to load MLM model: {e}")
            self.mlm_model = None
    
    def process_data(self, data_loader):
        """Process data through MLM autoencoder."""
        if self.mlm_model is None:
            raise RuntimeError("MLM model not loaded")
        
        processed_data = []
        labels = []
        actors = []
        
        with torch.no_grad():
            for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Processing through MLM")):
                try:
                    if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                        x1, x2, y1, y2, batch_actors, batch_actions = batch_content
                        x_data = x1
                    else:
                        continue
                    
                    if not isinstance(x_data, torch.Tensor):
                        x_data = torch.tensor(x_data, dtype=torch.float32)
                    
                    # Handle data format - ensure proper shape for MLM
                    if len(x_data.shape) == 3 and x_data.shape[2] == 75:
                        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 25, 3)
                    elif len(x_data.shape) == 2 and x_data.shape[1] == 75:
                        x_data = x_data.reshape(1, x_data.shape[0], 25, 3)
                    
                    x_data = x_data.to(self.device)
                    
                    # Process through MLM
                    mlm_output = self.mlm_model(x_data)
                    
                    # Handle MLM output format
                    if len(mlm_output.shape) == 5:  # (batch, frames, actor, joints, channels)
                        mlm_output = mlm_output.squeeze(2)  # Remove actor dimension
                    
                    # Convert back to format expected by downstream models
                    if len(mlm_output.shape) == 4:  # (batch, frames, joints, channels)
                        mlm_output = mlm_output.reshape(mlm_output.shape[0], mlm_output.shape[1], -1)
                    
                    processed_data.append(mlm_output.cpu())
                    
                    # Store labels
                    if isinstance(batch_actions, torch.Tensor):
                        labels.extend(batch_actions.cpu().numpy())
                    else:
                        labels.extend(batch_actions)
                    
                    if isinstance(batch_actors, torch.Tensor):
                        actors.extend(batch_actors.cpu().numpy())
                    else:
                        actors.extend(batch_actors)
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if processed_data:
            return torch.cat(processed_data, dim=0), np.array(labels), np.array(actors)
        else:
            return None, None, None


class DownstreamModelEvaluator:
    """Evaluate SGN and Mixformer models on raw vs MLM data using eval_model.py patterns."""

    def __init__(self, dataset, setting, device):
        self.dataset = dataset
        self.setting = setting
        self.device = device
        self.models = {}
        self.weight_manager = ModelWeightManager(dataset, setting)

        # Import datasets configuration
        from src.data import datasets
        self.datasets = datasets
    
    def _setup_model_configs(self):
        """Setup model configurations for different datasets."""
        if self.dataset == 'ntu':
            self.model_configs = {
                'sgn': {
                    'ar': {
                        'num_class': 60,
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/sgn/pretrained/ntu/{self.setting}_ar.pth'
                    },
                    'ri': {
                        'num_class': 40,  # Number of subjects for re-identification
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/sgn/pretrained/ntu/{self.setting}_ri.pth'
                    }
                },
                'mixformer': {
                    'ar': {
                        'num_class': 60,  # NTU has 60 action classes
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/mixformer/old_pretrained/ntu/ar.pth'
                    },
                    'ri': {
                        'num_class': 40,  # NTU has 40 subjects for RI
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/mixformer/old_pretrained/ntu/ri.pth'
                    }
                }
            }
        elif self.dataset == 'etri':
            self.model_configs = {
                'sgn': {
                    'ar': {
                        'num_class': 55,
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/sgn/pretrained/etri/{self.setting}_ar_best.pth'
                    },
                    'ri': {
                        'num_class': 100,
                        'num_point': 25,
                        'num_person': 2,
                        'graph': 'src.graph.ntu_rgb_d.Graph',
                        'weights_path': f'eval/sgn/pretrained/etri/{self.setting}_ri_best.pth'
                    }
                }
            }
    
    def _load_sgn_model(self, task):
        """Load SGN model using eval_model.py approach."""
        model_config = self.weight_manager.get_model_config('sgn', task)

        if not model_config['available']:
            print(f"⚠️  SGN {task.upper()} model not available")
            return None

        weights_path = model_config['path']

        if 'note' in model_config:
            print(f"ℹ️  {model_config['note']}")

        try:
            # Use import_class like eval_model.py
            SGN_Model = import_class('src.model.sgn.SGN')

            # Determine number of classes based on task
            if task == 'ar':
                num_classes = self.datasets[self.dataset]['num_class']
            elif task == 'ri':
                num_classes = self.datasets[self.dataset]['num_actor']
            elif task == 'gc':
                num_classes = 2  # Gender classification
            else:
                raise ValueError(f"Unknown task: {task}")

            model = SGN_Model(
                num_classes=num_classes,
                dataset=self.dataset.upper(),
                seg=20  # SGN pretrained models use seg=20
            )

            model = model.to(self.device)

            # Load weights using our local safe_load_model
            state_dict = safe_load_model(weights_path, self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            print(f"✓ Loaded SGN {task.upper()} model")
            return model

        except Exception as e:
            print(f"✗ Failed to load SGN {task} model: {e}")
            return None

    def _load_mixformer_model(self, task):
        """Load Mixformer model using eval_model.py approach."""
        model_config = self.weight_manager.get_model_config('mixformer', task)

        if not model_config['available']:
            print(f"⚠️  Mixformer {task.upper()} model not available")
            return None

        # Check if this is actually an SGN model being used as alternative
        if model_config.get('type') == 'sgn':
            print(f"ℹ️  {model_config.get('note', 'Using SGN model as alternative')}")
            print(f"❌ Skipping mixformer {task} - model not available")
            return None

        weights_path = model_config['path']

        if 'note' in model_config:
            print(f"ℹ️  {model_config['note']}")

        try:
            # Use import_class like eval_model.py
            MixFormer_Model = import_class('src.model.ske_mixf.Model')

            # Determine number of classes based on task
            if task == 'ar':
                num_classes = self.datasets[self.dataset]['num_class']
            elif task == 'ri':
                num_classes = self.datasets[self.dataset]['num_actor']
            elif task == 'gc':
                num_classes = 2  # Gender classification
            else:
                raise ValueError(f"Unknown task: {task}")

            print(f"🔧 Creating Mixformer {task.upper()} model: num_class={num_classes}, dataset={self.dataset}")

            model = MixFormer_Model(
                num_class=num_classes,
                num_point=25,
                num_person=2,
                graph=self.datasets[self.dataset]['graph']
            )

            print(f"🔧 Created model FC shape: {model.fc.weight.shape}")

            model = model.to(self.device)

            print(f"🔧 After moving to device, model FC shape: {model.fc.weight.shape}")

            # Load weights using our local safe_load_model
            state_dict = safe_load_model(weights_path, self.device)

            # Handle architecture mismatch in fc layer
            model_fc_weight = None
            for name, param in model.named_parameters():
                if name == 'fc.weight':  # Exact match for final FC layer
                    model_fc_weight = param
                    break

            print(f"🔧 Found model FC weight shape: {model_fc_weight.shape if model_fc_weight is not None else 'None'}")

            if model_fc_weight is not None and 'fc.weight' in state_dict:
                checkpoint_fc_shape = state_dict['fc.weight'].shape
                model_fc_shape = model_fc_weight.shape

                print(f"🔧 Checkpoint FC shape: {checkpoint_fc_shape}")
                print(f"🔧 Model FC shape: {model_fc_shape}")

                if checkpoint_fc_shape != model_fc_shape:
                    print(f"⚠️  FC layer shape mismatch: checkpoint {checkpoint_fc_shape} vs model {model_fc_shape}")
                    print(f"    Removing fc.weight and fc.bias from checkpoint to allow random initialization")
                    # Remove fc layer weights to allow random initialization
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

            # Load state dict with strict=False to handle missing keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"    Missing keys (will use random initialization): {missing_keys}")
            if unexpected_keys:
                print(f"    Unexpected keys (ignored): {unexpected_keys}")

            model.eval()

            print(f"✓ Loaded Mixformer {task.upper()} model")
            return model

        except Exception as e:
            print(f"✗ Failed to load Mixformer {task} model: {e}")
            return None
    
    def evaluate_model(self, model, data_loader, task='ar', model_type='sgn'):
        """Evaluate a model on given data."""
        if model is None:
            return {'accuracy': 0.0, 'predictions': [], 'labels': []}

        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_content in enumerate(tqdm(data_loader, desc=f"Evaluating {task.upper()}")):
                try:
                    if isinstance(batch_content, (tuple, list)) and len(batch_content) >= 3:
                        if len(batch_content) == 6:
                            x1, x2, y1, y2, actors, actions = batch_content
                            x_data = x1
                            if task == 'ar':
                                labels = actions
                            else:  # ri
                                labels = actors
                        else:
                            x_data, labels = batch_content[0], batch_content[1]
                    else:
                        continue

                    # Handle label format - take first person's labels only
                    if isinstance(labels, torch.Tensor):
                        if len(labels.shape) > 1:
                            labels = labels[:, 0]  # Take first person's labels
                        labels = labels.cpu().numpy()
                    elif isinstance(labels, (list, tuple)):
                        labels = np.array(labels)
                        if len(labels.shape) > 1:
                            labels = labels[:, 0]  # Take first person's labels
                    
                    if not isinstance(x_data, torch.Tensor):
                        x_data = torch.tensor(x_data, dtype=torch.float32)
                    
                    # Handle different model input formats
                    if model_type == 'sgn':
                        # SGN expects (batch, frames, joints*channels)
                        # Pretrained models use seg=20, so we need to downsample to 20 frames
                        if len(x_data.shape) == 4:  # (batch, frames, joints, channels)
                            x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], -1)

                        # Downsample to 20 frames for SGN (pretrained models expect this)
                        if x_data.shape[1] > 20:
                            # Sample 20 frames evenly from the sequence
                            indices = torch.linspace(0, x_data.shape[1] - 1, 20).long()
                            x_data = x_data[:, indices, :]
                        elif x_data.shape[1] < 20:
                            # Repeat last frame to reach 20 frames
                            last_frame = x_data[:, -1:, :]
                            num_repeats = 20 - x_data.shape[1]
                            repeated_frames = last_frame.repeat(1, num_repeats, 1)
                            x_data = torch.cat([x_data, repeated_frames], dim=1)

                        x_data = x_data.to(self.device)
                        outputs = model(x_data)
                    else:  # mixformer
                        # Mixformer expects (N, C, T, V, M) with 64 frames
                        if len(x_data.shape) == 3:  # (batch, frames, features)
                            if x_data.shape[2] == 75:  # joints*channels format
                                x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 25, 3)
                            # Convert to (N, C, T, V, M)
                            x_data = x_data.permute(0, 3, 1, 2).unsqueeze(-1)  # Add person dimension

                        # Ensure 64 frames for Mixformer
                        if x_data.shape[2] == 75:  # If 75 frames, truncate to 64
                            x_data = x_data[:, :, :64, :, :]

                        x_data = x_data.to(self.device)
                        outputs = model(x_data)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels)
                        
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Calculate accuracy
        if all_predictions and all_labels:
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        else:
            accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def evaluate_sgn_model(self, model, data_loader, task='ar', is_mlm_data=False):
        """Evaluate SGN model using exact eval_model.py approach."""
        if model is None:
            return {'accuracy': 0.0, 'predictions': [], 'labels': []}

        model.eval()

        # Process data to create anonymized data structure like eval_model.py
        anonymized_data = []

        print(f"Processing data for SGN {task.upper()} evaluation...")

        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc=f"Processing for SGN {task.upper()}")):
            try:
                if is_mlm_data:
                    # MLM data comes as (data, labels, actors) - this is the output from MLM processing
                    x_data, labels, actors = batch_content

                    for i in range(x_data.shape[0]):
                        skeleton = x_data[i]

                        # Convert labels to scalars - ensure they are 0-indexed like eval_model.py
                        if isinstance(labels, torch.Tensor):
                            action_label = labels[i].item() if labels[i].numel() == 1 else labels[i, 0].item()
                        else:
                            action_label = labels[i]

                        # Ensure action is 0-indexed (subtract 1 if it's 1-indexed)
                        if action_label > 0:
                            action_label = action_label - 1

                        if isinstance(actors, torch.Tensor):
                            actor_label = actors[i].item() if actors[i].numel() == 1 else actors[i, 0].item()
                        else:
                            actor_label = actors[i]

                        # Ensure actor is 0-indexed (subtract 1 if it's 1-indexed)
                        if actor_label > 0:
                            actor_label = actor_label - 1

                        # Create item structure exactly like eval_model.py
                        item = {
                            'skeleton': skeleton,
                            'action': action_label,
                            'retargeted_actor': actor_label,
                            'original_actor': actor_label,  # For MLM data, use same actor
                            'gt_skeleton': skeleton,  # For MLM data, use same skeleton as GT
                            'reference_skeleton': skeleton  # Add reference skeleton for consistency
                        }
                        anonymized_data.append(item)
                else:
                    # Raw data comes as 6-item batch - use get_anonymized_paired_raw
                    if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                        batch_anonymized = get_anonymized_paired_raw(batch_content)
                        anonymized_data.extend(batch_anonymized)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        if not anonymized_data:
            print("No anonymized data processed!")
            return {'accuracy': 0.0, 'predictions': [], 'labels': []}

        # Now evaluate using the exact same approach as eval_model.py
        test_size = len(anonymized_data)
        ar_correct = 0
        ri_ret_correct = 0
        ri_orig_correct = 0

        all_ar_preds = []
        all_ar_labels = []
        all_ri_preds = []
        all_ri_labels = []

        # Process each sample exactly like eval_model.py
        for item_idx, item in enumerate(tqdm(anonymized_data, desc=f"Evaluating SGN {task.upper()}")):
            try:
                # Extract data from item
                skel = item['skeleton']
                action_label = item['action']
                ret_actor = item['retargeted_actor']
                orig_actor = item['original_actor']

                # Convert to numpy for preprocessing - fix data types
                skel_np = skel.cpu().numpy().astype(np.float32)

                # Handle NaN values which can cause issues
                skel_np = np.nan_to_num(skel_np)

                # Create preprocessed crops for testing using the same function as used during training
                processed_crops = sgn_preprocess_single_skeleton(skel_np, seg=20, dataset=self.dataset.upper())

                # Run model using the exact same function as eval_model.py
                pred = self._process_sample_sgn_style(model, processed_crops)

                # Store predictions and labels for metrics
                if task == 'ar':
                    all_ar_preds.append(pred)
                    all_ar_labels.append(action_label)

                    # Check correctness
                    if pred == action_label:
                        ar_correct += 1
                else:  # ri
                    all_ri_preds.append(pred)
                    all_ri_labels.append(ret_actor)

                    # Check correctness - RI task should predict retargeted actor
                    if pred == ret_actor:
                        ri_ret_correct += 1
                    elif pred == orig_actor:
                        ri_orig_correct += 1

            except Exception as e:
                print(f"Error processing item {item_idx}: {e}")
                continue

        # Calculate metrics exactly like eval_model.py
        if task == 'ar':
            accuracy = 100 * ar_correct / test_size if test_size > 0 else 0

            # Calculate F1, precision, recall
            if len(all_ar_preds) > 0:
                from sklearn.metrics import f1_score, precision_score, recall_score
                f1 = f1_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
                precision = precision_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
                recall = recall_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
            else:
                f1 = precision = recall = 0

            return {
                'accuracy': accuracy / 100.0,  # Convert to 0-1 range
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        else:  # ri
            # For RI task, the primary metric is retargeted actor accuracy
            ri_ret_acc = 100 * ri_ret_correct / test_size if test_size > 0 else 0
            ri_orig_acc = 100 * ri_orig_correct / test_size if test_size > 0 else 0

            # Calculate F1 for RI (using retargeted actor predictions)
            if len(all_ri_preds) > 0:
                from sklearn.metrics import f1_score
                f1 = f1_score(all_ri_labels, all_ri_preds, average='macro', zero_division=0)
            else:
                f1 = 0

            print(f"    RI Debug: ret_correct={ri_ret_correct}, orig_correct={ri_orig_correct}, total={test_size}")
            print(f"    RI Debug: ret_acc={ri_ret_acc:.2f}%, orig_acc={ri_orig_acc:.2f}%")

            return {
                'accuracy': ri_ret_acc / 100.0,  # Convert to 0-1 range (retargeted actor accuracy)
                'f1': f1,
                'retargeted_correct': ri_ret_correct,
                'original_correct': ri_orig_correct,
                'retargeted_accuracy': ri_ret_acc / 100.0,
                'original_accuracy': ri_orig_acc / 100.0,
                'total_samples': test_size
            }

    def _process_sample_sgn_style(self, model, crops):
        """Process sample using SGN's test-time evaluation method - exact copy from eval_model.py."""
        # Make sure crops is a tensor
        if isinstance(crops, np.ndarray):
            crops = torch.from_numpy(crops).float()

        # Move to device if needed
        if next(model.parameters()).is_cuda and not crops.is_cuda:
            crops = crops.cuda()
        elif not next(model.parameters()).is_cuda and crops.is_cuda:
            crops = crops.cpu()

        # Run the model on all 5 crops
        outputs = model(crops)

        # Reshape to get outputs in the right format
        if outputs.dim() == 2 and outputs.size(0) == 5:
            # Handle case where model returns (5, num_classes)
            outputs = outputs.unsqueeze(0)  # Add batch dimension

        # Average over the 5 crops, as done in SGN test_snippet
        outputs = outputs.mean(1)

        # Get the prediction
        _, pred = torch.max(outputs, 1)

        return pred.item()

    def evaluate_mixformer_model(self, model, data_loader, task='ar', is_mlm_data=False):
        """Evaluate Mixformer model using eval_model.py approach."""
        if model is None:
            return {'accuracy': 0.0, 'predictions': [], 'labels': []}

        model.eval()

        total_samples = 0
        correct_predictions = 0

        print(f"Evaluating Mixformer {task.upper()} model...")

        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc=f"Evaluating Mixformer {task.upper()}")):
            try:
                if is_mlm_data:
                    # MLM data comes as (data, labels, actors)
                    x_data, labels, actors = batch_content
                    if task == 'ar':
                        batch_labels = labels
                    else:  # ri
                        batch_labels = actors

                    for i in range(x_data.shape[0]):
                        skeleton = x_data[i]

                        # Handle tensor labels properly
                        if task == 'ar':
                            if isinstance(batch_labels, torch.Tensor):
                                label = batch_labels[i].item() if batch_labels[i].numel() == 1 else batch_labels[i, 0].item()
                            else:
                                label = int(batch_labels[i]) if hasattr(batch_labels[i], '__int__') else batch_labels[i]
                        else:  # ri
                            if isinstance(batch_labels, torch.Tensor):
                                label = batch_labels[i].item() if batch_labels[i].numel() == 1 else batch_labels[i, 0].item()
                            else:
                                label = int(batch_labels[i]) if hasattr(batch_labels[i], '__int__') else batch_labels[i]

                        # Ensure label is 0-indexed
                        if label > 0:
                            label = label - 1

                        # Preprocess for Mixformer
                        skel_np = skeleton.detach().cpu().numpy()
                        prepped = mixformer_preprocess_single_skeleton(skel_np)

                        # Add second empty person for Mixformer model
                        zeros_ = np.zeros_like(prepped)
                        prepped_2p = np.concatenate([prepped, zeros_], axis=3)
                        mixformer_input = torch.tensor(prepped_2p, dtype=torch.float32).unsqueeze(0).to(self.device)

                        # Run model
                        with torch.no_grad():
                            output = model(mixformer_input)
                            _, pred = torch.max(output, 1)

                        if pred.item() == label:
                            correct_predictions += 1
                        total_samples += 1
                else:
                    # Raw data comes as 6-item batch
                    if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                        # Use get_anonymized_paired_raw to process the batch properly
                        anonymized_data = get_anonymized_paired_raw(batch_content)

                        for item in anonymized_data:
                            skeleton = item['skeleton']
                            if task == 'ar':
                                label = item['action']
                            else:  # ri
                                label = item['original_actor']  # Use original actor for RI

                            # Preprocess for Mixformer
                            skel_np = skeleton.detach().cpu().numpy()
                            prepped = mixformer_preprocess_single_skeleton(skel_np)

                            # Add second empty person for Mixformer model
                            zeros_ = np.zeros_like(prepped)
                            prepped_2p = np.concatenate([prepped, zeros_], axis=3)
                            mixformer_input = torch.tensor(prepped_2p, dtype=torch.float32).unsqueeze(0).to(self.device)

                            # Run model
                            with torch.no_grad():
                                output = model(mixformer_input)
                                _, pred = torch.max(output, 1)

                            if pred.item() == label:
                                correct_predictions += 1
                            total_samples += 1

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions
        }


class RawVsMLMComparator:
    """Main class for comparing raw vs MLM data performance."""
    
    def __init__(self, mlm_model_dir, dataset, setting, seq_len, device):
        self.mlm_model_dir = mlm_model_dir
        self.dataset = dataset
        self.setting = setting
        self.seq_len = seq_len
        self.device = device
        
        # Initialize components
        self.mlm_processor = MLMDataProcessor(mlm_model_dir, dataset, seq_len, device)
        self.evaluator = DownstreamModelEvaluator(dataset, setting, device)
    
    def run_comprehensive_evaluation(self, train_loader, test_loader):
        """Run comprehensive evaluation comparing raw vs MLM data."""
        print("🚀 Starting Raw vs MLM Comprehensive Evaluation")
        print("=" * 60)
        print(f"📊 Dataset: {self.dataset.upper()}, Setting: {self.setting.upper()}")
        print(f"🤖 MLM Model: {os.path.basename(self.mlm_model_dir)}")
        
        results = {
            'dataset': self.dataset,
            'setting': self.setting,
            'mlm_model_dir': self.mlm_model_dir,
            'timestamp': datetime.now().isoformat(),
            'raw_performance': {},
            'mlm_performance': {},
            'comparison': {}
        }
        
        # Process data through MLM
        print("\n📊 Processing test data through MLM...")
        mlm_test_data, mlm_test_labels, mlm_test_actors = self.mlm_processor.process_data(test_loader)
        
        if mlm_test_data is None:
            print("❌ Failed to process data through MLM")
            return results
        
        # Create MLM data loader
        mlm_dataset = torch.utils.data.TensorDataset(mlm_test_data, torch.tensor(mlm_test_labels), torch.tensor(mlm_test_actors))
        mlm_test_loader = DataLoader(mlm_dataset, batch_size=32, shuffle=False)
        
        # Evaluate all model combinations
        model_types = ['sgn', 'mixformer']
        tasks = ['ar', 'ri']
        
        for model_type in model_types:
            print(f"\n🔍 Evaluating {model_type.upper()} models...")
            
            results['raw_performance'][model_type] = {}
            results['mlm_performance'][model_type] = {}
            
            for task in tasks:
                print(f"\n  📈 Task: {task.upper()}")
                
                # Load model based on type
                if model_type == 'sgn':
                    model = self.evaluator._load_sgn_model(task)
                else:  # mixformer
                    model = self.evaluator._load_mixformer_model(task)

                if model is not None:
                    # Evaluate on raw data
                    print("    🔸 Evaluating on raw data...")
                    if model_type == 'sgn':
                        raw_results = self.evaluator.evaluate_sgn_model(model, test_loader, task, is_mlm_data=False)
                    else:
                        raw_results = self.evaluator.evaluate_mixformer_model(model, test_loader, task, is_mlm_data=False)
                    results['raw_performance'][model_type][task] = raw_results

                    # Evaluate on MLM data
                    print("    🔹 Evaluating on MLM data...")
                    if model_type == 'sgn':
                        mlm_results = self.evaluator.evaluate_sgn_model(model, mlm_test_loader, task, is_mlm_data=True)
                    else:
                        mlm_results = self.evaluator.evaluate_mixformer_model(model, mlm_test_loader, task, is_mlm_data=True)
                    results['mlm_performance'][model_type][task] = mlm_results
                    
                    # Calculate improvement
                    improvement = mlm_results['accuracy'] - raw_results['accuracy']
                    results['comparison'][f"{model_type}_{task}"] = {
                        'raw_accuracy': raw_results['accuracy'],
                        'mlm_accuracy': mlm_results['accuracy'],
                        'improvement': improvement,
                        'improvement_percent': (improvement / max(raw_results['accuracy'], 1e-8)) * 100
                    }
                    
                    print(f"    📊 Raw: {raw_results['accuracy']:.4f}, MLM: {mlm_results['accuracy']:.4f}, Δ: {improvement:+.4f}")
                else:
                    print(f"    ❌ Skipping {model_type} {task} - model not available")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Raw vs MLM Evaluation System')
    
    parser.add_argument('--mlm-model-dir', type=str,
                        help='Directory containing pretrained MLM model')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'],
                        help='Dataset name')
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'],
                        help='Evaluation setting')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results/raw_vs_mlm_evaluation',
                        help='Output directory for results')
    parser.add_argument('--test-samples', type=int, default=2000,
                        help='Number of test samples')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    return parser.parse_args()


def interactive_mode():
    """Run evaluation in interactive mode."""
    print("🎯 Raw vs MLM Interactive Evaluation")
    print("=" * 50)

    # Dataset selection
    print("\n📊 Select Dataset:")
    print("1. NTU RGB+D 60")
    print("2. NTU RGB+D 120")
    print("3. ETRI")

    dataset_choice = input("Enter choice (1-3): ").strip()
    dataset_map = {'1': 'ntu', '2': 'ntu120', '3': 'etri'}
    dataset = dataset_map.get(dataset_choice, 'ntu')

    # Setting selection
    print(f"\n🎯 Select Setting for {dataset.upper()}:")
    print("1. Cross-Subject (cs)")
    print("2. Cross-View (cv)")

    setting_choice = input("Enter choice (1-2): ").strip()
    setting_map = {'1': 'cs', '2': 'cv'}
    setting = setting_map.get(setting_choice, 'cv')

    # MLM model selection
    print(f"\n🤖 Available MLM Models for {dataset}/{setting}:")
    base_mlm_dir = f"eval/mixformer/pretrained/{dataset}"

    if os.path.exists(base_mlm_dir):
        mlm_models = [d for d in os.listdir(base_mlm_dir)
                      if os.path.isdir(os.path.join(base_mlm_dir, d)) and 'comprehensive' in d]

        if mlm_models:
            for i, model in enumerate(mlm_models, 1):
                print(f"{i}. {model}")

            model_choice = input(f"Enter choice (1-{len(mlm_models)}): ").strip()
            try:
                mlm_model_dir = os.path.join(base_mlm_dir, mlm_models[int(model_choice)-1])
            except (ValueError, IndexError):
                mlm_model_dir = os.path.join(base_mlm_dir, mlm_models[0])
        else:
            print("❌ No MLM models found!")
            return
    else:
        print("❌ MLM model directory not found!")
        return

    # Test samples
    print(f"\n📈 Number of test samples:")
    print("1. Quick test (500 samples)")
    print("2. Standard test (2000 samples)")
    print("3. Comprehensive test (5000 samples)")
    print("4. Custom")

    samples_choice = input("Enter choice (1-4): ").strip()
    samples_map = {'1': 500, '2': 2000, '3': 5000}

    if samples_choice == '4':
        test_samples = int(input("Enter number of test samples: "))
    else:
        test_samples = samples_map.get(samples_choice, 2000)

    # Confirm configuration
    print(f"\n✅ Configuration Summary:")
    print(f"   Dataset: {dataset.upper()}")
    print(f"   Setting: {setting.upper()}")
    print(f"   MLM Model: {os.path.basename(mlm_model_dir)}")
    print(f"   Test Samples: {test_samples}")

    confirm = input("\nProceed with evaluation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Evaluation cancelled.")
        return

    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting evaluation on {device}...")

    # Create output directory
    output_dir = f"results/raw_vs_mlm_evaluation/{dataset}_{setting}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data and run evaluation
    X = load_data(dataset, T=64)
    _, test_loader = get_cross_data(
        X, dataset, setting,
        batch_size=32,
        return_loader=True,
        train_samples=100,
        test_samples=test_samples,
        seg=64
    )

    comparator = RawVsMLMComparator(mlm_model_dir, dataset, setting, 64, device)
    results = comparator.run_comprehensive_evaluation(None, test_loader)

    # Save and display results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"interactive_evaluation_{timestamp}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {result_file}")

    # Display summary
    print("\n" + "="*80)
    print("🎯 EVALUATION RESULTS SUMMARY")
    print("="*80)

    if 'comparison' in results:
        for key, comp in results['comparison'].items():
            model_type, task = key.split('_')
            improvement_indicator = "📈" if comp['improvement'] > 0 else "📉" if comp['improvement'] < 0 else "➡️"
            print(f"{improvement_indicator} {model_type.upper()} {task.upper()}: Raw={comp['raw_accuracy']:.4f}, MLM={comp['mlm_accuracy']:.4f}, Δ={comp['improvement']:+.4f} ({comp['improvement_percent']:+.1f}%)")

    print("="*80)


def main():
    args = parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.mlm_model_dir:
        print("❌ --mlm-model-dir is required when not in interactive mode")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data: {args.dataset} ({args.setting})")
    X = load_data(args.dataset, T=args.seq_len)

    # Get test data loader (we don't need training for evaluation)
    _, test_loader = get_cross_data(
        X, args.dataset, args.setting,
        batch_size=args.batch_size,
        return_loader=True,
        train_samples=100,  # Minimal training samples
        test_samples=args.test_samples,
        seg=args.seq_len
    )

    # Initialize comparator
    comparator = RawVsMLMComparator(
        args.mlm_model_dir, args.dataset, args.setting, args.seq_len, device
    )

    # Run evaluation
    results = comparator.run_comprehensive_evaluation(None, test_loader)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"raw_vs_mlm_evaluation_{args.dataset}_{args.setting}_{timestamp}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {result_file}")

    # Print summary
    print("\n" + "="*80)
    print("🎯 RAW vs MLM EVALUATION SUMMARY")
    print("="*80)

    if 'comparison' in results:
        for key, comp in results['comparison'].items():
            model_type, task = key.split('_')
            print(f"{model_type.upper()} {task.upper()}: Raw={comp['raw_accuracy']:.4f}, MLM={comp['mlm_accuracy']:.4f}, Δ={comp['improvement']:+.4f} ({comp['improvement_percent']:+.1f}%)")

    print("="*80)


if __name__ == "__main__":
    main()
