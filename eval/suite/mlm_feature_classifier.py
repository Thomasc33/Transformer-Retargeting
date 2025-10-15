#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLM Feature-Based Classifier for evaluating pretrained encoder embeddings.

This module trains lightweight classifiers on MLM encoder features to evaluate
the quality of learned representations for action recognition and re-identification.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.pretrain import SkeletonAutoEncoder
from src.data import load_data, get_cross_data


class MLMFeatureClassifier(nn.Module):
    """Lightweight classifier for MLM encoder features."""

    def __init__(self, input_dim=320, num_classes=60, dropout=0.3):
        super(MLMFeatureClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class MLMFeatureExtractor:
    """Extract features from pretrained MLM encoder."""

    def __init__(self, model_dir, dataset, seq_len, device):
        self.device = device
        self.model = self.load_mlm_model(model_dir, dataset, seq_len)

        # Freeze the encoder to prevent updates during classification training
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen - parameters will not be updated during classification training")

    def load_mlm_model(self, model_dir, dataset, seq_len):
        """Load pretrained MLM model."""
        print(f"Loading MLM model from: {model_dir}")
        print(f"Dataset: {dataset}, Seq length: {seq_len}")

        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # List contents of model directory
        model_files = os.listdir(model_dir)
        print(f"Files in model directory: {model_files}")

        model = SkeletonAutoEncoder(dataset=dataset, seq_len=seq_len).to(self.device)

        # Load encoder weights
        encoder_path = os.path.join(model_dir, 'encoder_best.pth')
        if os.path.exists(encoder_path):
            model.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(f"Successfully loaded encoder weights from {encoder_path}")
        else:
            raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")

        model.eval()
        print("MLM model loaded and set to eval mode")
        return model

    @torch.no_grad()
    def extract_features(self, data_loader):
        """Extract features from data loader using FIXED approach."""
        features = []
        labels = []

        total_batches = 0
        processed_batches = 0
        skipped_batches = 0

        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Extracting features (FIXED)")):
            total_batches += 1

            # Handle Cross_Data format: (x1, x2, y1, y2, actors, actions)
            if isinstance(batch_content, (tuple, list)):
                if len(batch_content) == 6:
                    x1, x2, y1, y2, actors, actions = batch_content

                    # Use x1 as the primary skeleton data
                    x_data = x1

                    # Convert to tensors if needed
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.tensor(actions, dtype=torch.long)
                    if not isinstance(actors, torch.Tensor):
                        actors = torch.tensor(actors, dtype=torch.long)

                    # Ensure correct data types
                    actions = actions.long()
                    actors = actors.long()

                    # Extract labels - use first column
                    action_labels = actions[:, 0] if len(actions.shape) > 1 else actions
                    actor_labels = actors[:, 0] if len(actors.shape) > 1 else actors

                else:
                    skipped_batches += 1
                    continue
            else:
                skipped_batches += 1
                continue

            # Process data format for encoder
            if not isinstance(x_data, torch.Tensor):
                x_data = torch.tensor(x_data, dtype=torch.float32)

            # Handle different input shapes
            if len(x_data.shape) == 3 and x_data.shape[2] == 75:  # (batch, frames, joints*channels)
                x_data = x_data.view(x_data.shape[0], x_data.shape[1], 25, 3)
            elif len(x_data.shape) == 2 and x_data.shape[1] == 75:  # (frames, joints*channels)
                x_data = x_data.view(1, x_data.shape[0], 25, 3)
            elif len(x_data.shape) == 4:  # Already (batch, frames, joints, channels)
                pass  # Already in correct format
            else:
                skipped_batches += 1
                continue

            x_data = x_data.to(self.device)
            batch_size = x_data.shape[0]

            # Extract encoder features
            try:
                # Prepare input for encoder: (batch, channels, frames, joints, persons)
                if len(x_data.shape) == 4:  # (batch, frames, joints, channels)
                    x_encoder = x_data.permute(0, 3, 1, 2).unsqueeze(-1)  # (batch, channels, frames, joints, 1)
                else:
                    skipped_batches += 1
                    continue

                # Get encoder output
                encoder_output = self.model.encoder(x_encoder)

                # FIXED: Handle encoder output properly
                if len(encoder_output.shape) == 3:  # (seq_len, batch_size, feature_dim)
                    seq_len, batch_size_out, feature_dim = encoder_output.shape

                    # Use multiple pooling strategies for richer features
                    avg_pooled = encoder_output.mean(dim=0)  # (batch_size, feature_dim)
                    max_pooled = encoder_output.max(dim=0)[0]  # (batch_size, feature_dim)
                    first_token = encoder_output[0]  # (batch_size, feature_dim)
                    last_token = encoder_output[-1]  # (batch_size, feature_dim)

                    # Concatenate for richer representation
                    pooled_features = torch.cat([
                        avg_pooled, max_pooled, first_token, last_token
                    ], dim=1)  # (batch_size, 4*feature_dim)

                    # Ensure batch size matches
                    if batch_size_out != batch_size:
                        pooled_features = pooled_features[:batch_size]

                elif len(encoder_output.shape) == 4:  # (batch, features, frames, joints)
                    pooled_features = encoder_output.mean(dim=[2, 3])  # (batch, features)
                elif len(encoder_output.shape) == 2:  # (batch, features)
                    pooled_features = encoder_output
                else:
                    print(f"⚠️ Unexpected encoder output shape: {encoder_output.shape}")
                    skipped_batches += 1
                    continue

                # Ensure we have the right number of features for the batch
                if pooled_features.shape[0] != batch_size:
                    print(f"⚠️ Feature count mismatch: expected {batch_size}, got {pooled_features.shape[0]}")
                    if pooled_features.shape[0] > batch_size:
                        pooled_features = pooled_features[:batch_size]
                    else:
                        skipped_batches += 1
                        continue

                # Ensure label counts match feature counts
                if len(action_labels) != batch_size or len(actor_labels) != batch_size:
                    print(f"⚠️ Label count mismatch: features {batch_size}, actions {len(action_labels)}, actors {len(actor_labels)}")
                    skipped_batches += 1
                    continue

                features.append(pooled_features.cpu())
                labels.append({
                    'action': action_labels.cpu(),
                    'actor': actor_labels.cpu()
                })

                processed_batches += 1

            except Exception as e:
                print(f"Error extracting features for batch {batch_idx}: {e}")
                skipped_batches += 1
                continue

        # Print summary
        print(f"\n=== Feature Extraction Summary ===")
        print(f"Total batches: {total_batches}")
        print(f"Processed batches: {processed_batches}")
        print(f"Skipped batches: {skipped_batches}")
        print(f"Features extracted: {len(features)}")

        if not features:
            print("ERROR: No features were extracted!")
            print("This could be due to:")
            print("1. Empty data loader")
            print("2. All batches being skipped due to format issues")
            print("3. Model loading issues")
            print("4. Data format incompatibility")
            raise ValueError("No features extracted from data loader")

        # Validate batch consistency (only show if there are issues)
        batch_sizes = [feat.shape[0] for feat in features]
        if len(set(batch_sizes)) > 1:
            print("Warning: Inconsistent batch sizes detected:")
            for i, (feat, lab) in enumerate(zip(features, labels)):
                print(f"  Batch {i}: features {feat.shape}, actions {lab['action'].shape}, actors {lab['actor'].shape}")

        # Concatenate all features and labels
        all_features = torch.cat(features, dim=0)
        all_action_labels = torch.cat([l['action'] for l in labels], dim=0)
        all_actor_labels = torch.cat([l['actor'] for l in labels], dim=0)

        print(f"Final feature tensor shape: {all_features.shape}")
        print(f"Final action labels shape: {all_action_labels.shape}")
        print(f"Final actor labels shape: {all_actor_labels.shape}")

        # Final validation
        if all_features.shape[0] != all_action_labels.shape[0] or all_features.shape[0] != all_actor_labels.shape[0]:
            print("WARNING: Feature and label shapes don't match after concatenation!")
            print("This indicates an issue with label repetition logic.")

            # Fix by truncating to minimum size
            min_size = min(all_features.shape[0], all_action_labels.shape[0], all_actor_labels.shape[0])
            print(f"Truncating all tensors to size {min_size}")
            all_features = all_features[:min_size]
            all_action_labels = all_action_labels[:min_size]
            all_actor_labels = all_actor_labels[:min_size]

            print(f"After truncation:")
            print(f"  Features shape: {all_features.shape}")
            print(f"  Action labels shape: {all_action_labels.shape}")
            print(f"  Actor labels shape: {all_actor_labels.shape}")

        return all_features, all_action_labels, all_actor_labels


def train_classifier(features, labels, num_classes, device, epochs=200, lr=1e-3, batch_size=64):
    """Train a classifier on extracted features with FIXED label handling."""

    # Ensure features and labels have the same first dimension
    if features.shape[0] != labels.shape[0]:
        min_size = min(features.shape[0], labels.shape[0])
        print(f"Warning: Size mismatch detected! Truncating to {min_size} samples")
        features = features[:min_size]
        labels = labels[:min_size]

    # FIXED: Handle label indexing properly
    features = features.float()
    labels = labels.long()

    print(f"Training classifier:")
    print(f"  Features: {features.shape}, dtype: {features.dtype}")
    print(f"  Labels: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Unique labels: {torch.unique(labels).shape[0]}")
    print(f"  Label range: {labels.min().item()}-{labels.max().item()}")

    # Fix label indexing - convert to 0-based indexing
    unique_labels = torch.unique(labels)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}

    # Map labels to 0-based indexing
    mapped_labels = torch.zeros_like(labels)
    for i, label in enumerate(labels):
        mapped_labels[i] = label_mapping[label.item()]

    labels = mapped_labels
    actual_num_classes = len(unique_labels)

    print(f"  After mapping: {actual_num_classes} classes, range: {labels.min().item()}-{labels.max().item()}")

    # Create dataset and data loader
    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with actual number of classes
    model = MLMFeatureClassifier(
        input_dim=features.shape[1],
        num_classes=actual_num_classes
    ).to(device)

    # Loss and optimizer with better settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=False
    )

    # Training loop with early stopping
    model.train()
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 50

    print(f"Training classifier for {num_classes} classes with {features.shape[0]} samples...")
    print(f"Feature dimension: {features.shape[1]}")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        # Calculate epoch metrics
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)

        # Learning rate scheduling
        scheduler.step(accuracy)

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}')

        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1} (best accuracy: {best_accuracy:.2f}%)")
            break

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    return model


def evaluate_classifier(model, features, labels, device):
    """Evaluate classifier performance."""
    model.eval()

    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(labels, predicted)
    f1 = f1_score(labels, predicted, average='weighted')

    # Classification report
    report = classification_report(labels, predicted, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(labels, predicted)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predicted,
        'true_labels': labels
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLM Feature-Based Classification')

    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing pretrained MLM model')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'],
                        help='Dataset name')
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'],
                        help='Evaluation setting')
    parser.add_argument('--temporal-ratio', type=float, required=True,
                        help='Temporal masking ratio')
    parser.add_argument('--spatial-ratio', type=float, required=True,
                        help='Spatial masking ratio')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--classifier-epochs', type=int, default=50,
                        help='Training epochs for classifier')
    parser.add_argument('--classifier-lr', type=float, default=1e-3,
                        help='Learning rate for classifier')
    parser.add_argument('--output-dir', type=str, default='results/mlm_classification',
                        help='Output directory for results')
    parser.add_argument('--train-samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=2000,
                        help='Number of test samples')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data: {args.dataset} ({args.setting})")
    X = load_data(args.dataset, T=args.seq_len)

    # Get train/test data loaders
    train_loader, test_loader = get_cross_data(
        X, args.dataset, args.setting,
        batch_size=args.batch_size,
        return_loader=True,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seg=args.seq_len
    )

    # Initialize feature extractor
    print("Loading MLM model and extracting features...")
    extractor = MLMFeatureExtractor(args.model_dir, args.dataset, args.seq_len, device)

    # Extract features
    print("Extracting training features...")
    train_features, train_action_labels, train_actor_labels = extractor.extract_features(train_loader)

    print("Extracting test features...")
    test_features, test_action_labels, test_actor_labels = extractor.extract_features(test_loader)

    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Get number of classes
    if args.dataset == 'ntu':
        num_action_classes = 60
        num_actor_classes = 40
    elif args.dataset == 'ntu120':
        num_action_classes = 120
        num_actor_classes = 106
    else:  # etri
        num_action_classes = 55
        num_actor_classes = 100

    results = {}

    # Train and evaluate action recognition classifier
    print("\n=== Training Action Recognition Classifier ===")
    ar_model = train_classifier(
        train_features, train_action_labels, num_action_classes,
        device, args.classifier_epochs, args.classifier_lr
    )

    print("Evaluating Action Recognition...")
    ar_results = evaluate_classifier(ar_model, test_features, test_action_labels, device)
    results['action_recognition'] = ar_results

    print(f"AR Accuracy: {ar_results['accuracy']:.4f}")
    print(f"AR F1-Score: {ar_results['f1_score']:.4f}")

    # Train and evaluate re-identification classifier
    print("\n=== Training Re-Identification Classifier ===")
    ri_model = train_classifier(
        train_features, train_actor_labels, num_actor_classes,
        device, args.classifier_epochs, args.classifier_lr
    )

    print("Evaluating Re-Identification...")
    ri_results = evaluate_classifier(ri_model, test_features, test_actor_labels, device)
    results['re_identification'] = ri_results

    print(f"RI Accuracy: {ri_results['accuracy']:.4f}")
    print(f"RI F1-Score: {ri_results['f1_score']:.4f}")

    # Save results
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.setting}_temporal_{args.temporal_ratio}_spatial_{args.spatial_ratio}_classification.json"
    )

    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'dataset': args.dataset,
        'setting': args.setting,
        'temporal_ratio': args.temporal_ratio,
        'spatial_ratio': args.spatial_ratio,
        'action_recognition': {
            'accuracy': float(ar_results['accuracy']),
            'f1_score': float(ar_results['f1_score'])
        },
        're_identification': {
            'accuracy': float(ri_results['accuracy']),
            'f1_score': float(ri_results['f1_score'])
        }
    }

    with open(result_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    # Save models
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    ar_model_path = os.path.join(
        model_dir,
        f"{args.dataset}_{args.setting}_temporal_{args.temporal_ratio}_spatial_{args.spatial_ratio}_ar_classifier.pth"
    )
    ri_model_path = os.path.join(
        model_dir,
        f"{args.dataset}_{args.setting}_temporal_{args.temporal_ratio}_spatial_{args.spatial_ratio}_ri_classifier.pth"
    )

    torch.save(ar_model.state_dict(), ar_model_path)
    torch.save(ri_model.state_dict(), ri_model_path)

    print(f"Models saved to: {model_dir}")
    print("Classification evaluation completed!")
