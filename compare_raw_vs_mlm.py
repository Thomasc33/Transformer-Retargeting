#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Raw Skeleton vs MLM Pretrained Features

This script compares the performance of:
1. Raw skeleton features (simple statistical features)
2. MLM pretrained encoder features
3. Random encoder features (as baseline)

This will help identify if the issue is with:
- MLM pretraining quality
- Feature extraction approach
- Data preprocessing
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrain import SkeletonAutoEncoder
from data import load_data, get_cross_data


class RawSkeletonFeatureExtractor:
    """Extract simple statistical features from raw skeleton data."""
    
    def __init__(self, device):
        self.device = device
    
    def extract_features(self, data_loader):
        """Extract statistical features from raw skeleton data."""
        features = []
        action_labels = []
        actor_labels = []
        
        print("Extracting raw skeleton features...")
        
        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Raw feature extraction")):
            try:
                # Parse batch content
                if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                    x1, x2, y1, y2, actors, actions = batch_content
                    x_data = x1
                    
                    # Convert to tensors and fix data types
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.tensor(actions, dtype=torch.long)
                    if not isinstance(actors, torch.Tensor):
                        actors = torch.tensor(actors, dtype=torch.long)
                    
                    actions = actions.long()
                    actors = actors.long()
                    
                    # Extract labels
                    action_batch = actions[:, 0] if len(actions.shape) > 1 else actions
                    actor_batch = actors[:, 0] if len(actors.shape) > 1 else actors
                    
                else:
                    continue
                
                # Process data format
                if not isinstance(x_data, torch.Tensor):
                    x_data = torch.tensor(x_data, dtype=torch.float32)
                
                # Reshape to (batch, frames, joints, channels)
                if len(x_data.shape) == 3 and x_data.shape[2] == 75:
                    x_data = x_data.view(x_data.shape[0], x_data.shape[1], 25, 3)
                elif len(x_data.shape) == 2 and x_data.shape[1] == 75:
                    x_data = x_data.view(1, x_data.shape[0], 25, 3)
                
                batch_size = x_data.shape[0]
                
                # Extract statistical features
                batch_features = []
                for i in range(batch_size):
                    sample = x_data[i]  # (frames, joints, channels)
                    
                    # Statistical features
                    mean_features = sample.mean(dim=0).flatten()  # Mean over time
                    std_features = sample.std(dim=0).flatten()    # Std over time
                    max_features = sample.max(dim=0)[0].flatten() # Max over time
                    min_features = sample.min(dim=0)[0].flatten() # Min over time
                    
                    # Velocity features (difference between consecutive frames)
                    velocity = sample[1:] - sample[:-1]
                    vel_mean = velocity.mean(dim=0).flatten()
                    vel_std = velocity.std(dim=0).flatten()
                    
                    # Combine all features
                    sample_features = torch.cat([
                        mean_features, std_features, max_features, min_features,
                        vel_mean, vel_std
                    ])
                    
                    batch_features.append(sample_features)
                
                batch_features = torch.stack(batch_features)
                
                features.append(batch_features)
                action_labels.append(action_batch)
                actor_labels.append(actor_batch)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if features:
            all_features = torch.cat(features, dim=0)
            all_action_labels = torch.cat(action_labels, dim=0)
            all_actor_labels = torch.cat(actor_labels, dim=0)
            
            print(f"Raw features extracted:")
            print(f"  Features: {all_features.shape}")
            print(f"  Action labels: {all_action_labels.shape}")
            print(f"  Actor labels: {all_actor_labels.shape}")
            
            return all_features, all_action_labels, all_actor_labels
        else:
            return None, None, None


class RandomFeatureExtractor:
    """Extract random features as baseline."""
    
    def __init__(self, device, feature_dim=320):
        self.device = device
        self.feature_dim = feature_dim
    
    def extract_features(self, data_loader):
        """Extract random features."""
        features = []
        action_labels = []
        actor_labels = []
        
        print("Extracting random features...")
        
        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Random feature extraction")):
            try:
                # Parse batch content
                if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                    x1, x2, y1, y2, actors, actions = batch_content
                    
                    # Convert to tensors and fix data types
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.tensor(actions, dtype=torch.long)
                    if not isinstance(actors, torch.Tensor):
                        actors = torch.tensor(actors, dtype=torch.long)
                    
                    actions = actions.long()
                    actors = actors.long()
                    
                    # Extract labels
                    action_batch = actions[:, 0] if len(actions.shape) > 1 else actions
                    actor_batch = actors[:, 0] if len(actors.shape) > 1 else actors
                    
                    batch_size = len(action_batch)
                    
                    # Generate random features
                    random_features = torch.randn(batch_size, self.feature_dim)
                    
                    features.append(random_features)
                    action_labels.append(action_batch)
                    actor_labels.append(actor_batch)
                    
                else:
                    continue
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if features:
            all_features = torch.cat(features, dim=0)
            all_action_labels = torch.cat(action_labels, dim=0)
            all_actor_labels = torch.cat(actor_labels, dim=0)
            
            print(f"Random features extracted:")
            print(f"  Features: {all_features.shape}")
            print(f"  Action labels: {all_action_labels.shape}")
            print(f"  Actor labels: {all_actor_labels.shape}")
            
            return all_features, all_action_labels, all_actor_labels
        else:
            return None, None, None


class SimpleClassifier(nn.Module):
    """Simple classifier for comparison."""

    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(SimpleClassifier, self).__init__()
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


def train_and_evaluate_classifier(train_features, train_labels, test_features, test_labels, 
                                device, feature_name, epochs=30):
    """Train and evaluate a classifier."""
    
    # Fix label indexing
    unique_labels = torch.unique(train_labels)
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    
    # Map training labels
    mapped_train_labels = torch.zeros_like(train_labels)
    for i, label in enumerate(train_labels):
        mapped_train_labels[i] = label_mapping[label.item()]
    
    # Map test labels (only include labels seen in training)
    mapped_test_labels = []
    valid_test_features = []
    for i, label in enumerate(test_labels):
        if label.item() in label_mapping:
            mapped_test_labels.append(label_mapping[label.item()])
            valid_test_features.append(test_features[i])
    
    if not mapped_test_labels:
        print(f"No valid test labels for {feature_name}")
        return {'accuracy': 0.0, 'f1_score': 0.0}
    
    mapped_test_labels = torch.tensor(mapped_test_labels, dtype=torch.long)
    valid_test_features = torch.stack(valid_test_features)
    
    num_classes = len(unique_labels)
    
    print(f"\nTraining {feature_name} classifier:")
    print(f"  Train: {train_features.shape[0]} samples, {num_classes} classes")
    print(f"  Test: {valid_test_features.shape[0]} samples")
    
    # Create dataset and data loader
    dataset = TensorDataset(train_features.float(), mapped_train_labels.long())
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleClassifier(
        input_dim=train_features.shape[1],
        num_classes=num_classes
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        valid_test_features = valid_test_features.to(device)
        outputs = model(valid_test_features)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        true_labels = mapped_test_labels.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted)
    f1 = f1_score(true_labels, predicted, average='weighted')
    
    print(f"  Results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return {'accuracy': accuracy, 'f1_score': f1}


def main():
    """Main comparison function."""
    print("🚀 Comparing Raw vs MLM vs Random Features")
    print("=" * 60)
    
    # Configuration
    dataset = 'ntu'
    setting = 'cv'
    seq_len = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("Loading data...")
    X = load_data(dataset, T=seq_len)
    
    # Get test data loaders
    train_loader, test_loader = get_cross_data(
        X, dataset, setting,
        batch_size=8,
        return_loader=True,
        train_samples=200,  # Larger sample for better comparison
        test_samples=100,
        seg=seq_len
    )
    
    results = {}
    
    # 1. Raw skeleton features
    print("\n=== Raw Skeleton Features ===")
    raw_extractor = RawSkeletonFeatureExtractor(device)
    raw_train_features, raw_train_actions, raw_train_actors = raw_extractor.extract_features(train_loader)
    raw_test_features, raw_test_actions, raw_test_actors = raw_extractor.extract_features(test_loader)
    
    if raw_train_features is not None:
        results['raw_ar'] = train_and_evaluate_classifier(
            raw_train_features, raw_train_actions, raw_test_features, raw_test_actions,
            device, "Raw AR"
        )
        results['raw_ri'] = train_and_evaluate_classifier(
            raw_train_features, raw_train_actors, raw_test_features, raw_test_actors,
            device, "Raw RI"
        )
    
    # 2. Random features
    print("\n=== Random Features ===")
    random_extractor = RandomFeatureExtractor(device, feature_dim=320)
    random_train_features, random_train_actions, random_train_actors = random_extractor.extract_features(train_loader)
    random_test_features, random_test_actions, random_test_actors = random_extractor.extract_features(test_loader)
    
    if random_train_features is not None:
        results['random_ar'] = train_and_evaluate_classifier(
            random_train_features, random_train_actions, random_test_features, random_test_actions,
            device, "Random AR"
        )
        results['random_ri'] = train_and_evaluate_classifier(
            random_train_features, random_train_actors, random_test_features, random_test_actors,
            device, "Random RI"
        )
    
    # 3. MLM features (using our fixed extractor)
    print("\n=== MLM Features ===")
    import glob
    pattern = f"eval/mixformer/pretrained/{dataset}/epochs_{setting}_comprehensive_temporal_*_spatial_*"
    available_dirs = glob.glob(pattern)
    
    if available_dirs:
        from fix_mlm_evaluation import FixedMLMFeatureExtractor
        
        model_dir = available_dirs[0]
        mlm_extractor = FixedMLMFeatureExtractor(model_dir, dataset, seq_len, device)
        mlm_train_features, mlm_train_actions, mlm_train_actors = mlm_extractor.extract_features_fixed(train_loader)
        mlm_test_features, mlm_test_actions, mlm_test_actors = mlm_extractor.extract_features_fixed(test_loader)
        
        if mlm_train_features is not None:
            results['mlm_ar'] = train_and_evaluate_classifier(
                mlm_train_features, mlm_train_actions, mlm_test_features, mlm_test_actions,
                device, "MLM AR"
            )
            results['mlm_ri'] = train_and_evaluate_classifier(
                mlm_train_features, mlm_train_actors, mlm_test_features, mlm_test_actors,
                device, "MLM RI"
            )
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for key, result in results.items():
        print(f"{key:15}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
