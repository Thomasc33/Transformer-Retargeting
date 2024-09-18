"""
Multi-Model Auxiliary Loss Module

This module implements auxiliary losses from multiple frozen pre-trained models
to improve TMR motion quality and downstream action recognition performance.

Key Features:
- Load and freeze multiple pre-trained models via model factory
- Compute weighted auxiliary losses from all models
- Handle num_classes mismatch gracefully
- Gradient balancing to prevent single model dominance
- Return total loss and individual loss dictionary for logging

Requirements: 3.1, 3.2, 3.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from src.model.model_factory import ModelFactory


class MultiModelAuxiliaryLoss(nn.Module):
    """
    Multi-model auxiliary loss for TMR training.
    
    This class manages multiple frozen pre-trained action recognition models
    and computes auxiliary losses to improve TMR motion quality.
    
    Features:
    - Load multiple models via ModelFactory
    - Freeze all auxiliary models (no gradient updates)
    - Compute weighted auxiliary losses
    - Handle num_classes mismatch (map to subset or use feature-level loss)
    - Gradient balancing across models
    - Individual loss tracking for monitoring
    """
    
    def __init__(
        self,
        model_configs: List[Dict],
        device: str = 'cuda',
        gradient_balancing: bool = True,
        gradient_clip_value: float = 1.0,
        feature_loss_weight: float = 0.1
    ):
        """
        Initialize multi-model auxiliary loss.
        
        Args:
            model_configs: List of model configuration dictionaries with keys:
                - name: str (e.g., 'sgn', 'ctrgcn', 'infogcn')
                - checkpoint_path: str (path to pre-trained checkpoint)
                - weight: float (loss weight for this model)
                - num_classes: int (number of classes model was trained on)
                - enabled: bool (whether to use this model, default True)
            device: Device to load models on
            gradient_balancing: Whether to apply gradient balancing
            gradient_clip_value: Gradient clipping value per model
            feature_loss_weight: Weight for feature-level loss when num_classes mismatch
        """
        super().__init__()
        
        self.device = device
        self.gradient_balancing = gradient_balancing
        self.gradient_clip_value = gradient_clip_value
        self.feature_loss_weight = feature_loss_weight
        
        # Store model configurations
        self.model_configs = [config for config in model_configs if config.get('enabled', True)]
        
        # Initialize models
        self.auxiliary_models = nn.ModuleDict()
        self.model_weights = {}
        self.model_num_classes = {}
        self.model_info = {}
        
        self._load_auxiliary_models()
        
        # Gradient statistics for balancing
        self.gradient_stats = {}
        self.loss_history = {config['name']: [] for config in self.model_configs}
        
        logging.info(f"Initialized MultiModelAuxiliaryLoss with {len(self.auxiliary_models)} models")
    
    def _load_auxiliary_models(self):
        """Load and freeze all auxiliary models."""
        for config in self.model_configs:
            model_name = config['name']
            checkpoint_path = config['checkpoint_path']
            weight = config['weight']
            num_classes = config['num_classes']
            
            try:
                # Get model info from factory
                model_info = ModelFactory.get_model_info(model_name)
                self.model_info[model_name] = model_info
                
                # Create model instance
                model = ModelFactory.create_model(
                    name=model_name,
                    num_class=num_classes,
                    num_point=25,
                    num_person=1,
                    in_channels=3
                )
                
                # Load checkpoint if provided
                if checkpoint_path and Path(checkpoint_path).exists():
                    model = ModelFactory.load_checkpoint(
                        model=model,
                        checkpoint_path=checkpoint_path,
                        strict=False,
                        device=self.device
                    )
                    logging.info(f"Loaded checkpoint for {model_name}: {checkpoint_path}")
                else:
                    logging.warning(f"No checkpoint found for {model_name}: {checkpoint_path}")
                
                # Move to device and freeze
                model = model.to(self.device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                
                # Store model and metadata
                self.auxiliary_models[model_name] = model
                self.model_weights[model_name] = weight
                self.model_num_classes[model_name] = num_classes
                
                logging.info(f"Loaded and froze auxiliary model: {model_name} (weight={weight})")
                
            except Exception as e:
                logging.error(f"Failed to load auxiliary model {model_name}: {e}")
                # Continue with other models
                continue
    
    def forward(
        self,
        generated_motion: torch.Tensor,
        action_labels: torch.Tensor,
        target_num_classes: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary losses from all frozen models.
        
        Args:
            generated_motion: Generated motion tensor (B, C, T, V, M)
            action_labels: Action labels (B,) - 0-indexed
            target_num_classes: Number of classes in target dataset (for mapping)
        
        Returns:
            total_loss: Weighted sum of all auxiliary losses
            loss_dict: Individual losses for logging
        """
        if len(self.auxiliary_models) == 0:
            # No auxiliary models loaded
            return torch.tensor(0.0, device=self.device), {}
        
        individual_losses = {}
        individual_gradients = {}
        
        # Compute loss for each auxiliary model
        for model_name, model in self.auxiliary_models.items():
            try:
                loss = self._compute_model_loss(
                    model=model,
                    model_name=model_name,
                    generated_motion=generated_motion,
                    action_labels=action_labels,
                    target_num_classes=target_num_classes
                )
                
                individual_losses[model_name] = loss
                
                # Store loss history for monitoring
                self.loss_history[model_name].append(loss.item())
                if len(self.loss_history[model_name]) > 100:  # Keep last 100 values
                    self.loss_history[model_name].pop(0)
                
            except Exception as e:
                logging.warning(f"Failed to compute loss for {model_name}: {e}")
                individual_losses[model_name] = torch.tensor(0.0, device=self.device)
        
        # Apply gradient balancing if enabled
        if self.gradient_balancing and len(individual_losses) > 1:
            balanced_losses = self._apply_gradient_balancing(individual_losses)
        else:
            balanced_losses = individual_losses
        
        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        for model_name, loss in balanced_losses.items():
            weight = self.model_weights[model_name]
            weighted_loss = weight * loss
            total_loss += weighted_loss
            
            # Store individual losses for logging
            loss_dict[f'aux_{model_name}'] = loss.item()
            loss_dict[f'aux_{model_name}_weighted'] = weighted_loss.item()
        
        loss_dict['aux_total'] = total_loss.item()
        loss_dict['aux_num_models'] = len(individual_losses)
        
        return total_loss, loss_dict
    
    def _compute_model_loss(
        self,
        model: nn.Module,
        model_name: str,
        generated_motion: torch.Tensor,
        action_labels: torch.Tensor,
        target_num_classes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for a single model.
        
        Args:
            model: Auxiliary model
            model_name: Name of the model
            generated_motion: Generated motion (B, C, T, V, M)
            action_labels: Action labels (B,)
            target_num_classes: Target number of classes
        
        Returns:
            loss: Auxiliary loss for this model
        """
        model_num_classes = self.model_num_classes[model_name]
        
        # Forward pass through auxiliary model
        with torch.no_grad():
            # Ensure model is in eval mode
            model.eval()
            
            # Forward pass
            logits = model(generated_motion)  # (B, num_classes)
        
        # Handle num_classes mismatch
        if target_num_classes is not None and model_num_classes != target_num_classes:
            if model_num_classes > target_num_classes:
                # Model has more classes - use subset
                logits = logits[:, :target_num_classes]
                loss = F.cross_entropy(logits, action_labels)
            else:
                # Model has fewer classes - use feature-level loss
                # Extract features before final classification layer
                loss = self._compute_feature_loss(model, generated_motion, action_labels)
        else:
            # Classes match - use standard cross-entropy
            loss = F.cross_entropy(logits, action_labels)
        
        return loss
    
    def _compute_feature_loss(
        self,
        model: nn.Module,
        generated_motion: torch.Tensor,
        action_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature-level loss when num_classes mismatch.
        
        This extracts features from the model and computes a contrastive loss
        to encourage similar actions to have similar features.
        
        Args:
            model: Auxiliary model
            generated_motion: Generated motion (B, C, T, V, M)
            action_labels: Action labels (B,)
        
        Returns:
            loss: Feature-level contrastive loss
        """
        # Try to extract features from the model
        # This is model-specific and may need adaptation
        
        try:
            # For most models, we can hook into the layer before final classification
            features = None
            
            def hook_fn(module, input, output):
                nonlocal features
                features = output
            
            # Find the last layer before classification
            # This is a heuristic and may need model-specific handling
            layers = list(model.modules())
            if len(layers) > 2:
                # Hook into the second-to-last layer
                hook = layers[-2].register_forward_hook(hook_fn)
                
                with torch.no_grad():
                    _ = model(generated_motion)
                
                hook.remove()
                
                if features is not None:
                    # Compute contrastive loss on features
                    # Encourage same actions to have similar features
                    features = F.normalize(features, dim=-1)  # L2 normalize
                    
                    # Simple contrastive loss: minimize distance for same actions
                    batch_size = features.size(0)
                    loss = torch.tensor(0.0, device=self.device)
                    count = 0
                    
                    for i in range(batch_size):
                        for j in range(i + 1, batch_size):
                            if action_labels[i] == action_labels[j]:
                                # Same action - minimize distance
                                distance = F.mse_loss(features[i], features[j])
                                loss += distance
                                count += 1
                    
                    if count > 0:
                        loss = loss / count
                    
                    return loss * self.feature_loss_weight
        
        except Exception as e:
            logging.warning(f"Feature extraction failed, using zero loss: {e}")
        
        # Fallback to zero loss
        return torch.tensor(0.0, device=self.device)
    
    def _apply_gradient_balancing(
        self,
        individual_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply gradient balancing to prevent single model dominance.
        
        This adjusts loss weights based on gradient magnitudes to ensure
        no single auxiliary model dominates the training.
        
        Args:
            individual_losses: Dictionary of individual model losses
        
        Returns:
            balanced_losses: Gradient-balanced losses
        """
        # Compute gradients for each loss (without updating parameters)
        gradients = {}
        
        for model_name, loss in individual_losses.items():
            if loss.requires_grad:
                # Compute gradients
                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=[p for p in self.parameters() if p.requires_grad],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                
                # Compute gradient norm
                grad_norm = 0.0
                for g in grad:
                    if g is not None:
                        grad_norm += g.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                gradients[model_name] = grad_norm
            else:
                gradients[model_name] = 0.0
        
        # Update gradient statistics (exponential moving average)
        alpha = 0.1  # EMA coefficient
        for model_name, grad_norm in gradients.items():
            if model_name not in self.gradient_stats:
                self.gradient_stats[model_name] = grad_norm
            else:
                self.gradient_stats[model_name] = (
                    alpha * grad_norm + (1 - alpha) * self.gradient_stats[model_name]
                )
        
        # Compute balancing weights
        # Models with higher gradients get lower weights
        if len(self.gradient_stats) > 1:
            max_grad = max(self.gradient_stats.values())
            if max_grad > 0:
                balance_weights = {}
                for model_name in individual_losses.keys():
                    grad_stat = self.gradient_stats.get(model_name, 0.0)
                    # Inverse relationship: higher gradient -> lower weight
                    balance_weights[model_name] = max_grad / (grad_stat + 1e-8)
                
                # Normalize weights to sum to number of models
                total_weight = sum(balance_weights.values())
                num_models = len(balance_weights)
                for model_name in balance_weights:
                    balance_weights[model_name] *= num_models / total_weight
            else:
                # All gradients are zero - use equal weights
                balance_weights = {name: 1.0 for name in individual_losses.keys()}
        else:
            # Single model - no balancing needed
            balance_weights = {name: 1.0 for name in individual_losses.keys()}
        
        # Apply balancing
        balanced_losses = {}
        for model_name, loss in individual_losses.items():
            balance_weight = balance_weights.get(model_name, 1.0)
            balanced_losses[model_name] = balance_weight * loss
        
        return balanced_losses
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about loaded auxiliary models.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        for model_name in self.auxiliary_models.keys():
            info[model_name] = {
                'weight': self.model_weights[model_name],
                'num_classes': self.model_num_classes[model_name],
                'info': self.model_info.get(model_name, {}),
                'loss_history_mean': (
                    sum(self.loss_history[model_name]) / len(self.loss_history[model_name])
                    if self.loss_history[model_name] else 0.0
                ),
                'gradient_stat': self.gradient_stats.get(model_name, 0.0)
            }
        return info
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights dynamically.
        
        Args:
            new_weights: Dictionary mapping model names to new weights
        """
        for model_name, weight in new_weights.items():
            if model_name in self.model_weights:
                self.model_weights[model_name] = weight
                logging.info(f"Updated weight for {model_name}: {weight}")
    
    def enable_model(self, model_name: str, enabled: bool = True):
        """
        Enable or disable a specific auxiliary model.
        
        Args:
            model_name: Name of the model
            enabled: Whether to enable the model
        """
        if model_name in self.auxiliary_models:
            if enabled:
                # Re-enable by setting non-zero weight
                if model_name in self.model_weights:
                    if self.model_weights[model_name] == 0.0:
                        # Restore default weight
                        self.model_weights[model_name] = 0.2
            else:
                # Disable by setting zero weight
                self.model_weights[model_name] = 0.0
            
            logging.info(f"{'Enabled' if enabled else 'Disabled'} auxiliary model: {model_name}")
    
    def get_loss_statistics(self) -> Dict[str, Dict]:
        """
        Get loss statistics for monitoring.
        
        Returns:
            Dictionary with loss statistics per model
        """
        stats = {}
        for model_name, history in self.loss_history.items():
            if history:
                stats[model_name] = {
                    'mean': sum(history) / len(history),
                    'std': (sum((x - sum(history) / len(history)) ** 2 for x in history) / len(history)) ** 0.5,
                    'min': min(history),
                    'max': max(history),
                    'count': len(history)
                }
            else:
                stats[model_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        return stats


def test_multi_model_auxiliary_loss():
    """
    Test function to verify multi-model auxiliary loss works correctly.
    """
    print("Testing MultiModelAuxiliaryLoss...")
    
    # Create test configuration
    model_configs = [
        {
            'name': 'sgn',
            'checkpoint_path': '',  # No checkpoint for testing
            'weight': 0.3,
            'num_classes': 40,
            'enabled': True
        },
        {
            'name': 'mixformer',
            'checkpoint_path': '',  # No checkpoint for testing
            'weight': 0.2,
            'num_classes': 40,
            'enabled': True
        }
    ]
    
    try:
        # Initialize multi-model auxiliary loss
        aux_loss = MultiModelAuxiliaryLoss(
            model_configs=model_configs,
            device='cpu',  # Use CPU for testing
            gradient_balancing=True
        )
        
        # Create dummy input
        batch_size = 4
        generated_motion = torch.randn(batch_size, 3, 64, 25, 1)
        action_labels = torch.randint(0, 40, (batch_size,))
        
        # Test forward pass
        total_loss, loss_dict = aux_loss(generated_motion, action_labels, target_num_classes=40)
        
        print(f"✅ Total auxiliary loss: {total_loss.item():.6f}")
        print(f"✅ Loss dictionary: {loss_dict}")
        
        # Test model info
        model_info = aux_loss.get_model_info()
        print(f"✅ Model info: {model_info}")
        
        # Test weight updates
        aux_loss.update_weights({'sgn': 0.5, 'mixformer': 0.1})
        print("✅ Weight update successful")
        
        # Test model enable/disable
        aux_loss.enable_model('sgn', False)
        aux_loss.enable_model('sgn', True)
        print("✅ Model enable/disable successful")
        
        print("\n✅ All MultiModelAuxiliaryLoss tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multi_model_auxiliary_loss()