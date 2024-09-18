"""
Gradient Balancing Utilities for Multi-Model Training

This module provides advanced gradient balancing strategies to prevent
single model dominance in multi-task learning scenarios.

Key Features:
- Dynamic loss weight adjustment based on gradient magnitudes
- Gradient clipping per auxiliary model
- Gradient statistics logging and monitoring
- Multiple balancing strategies (inverse, adaptive, momentum-based)

Requirements: 3.3, 3.4
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from collections import defaultdict


class GradientBalancer:
    """
    Advanced gradient balancing for multi-model auxiliary losses.
    
    This class implements various strategies to balance gradients from
    multiple auxiliary models to prevent any single model from dominating
    the training process.
    """
    
    def __init__(
        self,
        strategy: str = 'adaptive',
        momentum: float = 0.9,
        clip_value: float = 1.0,
        warmup_steps: int = 100,
        log_frequency: int = 50
    ):
        """
        Initialize gradient balancer.
        
        Args:
            strategy: Balancing strategy ('inverse', 'adaptive', 'momentum', 'none')
            momentum: Momentum coefficient for exponential moving average
            clip_value: Gradient clipping value per model
            warmup_steps: Number of steps before applying balancing
            log_frequency: Frequency of gradient statistics logging
        """
        self.strategy = strategy
        self.momentum = momentum
        self.clip_value = clip_value
        self.warmup_steps = warmup_steps
        self.log_frequency = log_frequency
        
        # Gradient statistics
        self.gradient_stats = {}
        self.gradient_history = defaultdict(list)
        self.step_count = 0
        
        # Balancing weights
        self.balance_weights = {}
        self.weight_history = defaultdict(list)
        
        logging.info(f"Initialized GradientBalancer with strategy: {strategy}")
    
    def compute_gradient_norms(
        self,
        losses: Dict[str, torch.Tensor],
        parameters: List[torch.nn.Parameter]
    ) -> Dict[str, float]:
        """
        Compute gradient norms for each loss.
        
        Args:
            losses: Dictionary of individual model losses
            parameters: List of model parameters to compute gradients for
        
        Returns:
            Dictionary mapping model names to gradient norms
        """
        gradient_norms = {}
        
        for model_name, loss in losses.items():
            if loss.requires_grad and loss.item() > 0:
                try:
                    # Compute gradients
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=parameters,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Compute gradient norm
                    grad_norm = 0.0
                    param_count = 0
                    
                    for grad in grads:
                        if grad is not None:
                            grad_norm += grad.norm().item() ** 2
                            param_count += grad.numel()
                    
                    if param_count > 0:
                        grad_norm = (grad_norm ** 0.5) / (param_count ** 0.5)  # Normalize by param count
                    else:
                        grad_norm = 0.0
                    
                    gradient_norms[model_name] = grad_norm
                    
                except Exception as e:
                    logging.warning(f"Failed to compute gradients for {model_name}: {e}")
                    gradient_norms[model_name] = 0.0
            else:
                gradient_norms[model_name] = 0.0
        
        return gradient_norms
    
    def update_statistics(self, gradient_norms: Dict[str, float]):
        """
        Update gradient statistics with exponential moving average.
        
        Args:
            gradient_norms: Current gradient norms
        """
        self.step_count += 1
        
        for model_name, grad_norm in gradient_norms.items():
            # Update exponential moving average
            if model_name not in self.gradient_stats:
                self.gradient_stats[model_name] = {
                    'mean': grad_norm,
                    'variance': 0.0,
                    'count': 1
                }
            else:
                stats = self.gradient_stats[model_name]
                old_mean = stats['mean']
                
                # Update mean with momentum
                stats['mean'] = self.momentum * old_mean + (1 - self.momentum) * grad_norm
                
                # Update variance
                stats['variance'] = (
                    self.momentum * stats['variance'] + 
                    (1 - self.momentum) * (grad_norm - stats['mean']) ** 2
                )
                
                stats['count'] += 1
            
            # Store history for analysis
            self.gradient_history[model_name].append(grad_norm)
            if len(self.gradient_history[model_name]) > 1000:  # Keep last 1000 values
                self.gradient_history[model_name].pop(0)
    
    def compute_balance_weights(
        self,
        gradient_norms: Dict[str, float],
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute balanced weights based on gradient statistics.
        
        Args:
            gradient_norms: Current gradient norms
            base_weights: Base loss weights
        
        Returns:
            Balanced weights
        """
        if self.step_count < self.warmup_steps or self.strategy == 'none':
            # During warmup or if balancing disabled, use base weights
            return base_weights.copy()
        
        if len(gradient_norms) <= 1:
            # Single model - no balancing needed
            return base_weights.copy()
        
        # Apply balancing strategy
        if self.strategy == 'inverse':
            balance_weights = self._inverse_balancing(gradient_norms, base_weights)
        elif self.strategy == 'adaptive':
            balance_weights = self._adaptive_balancing(gradient_norms, base_weights)
        elif self.strategy == 'momentum':
            balance_weights = self._momentum_balancing(gradient_norms, base_weights)
        else:
            balance_weights = base_weights.copy()
        
        # Store weight history
        for model_name, weight in balance_weights.items():
            self.weight_history[model_name].append(weight)
            if len(self.weight_history[model_name]) > 1000:
                self.weight_history[model_name].pop(0)
        
        return balance_weights
    
    def _inverse_balancing(
        self,
        gradient_norms: Dict[str, float],
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Inverse balancing: models with higher gradients get lower weights.
        """
        # Compute inverse weights
        max_grad = max(gradient_norms.values()) if gradient_norms.values() else 1.0
        
        if max_grad > 0:
            inverse_weights = {}
            for model_name in gradient_norms.keys():
                grad_norm = gradient_norms[model_name]
                # Inverse relationship with smoothing
                inverse_weights[model_name] = max_grad / (grad_norm + 1e-8)
            
            # Normalize to preserve total weight
            total_base = sum(base_weights.values())
            total_inverse = sum(inverse_weights.values())
            
            balanced_weights = {}
            for model_name in base_weights.keys():
                base_weight = base_weights[model_name]
                inverse_weight = inverse_weights.get(model_name, 1.0)
                
                # Combine base weight with inverse balancing
                balanced_weights[model_name] = (
                    base_weight * inverse_weight * total_base / total_inverse
                )
        else:
            balanced_weights = base_weights.copy()
        
        return balanced_weights
    
    def _adaptive_balancing(
        self,
        gradient_norms: Dict[str, float],
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adaptive balancing: adjust weights based on gradient statistics.
        """
        balanced_weights = {}
        
        # Compute target gradient norm (median of all models)
        all_means = [
            self.gradient_stats[name]['mean'] 
            for name in gradient_norms.keys() 
            if name in self.gradient_stats
        ]
        
        if len(all_means) > 0:
            target_grad = np.median(all_means)
        else:
            target_grad = 1.0
        
        for model_name in base_weights.keys():
            base_weight = base_weights[model_name]
            
            if model_name in self.gradient_stats:
                current_mean = self.gradient_stats[model_name]['mean']
                
                if current_mean > 0 and target_grad > 0:
                    # Adjust weight to bring gradient closer to target
                    adjustment = target_grad / current_mean
                    # Smooth the adjustment to prevent oscillations
                    adjustment = np.clip(adjustment, 0.5, 2.0)
                    balanced_weights[model_name] = base_weight * adjustment
                else:
                    balanced_weights[model_name] = base_weight
            else:
                balanced_weights[model_name] = base_weight
        
        return balanced_weights
    
    def _momentum_balancing(
        self,
        gradient_norms: Dict[str, float],
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Momentum-based balancing: smooth weight adjustments over time.
        """
        if not hasattr(self, '_momentum_weights'):
            self._momentum_weights = base_weights.copy()
        
        # Compute target weights using inverse balancing
        target_weights = self._inverse_balancing(gradient_norms, base_weights)
        
        # Apply momentum to smooth weight changes
        balanced_weights = {}
        for model_name in base_weights.keys():
            current_weight = self._momentum_weights.get(model_name, base_weights[model_name])
            target_weight = target_weights.get(model_name, base_weights[model_name])
            
            # Momentum update
            new_weight = (
                self.momentum * current_weight + 
                (1 - self.momentum) * target_weight
            )
            
            balanced_weights[model_name] = new_weight
            self._momentum_weights[model_name] = new_weight
        
        return balanced_weights
    
    def apply_gradient_clipping(
        self,
        losses: Dict[str, torch.Tensor],
        parameters: List[torch.nn.Parameter]
    ):
        """
        Apply per-model gradient clipping.
        
        Args:
            losses: Dictionary of individual model losses
            parameters: List of model parameters
        """
        if self.clip_value <= 0:
            return
        
        for model_name, loss in losses.items():
            if loss.requires_grad and loss.item() > 0:
                try:
                    # Compute gradients for this loss
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=parameters,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Apply clipping
                    valid_grads = [g for g in grads if g is not None]
                    if valid_grads:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=[p for p, g in zip(parameters, grads) if g is not None],
                            max_norm=self.clip_value
                        )
                
                except Exception as e:
                    logging.warning(f"Failed to clip gradients for {model_name}: {e}")
    
    def log_statistics(self):
        """Log gradient and weight statistics."""
        if self.step_count % self.log_frequency == 0 and self.gradient_stats:
            logging.info(f"Gradient Balancing Statistics (Step {self.step_count}):")
            
            for model_name, stats in self.gradient_stats.items():
                mean_grad = stats['mean']
                std_grad = stats['variance'] ** 0.5
                current_weight = self.balance_weights.get(model_name, 0.0)
                
                logging.info(
                    f"  {model_name}: grad_mean={mean_grad:.6f}, "
                    f"grad_std={std_grad:.6f}, weight={current_weight:.4f}"
                )
    
    def get_statistics(self) -> Dict[str, Dict]:
        """
        Get detailed gradient and weight statistics.
        
        Returns:
            Dictionary with statistics for each model
        """
        stats = {}
        
        for model_name in self.gradient_stats.keys():
            grad_stats = self.gradient_stats[model_name]
            grad_history = self.gradient_history.get(model_name, [])
            weight_history = self.weight_history.get(model_name, [])
            
            stats[model_name] = {
                'gradient': {
                    'mean': grad_stats['mean'],
                    'std': grad_stats['variance'] ** 0.5,
                    'count': grad_stats['count'],
                    'recent_values': grad_history[-10:] if grad_history else []
                },
                'weight': {
                    'current': self.balance_weights.get(model_name, 0.0),
                    'recent_values': weight_history[-10:] if weight_history else []
                }
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset all gradient and weight statistics."""
        self.gradient_stats.clear()
        self.gradient_history.clear()
        self.weight_history.clear()
        self.balance_weights.clear()
        self.step_count = 0
        
        if hasattr(self, '_momentum_weights'):
            delattr(self, '_momentum_weights')
        
        logging.info("Reset gradient balancing statistics")


def test_gradient_balancer():
    """
    Test function to verify gradient balancer works correctly.
    """
    print("Testing GradientBalancer...")
    
    # Create test model and balancer
    model = nn.Linear(10, 5)
    balancer = GradientBalancer(strategy='adaptive', momentum=0.9)
    
    # Create dummy losses
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    
    losses = {
        'model1': nn.CrossEntropyLoss()(model(x), y),
        'model2': nn.CrossEntropyLoss()(model(x), y) * 2,  # Higher loss
        'model3': nn.CrossEntropyLoss()(model(x), y) * 0.5  # Lower loss
    }
    
    base_weights = {'model1': 0.3, 'model2': 0.3, 'model3': 0.4}
    
    # Test gradient norm computation
    gradient_norms = balancer.compute_gradient_norms(losses, list(model.parameters()))
    print(f"✅ Gradient norms: {gradient_norms}")
    
    # Test statistics update
    balancer.update_statistics(gradient_norms)
    print("✅ Statistics updated")
    
    # Test weight balancing
    balanced_weights = balancer.compute_balance_weights(gradient_norms, base_weights)
    print(f"✅ Balanced weights: {balanced_weights}")
    
    # Test gradient clipping
    balancer.apply_gradient_clipping(losses, list(model.parameters()))
    print("✅ Gradient clipping applied")
    
    # Test statistics retrieval
    stats = balancer.get_statistics()
    print(f"✅ Statistics: {stats}")
    
    print("\n✅ All GradientBalancer tests passed!")


if __name__ == "__main__":
    test_gradient_balancer()