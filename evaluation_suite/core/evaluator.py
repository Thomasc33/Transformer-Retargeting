"""
Comprehensive Evaluator - Main evaluation engine for the Transformer Retargeting project.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

from .metrics import MetricsCalculator
from .models import ModelManager
from .data_loader import DataManager
from .eval_model_integration import EvalModelIntegration


class ComprehensiveEvaluator:
    """
    Main evaluation engine that coordinates all evaluation tasks.
    """

    def __init__(self, config: Dict[str, Any], results_dir: str = "evaluation_suite/results"):
        """
        Initialize the comprehensive evaluator.

        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        self.eval_integration = EvalModelIntegration()

        # Setup logging
        self.setup_logging()

        # Track experiment state
        self.experiment_id = self.generate_experiment_id()
        self.results = {}

    def setup_logging(self):
        """Setup logging for the evaluation."""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"exp_{timestamp}"

    def run_experiment(self, experiment_name: str, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment_name: Name of the experiment
            experiment_config: Configuration for the experiment

        Returns:
            Dictionary containing experiment results
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        start_time = time.time()

        try:
            # Create experiment directory
            exp_dir = self.results_dir / "experiments" / experiment_name / self.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Load models and data
            models = self.model_manager.load_models(experiment_config.get('models', {}))
            data_loaders = self.data_manager.load_data(experiment_config.get('data', {}))

            # Run evaluation
            results = self.evaluate_models(models, data_loaders, experiment_config)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(results, experiment_config)

            # Save results
            experiment_results = {
                'experiment_name': experiment_name,
                'experiment_id': self.experiment_id,
                'config': experiment_config,
                'results': results,
                'metrics': metrics,
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            self.save_experiment_results(experiment_results, exp_dir)

            self.logger.info(f"Completed experiment: {experiment_name} in {time.time() - start_time:.2f}s")
            return experiment_results

        except Exception as e:
            self.logger.error(f"Error in experiment {experiment_name}: {str(e)}")
            raise

    def evaluate_models(self, models: Dict[str, Any], data_loaders: Dict[str, Any],
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate models on the given data.

        Args:
            models: Dictionary of loaded models
            data_loaders: Dictionary of data loaders
            config: Experiment configuration

        Returns:
            Raw evaluation results
        """
        results = {}

        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            model_results = {}

            for data_name, data_loader in data_loaders.items():
                self.logger.info(f"  On dataset: {data_name}")

                # Run model evaluation
                if config.get('evaluation_type') == 'privacy_utility':
                    model_results[data_name] = self.evaluate_privacy_utility(model, data_loader, config)
                elif config.get('evaluation_type') == 'ablation':
                    model_results[data_name] = self.evaluate_ablation(model, data_loader, config)
                elif config.get('evaluation_type') == 'robustness':
                    model_results[data_name] = self.evaluate_robustness(model, data_loader, config)
                else:
                    model_results[data_name] = self.evaluate_standard(model, data_loader, config)

            results[model_name] = model_results

        return results

    def evaluate_privacy_utility(self, model: Any, data_loader: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate privacy vs utility tradeoff using existing eval_model.py."""
        results = {}

        # Get model and eval model configurations
        models_config = config.get('models', {})
        eval_models_config = config.get('eval_models', {})
        data_config = config.get('data', {})

        # Run evaluation for each model and eval model combination
        for model_name, model_cfg in models_config.items():
            for eval_model_name, eval_cfg in eval_models_config.items():
                for data_name, data_cfg in data_config.items():

                    # Create output directory for this combination
                    output_dir = f"temp_results/{model_name}_{eval_model_name}_{data_name}"

                    # Run eval_model.py
                    result = self.eval_integration.run_eval_model(
                        model_type=model_cfg.get('type', 'raw'),
                        model_path=model_cfg.get('path', ''),
                        eval_model_type=eval_cfg.get('type', 'sgn'),
                        eval_model_path=eval_cfg.get('path', ''),
                        dataset=data_cfg.get('dataset', 'ntu'),
                        setting=data_cfg.get('setting', 'cv'),
                        task=eval_cfg.get('task', 'ar'),
                        output_dir=output_dir,
                        batch_size=data_cfg.get('batch_size', 32),
                        test_samples=data_cfg.get('test_samples')
                    )

                    results[f"{model_name}_{eval_model_name}_{data_name}"] = result

        return results

    def evaluate_ablation(self, model: Any, data_loader: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ablation studies using existing eval_model.py."""
        # Similar to privacy_utility but for ablation studies
        return self.evaluate_privacy_utility(model, data_loader, config)

    def evaluate_robustness(self, model: Any, data_loader: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate robustness analysis using existing eval_model.py."""
        # Similar to privacy_utility but for robustness analysis
        return self.evaluate_privacy_utility(model, data_loader, config)

    def evaluate_standard(self, model: Any, data_loader: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Standard evaluation using existing eval_model.py."""
        # Use the same approach as privacy_utility for standard evaluation
        return self.evaluate_privacy_utility(model, data_loader, config)

    def save_experiment_results(self, results: Dict[str, Any], exp_dir: Path):
        """Save experiment results to disk."""
        # Save main results
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save config separately for easy access
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(results['config'], f, indent=2)

        # Save metrics separately
        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)

        self.logger.info(f"Results saved to {exp_dir}")

    def run_experiment_suite(self, experiment_suite: List[str]) -> Dict[str, Any]:
        """
        Run a suite of experiments.

        Args:
            experiment_suite: List of experiment names to run

        Returns:
            Combined results from all experiments
        """
        suite_results = {}

        for experiment_name in experiment_suite:
            if experiment_name in self.config.get('experiments', {}):
                experiment_config = self.config['experiments'][experiment_name]
                suite_results[experiment_name] = self.run_experiment(experiment_name, experiment_config)
            else:
                self.logger.warning(f"Experiment {experiment_name} not found in config")

        return suite_results

    def get_experiment_status(self) -> Dict[str, Any]:
        """Get status of all experiments."""
        status = {
            'experiment_id': self.experiment_id,
            'results_dir': str(self.results_dir),
            'completed_experiments': list(self.results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        return status
