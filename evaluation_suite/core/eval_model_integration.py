"""
Integration layer to use existing eval_model.py functionality within the evaluation suite.
This ensures we use your tested and working evaluation code.
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class EvalModelIntegration:
    """Integration layer for eval_model.py functionality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = project_root
        
    def run_eval_model(self, 
                      model_type: str,
                      model_path: str,
                      eval_model_type: str,
                      eval_model_path: str,
                      dataset: str,
                      setting: str,
                      task: str,
                      output_dir: str,
                      batch_size: int = 32,
                      test_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run eval_model.py with specified parameters and return results.
        
        Args:
            model_type: Type of anonymization model ('raw', 'transformer', 'dmr', 'pmr')
            model_path: Path to anonymization model weights
            eval_model_type: Type of evaluation model ('sgn', 'mixformer')
            eval_model_path: Path to evaluation model weights
            dataset: Dataset name ('ntu', 'ntu120', 'etri')
            setting: Dataset setting ('cv', 'cs')
            task: Evaluation task ('ar', 'ri', 'gc')
            output_dir: Directory to save results
            batch_size: Batch size for evaluation
            test_samples: Number of test samples (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Prepare command arguments
            cmd = [
                'python', 'eval_model.py',
                '--model_type', model_type,
                '--eval_model_type', eval_model_type,
                '--eval_model_path', eval_model_path,
                '--dataset', dataset,
                '--setting', setting,
                '--task', task,
                '--batch_size', str(batch_size),
                '--output_dir', output_dir
            ]
            
            # Add model path if not raw
            if model_type != 'raw':
                cmd.extend(['--model_path', model_path])
            
            # Add test samples if specified
            if test_samples:
                cmd.extend(['--test_samples', str(test_samples)])
            
            self.logger.info(f"Running eval_model.py with command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("eval_model.py completed successfully")
                
                # Try to parse results from output directory
                results = self._parse_results(output_dir)
                return results
            else:
                self.logger.error(f"eval_model.py failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return {'error': f"eval_model.py failed: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            self.logger.error("eval_model.py timed out")
            return {'error': 'eval_model.py timed out'}
        except Exception as e:
            self.logger.error(f"Error running eval_model.py: {str(e)}")
            return {'error': f"Error running eval_model.py: {str(e)}"}
    
    def _parse_results(self, output_dir: str) -> Dict[str, Any]:
        """Parse results from eval_model.py output directory."""
        results = {}
        
        try:
            output_path = Path(output_dir)
            
            # Look for common result files
            result_files = [
                'results.json',
                'metrics.json',
                'evaluation_results.json',
                'summary.json'
            ]
            
            for result_file in result_files:
                file_path = output_path / result_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        file_results = json.load(f)
                        results.update(file_results)
            
            # If no JSON files found, look for CSV files
            if not results:
                csv_files = list(output_path.glob('*.csv'))
                if csv_files:
                    results['csv_files'] = [str(f) for f in csv_files]
            
            # If still no results, just indicate completion
            if not results:
                results = {'status': 'completed', 'output_dir': output_dir}
                
        except Exception as e:
            self.logger.warning(f"Could not parse results from {output_dir}: {str(e)}")
            results = {'status': 'completed_with_parse_error', 'output_dir': output_dir}
        
        return results
    
    def run_physical_metrics(self,
                           model_type: str,
                           model_path: str,
                           dataset: str,
                           setting: str,
                           output_dir: str,
                           batch_size: int = 32,
                           test_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run physical plausibility metrics evaluation.
        
        Args:
            model_type: Type of anonymization model
            model_path: Path to model weights
            dataset: Dataset name
            setting: Dataset setting
            output_dir: Directory to save results
            batch_size: Batch size
            test_samples: Number of test samples
            
        Returns:
            Dictionary containing physical metrics results
        """
        try:
            # Prepare command for physical metrics
            cmd = [
                'python', 'eval_model.py',
                '--model_type', model_type,
                '--dataset', dataset,
                '--setting', setting,
                '--batch_size', str(batch_size),
                '--output_dir', output_dir,
                '--physical_metrics_only'  # Assuming this flag exists
            ]
            
            # Add model path if not raw
            if model_type != 'raw':
                cmd.extend(['--model_path', model_path])
            
            # Add test samples if specified
            if test_samples:
                cmd.extend(['--test_samples', str(test_samples)])
            
            self.logger.info(f"Running physical metrics with command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Physical metrics evaluation completed successfully")
                results = self._parse_results(output_dir)
                return results
            else:
                self.logger.error(f"Physical metrics evaluation failed: {result.stderr}")
                return {'error': f"Physical metrics failed: {result.stderr}"}
                
        except Exception as e:
            self.logger.error(f"Error running physical metrics: {str(e)}")
            return {'error': f"Error running physical metrics: {str(e)}"}
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available model files."""
        models = {
            'transformer': [],
            'dmr': [],
            'pmr': [],
            'sgn': [],
            'mixformer': []
        }
        
        # Check common model directories
        model_dirs = [
            self.project_root / 'output',
            self.project_root / 'trained_models',
            self.project_root / 'models'
        ]
        
        for model_dir in model_dirs:
            if model_dir.exists():
                # Look for model files
                for model_file in model_dir.rglob('*.pth*'):
                    file_name = model_file.name.lower()
                    if 'transformer' in file_name or 'autoencoder' in file_name:
                        models['transformer'].append(str(model_file))
                    elif 'dmr' in file_name:
                        models['dmr'].append(str(model_file))
                    elif 'pmr' in file_name:
                        models['pmr'].append(str(model_file))
                    elif 'sgn' in file_name:
                        models['sgn'].append(str(model_file))
                    elif 'mixformer' in file_name:
                        models['mixformer'].append(str(model_file))
        
        return models
    
    def validate_model_path(self, model_path: str) -> bool:
        """Validate that a model path exists and is accessible."""
        if model_path == 'raw':
            return True
        
        path = Path(model_path)
        if not path.exists():
            self.logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        if not path.is_file():
            self.logger.warning(f"Model path is not a file: {model_path}")
            return False
        
        return True
