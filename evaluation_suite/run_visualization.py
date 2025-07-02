#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Runner for the Evaluation Suite

This script runs visualization experiments from the command line,
integrating with the pipeline system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation_suite.core.visualization_evaluator import VisualizationEvaluator
from evaluation_suite.experiments.visualization import VisualizationExperiments


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/visualization.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run visualization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run skeleton animations for NTU CV
  python evaluation_suite/run_visualization.py --visualizations skeleton_animations --dataset ntu --setting cv

  # Run MLM pretraining visualization
  python evaluation_suite/run_visualization.py --visualizations mlm_pretraining --dataset ntu --setting cv --temporal-ratio 0.3 --spatial-ratio 0.3

  # Run all visualizations
  python evaluation_suite/run_visualization.py --visualizations all --dataset ntu --setting cv

  # Run multiple specific visualizations
  python evaluation_suite/run_visualization.py --visualizations skeleton_animations,comparison_visualizations --dataset ntu --setting cv
        """
    )
    
    parser.add_argument('--visualizations', type=str, required=True,
                       help='Comma-separated list of visualizations to run (or "all")')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       required=True, help='Dataset to use')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'],
                       required=True, help='Evaluation setting')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to visualize')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames per animation (None for full length)')
    parser.add_argument('--windows', action='store_true',
                       help='Windows compatibility mode')
    parser.add_argument('--temporal-ratio', type=float, default=0.3,
                       help='Temporal masking ratio for MLM visualization (0.0-1.0)')
    parser.add_argument('--spatial-ratio', type=float, default=0.3,
                       help='Spatial masking ratio for MLM visualization (0.0-1.0)')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='Base directory for MLM model weights')

    return parser.parse_args()


def get_visualization_configs(visualizations: List[str], dataset: str, setting: str,
                            max_samples: int, max_frames: Optional[int], temporal_ratio: float = 0.3,
                            spatial_ratio: float = 0.3, base_dir: str = '.') -> Dict[str, Any]:
    """Get visualization configurations based on selected visualizations."""
    all_configs = VisualizationExperiments.get_experiment_configs()
    
    if 'all' in visualizations:
        selected_configs = all_configs
    else:
        selected_configs = {name: config for name, config in all_configs.items() 
                          if name in visualizations}
    
    # Update configurations with dataset, setting, and limits
    for config_name, config in selected_configs.items():
        # Update data configurations
        for data_name, data_config in config.get('data', {}).items():
            data_config['dataset'] = dataset
            data_config['setting'] = setting
            data_config['test_samples'] = min(data_config.get('test_samples', 10), max_samples)

            # Update MLM-specific parameters
            if config.get('evaluation_type') == 'mlm_pretraining':
                data_config['temporal_masking_ratio'] = temporal_ratio
                data_config['spatial_masking_ratio'] = spatial_ratio

        # Update model configurations for MLM
        if config.get('evaluation_type') == 'mlm_pretraining':
            models_config = config.get('models', {})
            if 'mlm_autoencoder' in models_config:
                models_config['mlm_autoencoder']['base_dir'] = base_dir
                models_config['mlm_autoencoder']['temporal_ratio'] = temporal_ratio
                models_config['mlm_autoencoder']['spatial_ratio'] = spatial_ratio

        # Update quality settings for animations
        if 'quality_settings' in config:
            if max_frames is not None:
                config['quality_settings']['max_frames'] = max_frames
            else:
                # Remove max_frames to allow full length
                config['quality_settings'].pop('max_frames', None)
        elif config.get('evaluation_type') == 'skeleton_animation' and max_frames is not None:
            config['quality_settings'] = {'max_frames': max_frames}
    
    return selected_configs


def run_visualizations(args):
    """Run the selected visualizations."""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting visualization runner")
    logger.info(f"Dataset: {args.dataset}, Setting: {args.setting}")
    logger.info(f"Visualizations: {args.visualizations}")
    
    # Parse visualization list
    if args.visualizations.lower() == 'all':
        visualizations = ['all']
    else:
        visualizations = [v.strip() for v in args.visualizations.split(',')]
    
    logger.info(f"Selected visualizations: {visualizations}")
    
    # Get visualization configurations
    configs = get_visualization_configs(
        visualizations, args.dataset, args.setting,
        args.max_samples, args.max_frames, args.temporal_ratio,
        args.spatial_ratio, args.base_dir
    )
    
    if not configs:
        logger.error("No valid visualization configurations found")
        return False
    
    # Initialize evaluator
    evaluator = VisualizationEvaluator(
        device=args.device,
        output_base_dir=args.output_dir
    )
    
    # Run visualizations
    all_results = {}
    success_count = 0
    total_count = len(configs)
    
    for config_name, config in configs.items():
        logger.info(f"Running visualization: {config_name}")
        
        try:
            results = evaluator.run_visualization_experiment(config)
            all_results[config_name] = results
            
            if results.get('success', False):
                success_count += 1
                logger.info(f"✅ Completed visualization: {config_name}")
                
                # Log output paths
                if 'visualizations' in results:
                    for model_name, model_results in results['visualizations'].items():
                        if isinstance(model_results, dict):
                            for viz_type, viz_path in model_results.items():
                                logger.info(f"   📁 {model_name}/{viz_type}: {viz_path}")
                        else:
                            logger.info(f"   📁 {model_name}: {model_results}")
            else:
                logger.error(f"❌ Failed visualization: {config_name}")
                if 'error' in results:
                    logger.error(f"   Error: {results['error']}")
                    
        except Exception as e:
            logger.error(f"❌ Exception in visualization {config_name}: {e}")
            all_results[config_name] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info(f"\n🎨 VISUALIZATION SUMMARY")
    logger.info(f"=" * 50)
    logger.info(f"✅ Successful: {success_count}/{total_count}")
    logger.info(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count > 0:
        logger.info(f"\n📁 Output directory: {args.output_dir}")
        logger.info(f"💡 Check the output directory for generated visualizations")
    
    return success_count == total_count


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🎨 VISUALIZATION RUNNER")
    print("=" * 50)
    print(f"📊 Dataset: {args.dataset}")
    print(f"⚙️  Setting: {args.setting}")
    print(f"🎬 Visualizations: {args.visualizations}")
    print(f"📁 Output: {args.output_dir}")
    
    try:
        success = run_visualizations(args)
        
        if success:
            print("\n🎉 ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("Check the output directory for generated files.")
            sys.exit(0)
        else:
            print("\n💥 SOME VISUALIZATIONS FAILED")
            print("Check the logs for detailed error information.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Visualization runner interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
