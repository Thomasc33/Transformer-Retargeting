#!/usr/bin/env python3
"""
Unified Data Sampling Script for Transformer Retargeting Project

This script handles all data sampling tasks:
- Generate paired data for training
- Create comprehensive datasets with different sample sizes
- Support for different sampling strategies
- Interactive configuration
- Batch processing support

Usage:
    # Interactive mode
    python scripts/sample.py --interactive
    
    # Sample specific dataset
    python scripts/sample.py --dataset ntu --setting cv --train-samples 50000 --test-samples 5000
    
    # Generate comprehensive dataset
    python scripts/sample.py --dataset ntu120 --setting cs --comprehensive
    
    # Sample all datasets
    python scripts/sample.py --all
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/sample.log')
        ]
    )

def load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'python scripts/setup.py' to initialize the project")
        sys.exit(1)

def validate_environment() -> bool:
    """Validate that required files and directories exist."""
    required_paths = [
        "src/data",
        "configs/main_config.yaml"
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("⚠️  Missing required files/directories:")
        for path in missing:
            print(f"   - {path}")
        print("\n💡 Run 'python scripts/setup.py' to initialize the project")
        return False
    
    return True

def check_processed_data_availability(dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Check availability of processed data files."""
    dataset_config = config['datasets'].get(dataset)
    if not dataset_config:
        return {'available': False, 'error': f'Unknown dataset: {dataset}'}
    
    data_path = dataset_config['data_path']
    
    status = {
        'dataset': dataset,
        'data_path': data_path,
        'available': False,
        'files_found': [],
        'total_size': 0
    }
    
    # Check for processed data directory
    if not os.path.exists(data_path):
        status['error'] = f'Processed data directory not found: {data_path}'
        return status
    
    # Look for processed data files
    data_files = list(Path(data_path).glob('*.pkl'))
    if not data_files:
        data_files = list(Path(data_path).glob('*.pt'))
    
    if data_files:
        status['files_found'] = [str(f) for f in data_files]
        status['total_size'] = sum(f.stat().st_size for f in data_files)
        status['available'] = True
    else:
        status['error'] = f'No processed data files found in {data_path}'
    
    return status

def check_step_completion(step: str, dataset: str, setting: str, config: Dict[str, Any]) -> bool:
    """Check if a pipeline step has already been completed."""
    if 'step_completion' not in config or step not in config['step_completion']:
        return False

    step_config = config['step_completion'][step]
    check_files = step_config.get('check_files', [])

    dataset_config = config['datasets'].get(dataset, {})

    # Format file paths with dataset and setting variables
    formatted_files = []
    for file_pattern in check_files:
        try:
            formatted_file = file_pattern.format(
                dataset=dataset,
                setting=setting,
                data_path=dataset_config.get('data_path', ''),
                processed_files=dataset_config.get('processed_files', {})
            )
            formatted_files.append(formatted_file)
        except KeyError:
            # Skip files that can't be formatted (missing variables)
            continue

    # Check if any of the expected files exist
    existing_files = []
    for file_path in formatted_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)

    if existing_files:
        print(f"  ✅ Step '{step}' appears to be completed:")
        for file_path in existing_files:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"     📁 {file_path} ({size_mb:.1f} MB)")
        return True

    return False

def interactive_mode(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interactive mode for sampling configuration."""
    print("\n🎮 INTERACTIVE DATA SAMPLING")
    print("=" * 50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. 📊 Check processed data availability")
        print("2. 🎯 Sample specific dataset")
        print("3. 📦 Generate comprehensive dataset")
        print("4. 🔄 Sample all datasets")
        print("5. ✅ Validate sampled data")
        print("6. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            check_all_processed_data(config)
        elif choice == "2":
            return configure_specific_sampling(config)
        elif choice == "3":
            return configure_comprehensive_sampling(config)
        elif choice == "4":
            return {'action': 'sample_all'}
        elif choice == "5":
            return configure_validation(config)
        elif choice == "6":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def check_all_processed_data(config: Dict[str, Any]) -> None:
    """Check processed data availability for all datasets."""
    print("\n📊 PROCESSED DATA AVAILABILITY CHECK")
    print("=" * 40)
    
    datasets = list(config['datasets'].keys())
    
    for dataset in datasets:
        print(f"\n🔍 Checking {dataset.upper()}...")
        status = check_processed_data_availability(dataset, config)
        
        if status['available']:
            size_mb = status['total_size'] / (1024 * 1024)
            print(f"  ✅ Available: {len(status['files_found'])} files ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ Not available: {status.get('error', 'Unknown error')}")

def configure_specific_sampling(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Configure specific dataset sampling."""
    datasets = list(config['datasets'].keys())
    
    print("\n📊 Available Datasets:")
    for i, dataset in enumerate(datasets, 1):
        dataset_config = config['datasets'][dataset]
        status = check_processed_data_availability(dataset, config)
        status_icon = "✅" if status['available'] else "❌"
        print(f"{i}. {status_icon} {dataset}: {dataset_config['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'b' to go back: ").strip()
            if choice.lower() == 'b':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                
                # Check if data is available
                status = check_processed_data_availability(selected_dataset, config)
                if not status['available']:
                    print(f"❌ Processed data not available for {selected_dataset}")
                    print(f"   {status.get('error', 'Unknown error')}")
                    print("💡 Run preprocessing first: python scripts/preprocess.py --dataset {selected_dataset}")
                    continue
                
                # Select settings
                settings = config['datasets'][selected_dataset]['settings']
                print(f"\n⚙️  Available Settings for {selected_dataset}:")
                for i, setting in enumerate(settings, 1):
                    print(f"{i}. {setting}")
                
                setting_choice = input(f"Select setting (1-{len(settings)}): ").strip()
                try:
                    setting_idx = int(setting_choice) - 1
                    if 0 <= setting_idx < len(settings):
                        selected_setting = settings[setting_idx]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(settings)}")
                        continue
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue
                
                # Configure sample sizes
                print(f"\n🎯 Sampling Configuration:")
                train_samples = int(input("Training samples [50000]: ") or "50000")
                test_samples = int(input("Test samples [5000]: ") or "5000")
                
                # Configure sampling strategy
                print(f"\n📋 Sampling Strategy:")
                print("1. Random sampling")
                print("2. Balanced sampling (equal samples per class)")
                print("3. Stratified sampling (proportional to class distribution)")
                
                strategy_choice = input("Select strategy (1-3) [1]: ").strip() or "1"
                strategies = ['random', 'balanced', 'stratified']
                strategy = strategies[int(strategy_choice) - 1] if strategy_choice in ['1', '2', '3'] else 'random'
                
                return {
                    'action': 'sample_specific',
                    'dataset': selected_dataset,
                    'setting': selected_setting,
                    'train_samples': train_samples,
                    'test_samples': test_samples,
                    'strategy': strategy
                }
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def configure_comprehensive_sampling(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Configure comprehensive dataset sampling."""
    datasets = list(config['datasets'].keys())
    
    print("\n📦 COMPREHENSIVE DATASET GENERATION")
    print("This creates large datasets with multiple sample sizes for extensive experiments")
    
    print("\n📊 Available Datasets:")
    for i, dataset in enumerate(datasets, 1):
        dataset_config = config['datasets'][dataset]
        status = check_processed_data_availability(dataset, config)
        status_icon = "✅" if status['available'] else "❌"
        print(f"{i}. {status_icon} {dataset}: {dataset_config['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'b' to go back: ").strip()
            if choice.lower() == 'b':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                
                # Check if data is available
                status = check_processed_data_availability(selected_dataset, config)
                if not status['available']:
                    print(f"❌ Processed data not available for {selected_dataset}")
                    continue
                
                # Select settings
                settings = config['datasets'][selected_dataset]['settings']
                print(f"\n⚙️  Available Settings for {selected_dataset}:")
                for i, setting in enumerate(settings, 1):
                    print(f"{i}. {setting}")
                
                setting_choice = input(f"Select setting (1-{len(settings)}): ").strip()
                try:
                    setting_idx = int(setting_choice) - 1
                    if 0 <= setting_idx < len(settings):
                        selected_setting = settings[setting_idx]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(settings)}")
                        continue
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue
                
                print(f"\n📦 Comprehensive dataset will include multiple sample sizes:")
                print("   - Small: 10K train, 1K test")
                print("   - Medium: 50K train, 5K test") 
                print("   - Large: 100K train, 10K test")
                print("   - Extra Large: 200K train, 20K test")
                print("   - Full: All available data")
                
                confirm = input("\nProceed with comprehensive sampling? (y/n): ").strip().lower()
                if confirm == 'y':
                    return {
                        'action': 'sample_comprehensive',
                        'dataset': selected_dataset,
                        'setting': selected_setting
                    }
                else:
                    continue
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def configure_validation(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Configure sampled data validation."""
    print("\n✅ SAMPLED DATA VALIDATION")
    
    # Look for existing sampled data files
    sampled_files = []
    for dataset in config['datasets'].keys():
        data_path = Path("data")
        pattern = f"{dataset}_*_paired*.pt"
        files = list(data_path.glob(pattern))
        sampled_files.extend([(dataset, f) for f in files])
    
    if not sampled_files:
        print("❌ No sampled data files found")
        return None
    
    print(f"\n📊 Found {len(sampled_files)} sampled data files:")
    for i, (dataset, file_path) in enumerate(sampled_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {dataset}: {file_path.name} ({size_mb:.1f} MB)")
    
    choice = input(f"\nSelect file to validate (1-{len(sampled_files)}) or 'all': ").strip()
    
    if choice.lower() == 'all':
        return {'action': 'validate_all'}
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sampled_files):
                dataset, file_path = sampled_files[idx]
                return {
                    'action': 'validate_specific',
                    'dataset': dataset,
                    'file_path': str(file_path)
                }
            else:
                print(f"❌ Please enter a number between 1 and {len(sampled_files)}")
                return None
        except ValueError:
            print("❌ Please enter a valid number or 'all'")
            return None

def sample_dataset(dataset: str, setting: str, train_samples: int, test_samples: int,
                  strategy: str, config: Dict[str, Any]) -> bool:
    """Sample a specific dataset."""
    print(f"\n🎯 SAMPLING {dataset.upper()} ({setting})")
    print("=" * 40)

    dataset_config = config['datasets'][dataset]

    # Generate output filename
    output_file = f"data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt"

    # Check if sampling is already completed
    if check_step_completion('sample', dataset, setting, config):
        print(f"✅ Sampling already completed for {dataset} ({setting})")
        print("💡 Skipping sampling step")
        return True

    # Also check for the specific output file
    if os.path.exists(output_file):
        size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"✅ Sampling output already exists: {output_file} ({size_mb:.1f} MB)")
        print("💡 Skipping sampling step")
        return True

    # Check processed data availability
    status = check_processed_data_availability(dataset, config)
    if not status['available']:
        print(f"❌ Cannot sample {dataset}: {status.get('error', 'Processed data not available')}")
        print("💡 Run preprocessing first: python scripts/preprocess.py --dataset {dataset}")
        return False

    print(f"📊 Dataset: {dataset_config['name']}")
    print(f"📁 Data path: {status['data_path']}")
    print(f"⚙️  Setting: {setting}")
    print(f"🎯 Strategy: {strategy}")
    print(f"📈 Train samples: {train_samples:,}")
    print(f"📉 Test samples: {test_samples:,}")
    print(f"💾 Output: {output_file}")

    # TODO: Implement actual sampling logic
    print("\n💡 Sampling logic will be implemented in the next phase")
    print("💡 This will include:")
    print("   - Loading processed data")
    print("   - Applying sampling strategy")
    print("   - Generating paired samples")
    print("   - Cross-validation splits")
    print("   - Data augmentation")
    print("   - Quality validation")

    return True

def sample_comprehensive_dataset(dataset: str, setting: str, config: Dict[str, Any]) -> bool:
    """Generate comprehensive dataset with multiple sample sizes."""
    print(f"\n📦 COMPREHENSIVE SAMPLING {dataset.upper()} ({setting})")
    print("=" * 50)
    
    sample_configs = [
        (10000, 1000, "small"),
        (50000, 5000, "medium"),
        (100000, 10000, "large"),
        (200000, 20000, "extra_large"),
        (999999, 99999, "comprehensive")  # Use all available data
    ]
    
    for train_samples, test_samples, size_name in sample_configs:
        print(f"\n🎯 Generating {size_name} dataset...")
        success = sample_dataset(dataset, setting, train_samples, test_samples, 'stratified', config)
        if not success:
            print(f"❌ Failed to generate {size_name} dataset")
            return False
    
    print(f"\n✅ Comprehensive dataset generation completed for {dataset} ({setting})")
    return True

def validate_sampled_data(file_path: str) -> bool:
    """Validate a sampled data file."""
    print(f"\n✅ VALIDATING {Path(file_path).name}")
    print("=" * 40)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    print(f"📁 File: {file_path}")
    print(f"📏 Size: {size_mb:.1f} MB")
    
    # TODO: Implement validation logic
    print("\n💡 Validation logic will be implemented in the next phase")
    print("💡 This will include:")
    print("   - Data integrity checks")
    print("   - Sample count verification")
    print("   - Pair consistency validation")
    print("   - Statistical analysis")
    print("   - Memory usage estimation")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Data Sampling Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/sample.py --interactive
  
  # Sample specific dataset
  python scripts/sample.py --dataset ntu --setting cv --train-samples 50000 --test-samples 5000
  
  # Generate comprehensive dataset
  python scripts/sample.py --dataset ntu120 --setting cs --comprehensive
  
  # Validate sampled data
  python scripts/sample.py --validate --file data/ntu_cv_paired_50000_5000.pt
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       help='Dataset to sample')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'],
                       help='Evaluation setting')
    parser.add_argument('--train-samples', type=int, default=50000,
                       help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=5000,
                       help='Number of test samples')
    parser.add_argument('--strategy', type=str, choices=['random', 'balanced', 'stratified'],
                       default='stratified', help='Sampling strategy')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive dataset with multiple sample sizes')
    parser.add_argument('--all', action='store_true',
                       help='Sample all datasets')
    parser.add_argument('--validate', action='store_true',
                       help='Validate sampled data')
    parser.add_argument('--file', type=str,
                       help='Specific file to validate')
    parser.add_argument('--config', type=str, default='configs/main_config.yaml',
                       help='Path to main configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    print("🚀 TRANSFORMER RETARGETING DATA SAMPLING")
    print("=" * 50)
    
    # Handle different modes
    if args.interactive:
        while True:
            action_config = interactive_mode(config)
            if not action_config:
                break
            
            action = action_config['action']
            if action == 'sample_specific':
                sample_dataset(
                    action_config['dataset'], 
                    action_config['setting'],
                    action_config['train_samples'],
                    action_config['test_samples'],
                    action_config['strategy'],
                    config
                )
            elif action == 'sample_comprehensive':
                sample_comprehensive_dataset(
                    action_config['dataset'],
                    action_config['setting'],
                    config
                )
            elif action == 'sample_all':
                for dataset in config['datasets'].keys():
                    for setting in config['datasets'][dataset]['settings']:
                        sample_dataset(dataset, setting, 50000, 5000, 'stratified', config)
            elif action == 'validate_specific':
                validate_sampled_data(action_config['file_path'])
            elif action == 'validate_all':
                # Find and validate all sampled files
                for dataset in config['datasets'].keys():
                    data_path = Path("data")
                    pattern = f"{dataset}_*_paired*.pt"
                    files = list(data_path.glob(pattern))
                    for file_path in files:
                        validate_sampled_data(str(file_path))
    
    elif args.validate:
        if args.file:
            validate_sampled_data(args.file)
        else:
            print("❌ File path required for validation")
            sys.exit(1)
    
    elif args.comprehensive:
        if args.dataset and args.setting:
            sample_comprehensive_dataset(args.dataset, args.setting, config)
        else:
            print("❌ Dataset and setting required for comprehensive sampling")
            sys.exit(1)
    
    elif args.all:
        print("🔄 Sampling all datasets...")
        for dataset in config['datasets'].keys():
            for setting in config['datasets'][dataset]['settings']:
                sample_dataset(dataset, setting, args.train_samples, args.test_samples, args.strategy, config)
    
    elif args.dataset and args.setting:
        sample_dataset(args.dataset, args.setting, args.train_samples, args.test_samples, args.strategy, config)
    
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --interactive to explore sampling options!")

if __name__ == "__main__":
    main()
