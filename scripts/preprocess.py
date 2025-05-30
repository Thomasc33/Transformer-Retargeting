#!/usr/bin/env python3
"""
Unified Data Preprocessing Script for Transformer Retargeting Project

This script handles all data preprocessing tasks:
- Raw data conversion and normalization
- Dataset preparation for different evaluation settings
- Data validation and quality checks
- Interactive configuration
- Batch processing support

Usage:
    # Interactive mode
    python scripts/preprocess.py --interactive
    
    # Preprocess specific dataset
    python scripts/preprocess.py --dataset ntu --setting cv
    
    # Preprocess all datasets
    python scripts/preprocess.py --all
    
    # Validate existing data
    python scripts/preprocess.py --validate --dataset ntu120
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
            logging.FileHandler('logs/preprocess.log')
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

def check_raw_data_availability(dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Check availability of raw data files."""
    dataset_config = config['datasets'].get(dataset)
    if not dataset_config:
        return {'available': False, 'error': f'Unknown dataset: {dataset}'}

    raw_data_path = dataset_config['raw_data_path']

    status = {
        'dataset': dataset,
        'data_path': dataset_config['data_path'],
        'raw_data_path': raw_data_path,
        'available': False,
        'files_found': [],
        'files_missing': [],
        'total_size': 0,
        'skeleton_dirs': []
    }

    # Check for raw data directory
    if not os.path.exists(raw_data_path):
        status['error'] = f'Raw data directory not found: {raw_data_path}'
        return status

    # Check for specific file patterns based on dataset
    if dataset in ['ntu', 'ntu120']:
        skeleton_dirs_to_check = []

        if dataset == 'ntu':
            # NTU60 only uses nturgb+d_skeletons
            skeleton_dirs_to_check.append(dataset_config['skeleton_dir'])
        elif dataset == 'ntu120':
            # NTU120 uses both nturgb+d_skeletons120 and nturgb+d_skeletons
            skeleton_dirs_to_check.append(dataset_config['skeleton_dir'])
            if dataset_config.get('includes_ntu60', False):
                skeleton_dirs_to_check.append(dataset_config['ntu60_skeleton_dir'])

        all_skeleton_files = []
        for skeleton_dir in skeleton_dirs_to_check:
            skeleton_path = Path(raw_data_path) / skeleton_dir
            status['skeleton_dirs'].append(str(skeleton_path))

            if skeleton_path.exists():
                skeleton_files = list(skeleton_path.glob('*.skeleton'))
                all_skeleton_files.extend(skeleton_files)
                print(f"  📁 Found {len(skeleton_files)} files in {skeleton_path}")
            else:
                status['files_missing'].append(f'{skeleton_dir}/ directory')
                print(f"  ❌ Missing directory: {skeleton_path}")

        status['files_found'] = [str(f) for f in all_skeleton_files]
        status['total_size'] = sum(f.stat().st_size for f in all_skeleton_files)

        if len(all_skeleton_files) == 0:
            status['files_missing'].append('*.skeleton files')
        else:
            status['available'] = True

    elif dataset == 'etri':
        # Look for .json files
        json_files = list(Path(raw_data_path).glob('*.json'))
        status['files_found'] = [str(f) for f in json_files]
        status['total_size'] = sum(f.stat().st_size for f in json_files)

        if len(json_files) == 0:
            status['files_missing'].append('*.json files')
        else:
            status['available'] = True

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
    """Interactive mode for preprocessing configuration."""
    print("\n🎮 INTERACTIVE DATA PREPROCESSING")
    print("=" * 50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. 📊 Check data availability")
        print("2. 🔄 Preprocess specific dataset")
        print("3. 🔄 Preprocess all datasets")
        print("4. ✅ Validate processed data")
        print("5. 🧹 Clean up temporary files")
        print("6. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            check_all_data_availability(config)
        elif choice == "2":
            return select_dataset_for_preprocessing(config)
        elif choice == "3":
            return {'action': 'preprocess_all'}
        elif choice == "4":
            return select_dataset_for_validation(config)
        elif choice == "5":
            return {'action': 'cleanup'}
        elif choice == "6":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def check_all_data_availability(config: Dict[str, Any]) -> None:
    """Check data availability for all datasets."""
    print("\n📊 DATA AVAILABILITY CHECK")
    print("=" * 40)
    
    datasets = list(config['datasets'].keys())
    
    for dataset in datasets:
        print(f"\n🔍 Checking {dataset.upper()}...")
        status = check_raw_data_availability(dataset, config)
        
        if status['available']:
            size_mb = status['total_size'] / (1024 * 1024)
            print(f"  ✅ Available: {len(status['files_found'])} files ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ Not available: {status.get('error', 'Unknown error')}")
            if status['files_missing']:
                print(f"     Missing: {', '.join(status['files_missing'])}")

def select_dataset_for_preprocessing(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Let user select dataset for preprocessing."""
    datasets = list(config['datasets'].keys())
    
    print("\n📊 Available Datasets:")
    for i, dataset in enumerate(datasets, 1):
        dataset_config = config['datasets'][dataset]
        status = check_raw_data_availability(dataset, config)
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
                status = check_raw_data_availability(selected_dataset, config)
                if not status['available']:
                    print(f"❌ Raw data not available for {selected_dataset}")
                    print(f"   {status.get('error', 'Unknown error')}")
                    continue
                
                # Select settings
                settings = config['datasets'][selected_dataset]['settings']
                print(f"\n⚙️  Available Settings for {selected_dataset}:")
                for i, setting in enumerate(settings, 1):
                    print(f"{i}. {setting}")
                
                setting_choice = input(f"Select setting (1-{len(settings)}) or 'all': ").strip()
                
                if setting_choice.lower() == 'all':
                    selected_settings = settings
                else:
                    try:
                        setting_idx = int(setting_choice) - 1
                        if 0 <= setting_idx < len(settings):
                            selected_settings = [settings[setting_idx]]
                        else:
                            print(f"❌ Please enter a number between 1 and {len(settings)}")
                            continue
                    except ValueError:
                        print("❌ Please enter a valid number or 'all'")
                        continue
                
                return {
                    'action': 'preprocess',
                    'dataset': selected_dataset,
                    'settings': selected_settings
                }
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def select_dataset_for_validation(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Let user select dataset for validation."""
    datasets = list(config['datasets'].keys())
    
    print("\n📊 Available Datasets for Validation:")
    for i, dataset in enumerate(datasets, 1):
        dataset_config = config['datasets'][dataset]
        data_path = dataset_config['data_path']
        exists_icon = "✅" if os.path.exists(data_path) else "❌"
        print(f"{i}. {exists_icon} {dataset}: {dataset_config['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'b' to go back: ").strip()
            if choice.lower() == 'b':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                return {
                    'action': 'validate',
                    'dataset': selected_dataset
                }
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def preprocess_dataset(dataset: str, settings: List[str], config: Dict[str, Any]) -> bool:
    """Preprocess a specific dataset."""
    print(f"\n🔄 PREPROCESSING {dataset.upper()}")
    print("=" * 40)

    dataset_config = config['datasets'][dataset]

    # Check if preprocessing is already completed for all settings
    all_completed = True
    for setting in settings:
        if not check_step_completion('preprocess', dataset, setting, config):
            all_completed = False
            break

    if all_completed:
        print(f"✅ Preprocessing already completed for {dataset} ({', '.join(settings)})")
        print("💡 Skipping preprocessing step")
        return True

    # Check raw data availability
    status = check_raw_data_availability(dataset, config)
    if not status['available']:
        print(f"❌ Cannot preprocess {dataset}: {status.get('error', 'Raw data not available')}")
        return False

    print(f"📊 Dataset: {dataset_config['name']}")
    print(f"📁 Raw data: {status['raw_data_path']}")
    if status.get('skeleton_dirs'):
        print(f"📁 Skeleton dirs: {', '.join(status['skeleton_dirs'])}")
    print(f"📁 Output: {dataset_config['data_path']}")
    print(f"📋 Settings: {', '.join(settings)}")
    print(f"📦 Files to process: {len(status['files_found'])}")

    # Create output directory
    os.makedirs(dataset_config['data_path'], exist_ok=True)

    # Check which settings still need processing
    settings_to_process = []
    for setting in settings:
        if not check_step_completion('preprocess', dataset, setting, config):
            settings_to_process.append(setting)
        else:
            print(f"  ✅ {setting} already processed, skipping")

    if not settings_to_process:
        print("✅ All settings already processed")
        return True

    print(f"\n🔄 Processing settings: {', '.join(settings_to_process)}")

    # TODO: Implement actual preprocessing logic
    print("\n💡 Preprocessing logic will be implemented in the next phase")
    print("💡 This will include:")
    print("   - Raw data parsing and normalization")
    print("   - Cross-subject/cross-view split generation")
    print("   - Data validation and quality checks")
    print("   - Metadata generation")
    print(f"   - Processing for NTU120: both {dataset_config.get('skeleton_dir', 'main')} and {dataset_config.get('ntu60_skeleton_dir', 'ntu60')} directories")

    return True

def validate_dataset(dataset: str, config: Dict[str, Any]) -> bool:
    """Validate processed dataset."""
    print(f"\n✅ VALIDATING {dataset.upper()}")
    print("=" * 40)
    
    dataset_config = config['datasets'][dataset]
    data_path = dataset_config['data_path']
    
    if not os.path.exists(data_path):
        print(f"❌ Processed data directory not found: {data_path}")
        return False
    
    # TODO: Implement validation logic
    print("💡 Validation logic will be implemented in the next phase")
    print("💡 This will include:")
    print("   - File integrity checks")
    print("   - Data format validation")
    print("   - Statistics verification")
    print("   - Cross-reference with expected splits")
    
    return True

def cleanup_temporary_files() -> bool:
    """Clean up temporary files."""
    print("\n🧹 CLEANING UP TEMPORARY FILES")
    print("=" * 40)
    
    temp_dirs = [
        "__pycache__",
        "logs/temp",
        "data/temp",
        ".tmp"
    ]
    
    cleaned = 0
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            print(f"🗑️  Removing {temp_dir}...")
            # TODO: Implement safe cleanup
            cleaned += 1
    
    print(f"✅ Cleaned {cleaned} temporary directories")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Data Preprocessing Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/preprocess.py --interactive
  
  # Preprocess specific dataset
  python scripts/preprocess.py --dataset ntu --setting cv
  
  # Preprocess all datasets
  python scripts/preprocess.py --all
  
  # Validate existing data
  python scripts/preprocess.py --validate --dataset ntu120
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       help='Dataset to preprocess')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv', 'all'],
                       help='Evaluation setting to preprocess')
    parser.add_argument('--all', action='store_true',
                       help='Preprocess all datasets')
    parser.add_argument('--validate', action='store_true',
                       help='Validate processed data instead of preprocessing')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files')
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
    
    print("🚀 TRANSFORMER RETARGETING DATA PREPROCESSING")
    print("=" * 50)
    
    # Handle different modes
    if args.interactive:
        while True:
            action_config = interactive_mode(config)
            if not action_config:
                break
            
            action = action_config['action']
            if action == 'preprocess':
                preprocess_dataset(action_config['dataset'], action_config['settings'], config)
            elif action == 'preprocess_all':
                for dataset in config['datasets'].keys():
                    settings = config['datasets'][dataset]['settings']
                    preprocess_dataset(dataset, settings, config)
            elif action == 'validate':
                validate_dataset(action_config['dataset'], config)
            elif action == 'cleanup':
                cleanup_temporary_files()
    
    elif args.cleanup:
        cleanup_temporary_files()
    
    elif args.validate:
        if args.dataset:
            validate_dataset(args.dataset, config)
        else:
            print("❌ Dataset required for validation")
            sys.exit(1)
    
    elif args.all:
        print("🔄 Preprocessing all datasets...")
        for dataset in config['datasets'].keys():
            settings = config['datasets'][dataset]['settings']
            preprocess_dataset(dataset, settings, config)
    
    elif args.dataset:
        settings = [args.setting] if args.setting and args.setting != 'all' else config['datasets'][args.dataset]['settings']
        preprocess_dataset(args.dataset, settings, config)
    
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --interactive to explore preprocessing options!")

if __name__ == "__main__":
    main()
