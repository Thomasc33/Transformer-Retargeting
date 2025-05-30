#!/usr/bin/env python3
"""
Setup Script for Transformer Retargeting Project

This script initializes the project environment:
- Creates necessary directories
- Validates dependencies
- Sets up configuration files
- Checks data availability
- Provides setup guidance

Usage:
    python scripts/setup.py
    python scripts/setup.py --check-only
    python scripts/setup.py --install-deps
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("💡 Please upgrade to Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = {
        'torch': 'PyTorch for deep learning',
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing',
        'matplotlib': 'Plotting and visualization',
        'seaborn': 'Statistical visualization',
        'pandas': 'Data manipulation',
        'yaml': 'YAML configuration files',
        'tqdm': 'Progress bars',
        'sklearn': 'Machine learning utilities'
    }
    
    missing = []
    installed = []
    
    for package, description in required_packages.items():
        try:
            if package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            installed.append(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}: {description}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required packages")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    print(f"\n✅ All {len(installed)} required packages are installed")
    return True

def create_directories():
    """Create necessary project directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "src/model", "src/data", "src/training", "src/evaluation", "src/utils",
        "scripts/generated", "docs", "configs", "logs", "results", "checkpoints",
        "data/temp", "evaluation_suite/results"
    ]
    
    created = []
    existing = []
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created.append(directory)
            print(f"  ✅ Created: {directory}")
        else:
            existing.append(directory)
            print(f"  📁 Exists: {directory}")
    
    if created:
        print(f"\n✅ Created {len(created)} new directories")
    if existing:
        print(f"📁 Found {len(existing)} existing directories")
    
    return True

def setup_configuration():
    """Set up configuration files."""
    print("\n⚙️  Setting up configuration files...")
    
    config_file = "configs/main_config.yaml"
    
    if os.path.exists(config_file):
        print(f"  📁 Configuration file already exists: {config_file}")
        return True
    
    # The main config was already created in the consolidation process
    if os.path.exists(config_file):
        print(f"  ✅ Configuration file ready: {config_file}")
    else:
        print(f"  ⚠️  Configuration file missing: {config_file}")
        print("  💡 This should have been created during consolidation")
        return False
    
    return True

def check_data_availability():
    """Check for available datasets."""
    print("\n📊 Checking data availability...")
    
    datasets = {
        'ntu': 'data/nturgbd_raw/nturgb+d_skeletons',
        'ntu120': 'data/nturgbd_raw/nturgb+d_skeletons120', 
        'etri': 'data/etri_raw'
    }
    
    available = []
    missing = []
    
    for dataset, path in datasets.items():
        if os.path.exists(path):
            files = list(Path(path).glob('*'))
            if files:
                available.append(dataset)
                print(f"  ✅ {dataset.upper()}: {len(files)} files in {path}")
            else:
                missing.append(dataset)
                print(f"  📁 {dataset.upper()}: Directory exists but empty: {path}")
        else:
            missing.append(dataset)
            print(f"  ❌ {dataset.upper()}: Not found: {path}")
    
    if available:
        print(f"\n✅ Found data for: {', '.join(available)}")
    
    if missing:
        print(f"\n⚠️  Missing data for: {', '.join(missing)}")
        print("💡 Download datasets and place in the respective directories")
        print("💡 Or run preprocessing to convert existing data")
    
    return len(available) > 0

def check_gpu_availability():
    """Check if GPU is available for training."""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"  🎮 Primary GPU: {gpu_name}")
            
            # Check memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  💾 GPU Memory: {memory_gb:.1f} GB")
            
            if memory_gb < 8:
                print("  ⚠️  GPU memory is low (<8GB). Consider using smaller batch sizes.")
            
            return True
        else:
            print("  ❌ CUDA not available")
            print("  💡 Training will use CPU (much slower)")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def install_dependencies():
    """Install missing dependencies."""
    print("\n📦 Installing dependencies...")
    
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas',
        'pyyaml', 'tqdm', 'scikit-learn', 'wandb'
    ]
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        print(f"💻 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def print_next_steps():
    """Print guidance for next steps."""
    print("\n🚀 SETUP COMPLETE!")
    print("=" * 50)
    print("\n📋 Next Steps:")
    print("1. 📊 Prepare your data:")
    print("   - Download NTU RGB+D, NTU RGB+D 120, or ETRI datasets")
    print("   - Place raw data in data/[dataset]_raw/ directories")
    print("   - Run: python scripts/preprocess.py --interactive")
    
    print("\n2. 🎯 Quick Start:")
    print("   - Run: python scripts/pipeline.py --interactive")
    print("   - Or: python scripts/pipeline.py --quick-start --dataset ntu --setting cv")
    
    print("\n3. 🧪 Explore Individual Components:")
    print("   - Data sampling: python scripts/sample.py --interactive")
    print("   - Pretraining: python scripts/pretrain.py --interactive")
    print("   - Training: python scripts/train.py --interactive")
    print("   - Evaluation: python scripts/evaluate.py --interactive")
    
    print("\n4. 📚 Documentation:")
    print("   - Check docs/ directory for detailed guides")
    print("   - Review configs/main_config.yaml for settings")
    
    print("\n💡 Tips:")
    print("   - Use --interactive flags for guided setup")
    print("   - Use --slurm flags for HPC job generation")
    print("   - Check logs/ directory for detailed output")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup Script for Transformer Retargeting Project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--check-only', action='store_true',
                       help='Only check environment, do not install or create anything')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install missing Python dependencies')
    
    args = parser.parse_args()
    
    print("🚀 TRANSFORMER RETARGETING PROJECT SETUP")
    print("=" * 50)
    print("🎯 Initializing your development environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok and args.install_deps:
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            sys.exit(1)
        deps_ok = True
    
    if args.check_only:
        print("\n🔍 CHECK-ONLY MODE")
        check_gpu_availability()
        check_data_availability()
        
        if deps_ok:
            print("\n✅ Environment check passed")
        else:
            print("\n⚠️  Environment check found issues")
            print("💡 Run without --check-only to fix issues")
        
        sys.exit(0 if deps_ok else 1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Failed to create directories")
        sys.exit(1)
    
    # Setup configuration
    if not setup_configuration():
        print("\n❌ Failed to setup configuration")
        sys.exit(1)
    
    # Check additional components
    check_gpu_availability()
    check_data_availability()
    
    # Print next steps
    print_next_steps()
    
    print("\n🎉 SETUP SUCCESSFUL!")
    print("Your Transformer Retargeting environment is ready!")

if __name__ == "__main__":
    main()
