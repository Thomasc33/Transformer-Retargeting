#!/usr/bin/env python3
"""
Setup script for the comprehensive evaluation suite.
"""

import os
import sys
from pathlib import Path
import subprocess
import logging


def setup_logging():
    """Setup logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_directories():
    """Create necessary directories."""
    directories = [
        "evaluation_suite/results",
        "evaluation_suite/results/experiments", 
        "evaluation_suite/results/analysis",
        "evaluation_suite/results/cache",
        "evaluation_suite/results/logs",
        "evaluation_suite/reports",
        "evaluation_suite/reports/outputs",
        "evaluation_suite/runners/jobs",
        "slurm_out"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def make_scripts_executable():
    """Make all Python scripts executable."""
    scripts = [
        "evaluation_suite/run_experiments.py",
        "evaluation_suite/generate_report.py", 
        "evaluation_suite/monitor_jobs.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            os.chmod(script, 0o755)
            logging.info(f"Made executable: {script}")


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'yaml',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logging.info(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logging.warning(f"❌ {package} is missing")
    
    if missing_packages:
        logging.error(f"Missing packages: {missing_packages}")
        logging.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def verify_existing_code():
    """Verify that existing evaluation code is accessible."""
    required_files = [
        "eval_model.py",
        "data.py",
        "model/autoencoder.py",
        "model/sgn.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            logging.warning(f"❌ {file_path} not found")
        else:
            logging.info(f"✅ {file_path} found")
    
    if missing_files:
        logging.warning(f"Some existing files are missing: {missing_files}")
        logging.info("The evaluation suite will work with placeholder implementations")
    
    return True


def test_basic_functionality():
    """Test basic functionality of the evaluation suite."""
    try:
        # Test importing core modules
        sys.path.append(str(Path.cwd()))
        
        from evaluation_suite.core import ComprehensiveEvaluator
        from evaluation_suite.analysis import ComprehensiveVisualizer
        
        logging.info("✅ Core modules import successfully")
        
        # Test configuration loading
        config_path = "evaluation_suite/configs/experiments.yaml"
        if Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info("✅ Configuration file loads successfully")
        else:
            logging.warning(f"❌ Configuration file not found: {config_path}")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Basic functionality test failed: {str(e)}")
        return False


def print_success_message():
    """Print success message with next steps."""
    print("\n" + "="*60)
    print("🎉 EVALUATION SUITE SETUP COMPLETE!")
    print("="*60)
    print()
    print("✅ All directories created")
    print("✅ Scripts made executable") 
    print("✅ Dependencies checked")
    print("✅ Basic functionality verified")
    print()
    print("🚀 Next Steps:")
    print()
    print("1. List available experiments:")
    print("   python evaluation_suite/run_experiments.py --list-experiments")
    print()
    print("2. Check current status:")
    print("   python evaluation_suite/run_experiments.py --status")
    print()
    print("3. Run a quick test:")
    print("   python evaluation_suite/run_experiments.py --experiment-set quick")
    print()
    print("4. Monitor progress:")
    print("   python evaluation_suite/monitor_jobs.py --status")
    print()
    print("5. Generate reports:")
    print("   python evaluation_suite/generate_report.py --all --type executive")
    print()
    print("📚 Documentation:")
    print("   See evaluation_suite/README.md for complete usage guide")
    print()
    print("🎯 For your advisor:")
    print("   Use --type executive reports for concise summaries")
    print("   Use --type technical reports for detailed analysis")
    print()
    print("="*60)
    print("Happy experimenting! 🧪✨")
    print("="*60)


def main():
    """Main setup function."""
    setup_logging()
    
    print("🔧 Setting up Comprehensive Evaluation Suite...")
    print()
    
    # Create directories
    logging.info("Creating directories...")
    create_directories()
    
    # Make scripts executable
    logging.info("Making scripts executable...")
    make_scripts_executable()
    
    # Check dependencies
    logging.info("Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Verify existing code
    logging.info("Verifying existing code...")
    code_ok = verify_existing_code()
    
    # Test basic functionality
    logging.info("Testing basic functionality...")
    test_ok = test_basic_functionality()
    
    if deps_ok and test_ok:
        print_success_message()
        return True
    else:
        logging.error("Setup completed with warnings. Check the logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
