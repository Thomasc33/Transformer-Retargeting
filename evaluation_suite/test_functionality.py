#!/usr/bin/env python3
"""
Test script to verify all evaluation suite functionality works.
This script tests imports, configurations, and basic functionality without running actual experiments.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from evaluation_suite.core import ComprehensiveEvaluator, MetricsCalculator, ModelManager, DataManager
        print("✅ Core modules imported successfully")
    except Exception as e:
        print(f"❌ Core modules import failed: {e}")
        return False
    
    try:
        # Experiment modules
        from evaluation_suite.experiments import PrimaryExperiments, AblationExperiments
        print("✅ Experiment modules imported successfully")
    except Exception as e:
        print(f"❌ Experiment modules import failed: {e}")
        return False
    
    try:
        # Runner modules
        from evaluation_suite.runners import SlurmRunner, JobMonitor
        print("✅ Runner modules imported successfully")
    except Exception as e:
        print(f"❌ Runner modules import failed: {e}")
        return False
    
    try:
        # Analysis modules
        from evaluation_suite.analysis import ComprehensiveVisualizer, ResultComparator
        print("✅ Analysis modules imported successfully")
    except Exception as e:
        print(f"❌ Analysis modules import failed: {e}")
        return False
    
    return True

def test_configuration_loading():
    """Test configuration loading."""
    print("\n📋 Testing configuration loading...")
    
    try:
        import yaml
        config_path = "evaluation_suite/configs/experiments.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check required sections
        required_sections = ['primary_experiments', 'hpc', 'experiment_sets']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
                
        print("✅ Configuration loaded and validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_experiment_definitions():
    """Test experiment definitions."""
    print("\n🧪 Testing experiment definitions...")
    
    try:
        from evaluation_suite.experiments.primary import PrimaryExperiments
        
        # Test getting experiment configs
        configs = PrimaryExperiments.get_experiment_configs()
        
        expected_experiments = [
            'privacy_utility_sgn',
            'privacy_utility_mixformer', 
            'baseline_comparison',
            'physical_plausibility'
        ]
        
        for exp_name in expected_experiments:
            if exp_name not in configs:
                print(f"❌ Missing experiment definition: {exp_name}")
                return False
            
            # Check required fields
            exp_config = configs[exp_name]
            required_fields = ['name', 'description', 'models', 'data', 'metrics']
            for field in required_fields:
                if field not in exp_config:
                    print(f"❌ Missing field '{field}' in experiment {exp_name}")
                    return False
                    
        print("✅ Experiment definitions validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Experiment definitions test failed: {e}")
        return False

def test_model_manager():
    """Test model manager functionality."""
    print("\n🤖 Testing model manager...")
    
    try:
        from evaluation_suite.core.models import ModelManager
        
        manager = ModelManager()
        
        # Test model info for non-existent model
        info = manager.get_model_info("non_existent_model")
        if 'status' not in info or info['status'] != 'not_loaded':
            print("❌ Model info for non-existent model failed")
            return False
            
        print("✅ Model manager basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False

def test_data_manager():
    """Test data manager functionality."""
    print("\n📊 Testing data manager...")
    
    try:
        from evaluation_suite.core.data_loader import DataManager
        
        manager = DataManager()
        
        # Test configuration validation
        valid_config = {
            'dataset': 'ntu',
            'setting': 'cv',
            'batch_size': 32
        }
        
        is_valid = manager.validate_data_config(valid_config)
        if not is_valid:
            print("❌ Valid configuration rejected")
            return False
            
        # Test invalid configuration
        invalid_config = {
            'dataset': 'invalid_dataset',
            'setting': 'cv'
        }
        
        is_valid = manager.validate_data_config(invalid_config)
        if is_valid:
            print("❌ Invalid configuration accepted")
            return False
            
        print("✅ Data manager validation works")
        return True
        
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def test_slurm_runner():
    """Test Slurm runner functionality."""
    print("\n🚀 Testing Slurm runner...")
    
    try:
        from evaluation_suite.runners.slurm_runner import SlurmRunner
        
        hpc_config = {
            'default_partition': 'GPU',
            'default_time': '4:00:00',
            'default_mem': '32GB'
        }
        
        runner = SlurmRunner(hpc_config)
        
        # Test job template selection
        exp_config = {'estimated_time': '2 hours'}
        template = runner.get_job_template(exp_config)
        
        if not isinstance(template, dict):
            print("❌ Job template generation failed")
            return False
            
        print("✅ Slurm runner basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Slurm runner test failed: {e}")
        return False

def test_metrics_calculator():
    """Test metrics calculator."""
    print("\n📈 Testing metrics calculator...")
    
    try:
        from evaluation_suite.core.metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        # Test with dummy data
        dummy_results = {
            'model1': {
                'data1': {
                    'action_recognition': {'accuracy': 85.5},
                    'reidentification': {'identity_accuracy': 25.3}
                }
            }
        }
        
        dummy_config = {'dataset': 'ntu'}
        
        metrics = calculator.calculate_all_metrics(dummy_results, dummy_config)
        
        if 'model1' not in metrics:
            print("❌ Metrics calculation failed")
            return False
            
        print("✅ Metrics calculator works")
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculator test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "evaluation_suite/core",
        "evaluation_suite/experiments", 
        "evaluation_suite/runners",
        "evaluation_suite/analysis",
        "evaluation_suite/configs",
        "evaluation_suite/results",
        "evaluation_suite/reports"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
            
    print("✅ Directory structure is complete")
    return True

def main():
    """Run all tests."""
    setup_logging()
    
    print("🧪 EVALUATION SUITE FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_configuration_loading,
        test_experiment_definitions,
        test_model_manager,
        test_data_manager,
        test_slurm_runner,
        test_metrics_calculator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The evaluation suite is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: python evaluation_suite/run_experiments.py --list-experiments")
        print("2. Test: python evaluation_suite/run_experiments.py --experiment-set quick")
        print("3. Monitor: python evaluation_suite/monitor_jobs.py --status")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
