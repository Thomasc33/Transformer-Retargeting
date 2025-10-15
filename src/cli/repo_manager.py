"""
Repository Manager for TMR
Handles repository structure checks, status, and validation
"""

from pathlib import Path
from typing import Dict, List, Tuple

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, check_file_exists, get_model_path, get_dataset_path, validate_environment
)


class RepoManager:
    """Manages repository structure and status"""
    
    def __init__(self):
        self.root = get_root_dir()
    
    def check_structure(self):
        """Check repository structure"""
        print_header("Repository Structure Check")
        
        # Check directories
        print_section("Directories")
        dirs = [
            ("data", "Data directory"),
            ("src", "Source code"),
            ("configs", "Configuration files"),
            ("output", "Model outputs"),
            ("logs", "Training logs"),
            ("trained_models", "Pretrained models"),
        ]
        
        for dir_name, description in dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                print_success(f"{description}: {dir_path}")
            else:
                print_error(f"{description} missing: {dir_path}")
        
        # Check key files
        print_section("Key Files")
        files = [
            ("README.md", "Project README"),
            ("STATUS_REPORT.md", "Status report"),
            ("RETRAINING_PLAN.md", "Retraining plan"),
            ("configs/main_config.yaml", "Main config"),
            ("data.py", "Data loader"),
            ("eval_model.py", "Baseline evaluation"),
            ("eval_anonymization_v2.py", "Anonymization evaluation"),
        ]
        
        for file_name, description in files:
            file_path = self.root / file_name
            if file_path.exists():
                print_success(f"{description}: {file_path.name}")
            else:
                print_warning(f"{description} missing: {file_path.name}")
    
    def check_data(self):
        """Check available data"""
        print_header("Data Availability")
        
        datasets = [
            ("ntu_cv", "NTU RGB+D Cross-View"),
            ("ntu_cs", "NTU RGB+D Cross-Subject"),
            ("ntu120_cv", "NTU RGB+D 120 Cross-View"),
            ("ntu120_cs", "NTU RGB+D 120 Cross-Subject"),
        ]
        
        for dataset, description in datasets:
            data_path = get_dataset_path(dataset)
            if data_path.exists():
                print_success(f"{description}: {data_path.name}")
            else:
                print_warning(f"{description} missing: {data_path.name}")
    
    def check_models(self):
        """Check available models"""
        print_header("Model Availability")
        
        # TMR models
        print_section("TMR Models")
        tmr_path = self.root / "data" / "models_output" / "model_all.pth"
        if tmr_path.exists():
            print_success(f"TMR model: {tmr_path}")
        else:
            print_warning(f"TMR model missing: {tmr_path}")
        
        # Baseline models
        print_section("Baseline Models")
        baseline_models = [
            ("mixformer", "ntu_cv", "ar", "Mixformer AR"),
            ("mixformer", "ntu_cv", "ri", "Mixformer RI"),
            ("sgn", "ntu_cv", "ar", "SGN AR"),
            ("sgn", "ntu_cv", "ri", "SGN RI"),
        ]
        
        for model_type, dataset, task, description in baseline_models:
            model_path = get_model_path(model_type, dataset, task)
            if model_path.exists():
                print_success(f"{description}: {model_path.name}")
            else:
                print_warning(f"{description} missing: {model_path.name}")
        
        # Comparison models
        print_section("Comparison Models")
        comparison_models = [
            ("pmr", "ntu_cv", "PMR"),
            ("dmr", "ntu_cv", "DMR"),
        ]
        
        for model_type, dataset, description in comparison_models:
            model_path = get_model_path(model_type, dataset, "")
            if model_path.exists():
                print_success(f"{description}: {model_path.name}")
            else:
                print_warning(f"{description} missing: {model_path.name}")
    
    def check_results(self):
        """Check available results"""
        print_header("Results Availability")
        
        results_dir = self.root / "test_results"
        if not results_dir.exists():
            print_warning("No results directory found")
            return
        
        # Count result files
        result_files = list(results_dir.rglob("*.log"))
        print_info(f"Found {len(result_files)} log files")
        
        # Check for visualizations
        viz_dirs = list(results_dir.glob("*visualizations*"))
        print_info(f"Found {len(viz_dirs)} visualization directories")
    
    def validate_env(self):
        """Validate environment"""
        print_header("Environment Validation")
        
        checks = validate_environment()
        
        print_section("System Checks")
        for check_name, status in checks.items():
            if status:
                print_success(f"{check_name}: OK")
            else:
                print_error(f"{check_name}: FAILED")
        
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print_success(f"CUDA available: {torch.cuda.device_count()} GPUs")
                for i in range(torch.cuda.device_count()):
                    print_info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print_warning("CUDA not available (expected on login node)")
        except ImportError:
            print_error("PyTorch not installed")
    
    def show_status(self):
        """Show comprehensive repository status"""
        print_header("TMR Repository Status")
        
        # Environment
        print_section("Environment")
        checks = validate_environment()
        all_passed = all(checks.values())
        if all_passed:
            print_success("All environment checks passed")
        else:
            print_warning("Some environment checks failed")
        
        # Data
        print_section("Data")
        datasets = ["ntu_cv", "ntu_cs", "ntu120_cv"]
        data_count = 0
        for d in datasets:
            # Check for both comprehensive and 10000_2000 variants
            if get_dataset_path(d).exists():
                data_count += 1
            elif (self.root / "data" / f"{d}_paired_10000_2000.pt").exists():
                data_count += 1
        print_info(f"{data_count}/{len(datasets)} datasets available")
        
        # Models
        print_section("Models")
        tmr_exists = (self.root / "data" / "models_output" / "model_all.pth").exists()
        if tmr_exists:
            print_info("TMR model: Available")
        else:
            print_warning("TMR model: Missing")
        
        baseline_count = sum(1 for _ in (self.root / "output").glob("*/model_best.pth.tar"))
        print_info(f"Baseline models: {baseline_count} available")
        
        # Results
        print_section("Results")
        results_dir = self.root / "test_results"
        if results_dir.exists():
            result_count = len(list(results_dir.rglob("*.log")))
            print_info(f"Result files: {result_count}")
        else:
            print_warning("No results directory")
        
        # Jobs
        print_section("Jobs")
        jobs_file = self.root / "jobs.json"
        if jobs_file.exists():
            import json
            with open(jobs_file) as f:
                jobs_data = json.load(f)
            job_count = len(jobs_data.get("jobs", []))
            print_info(f"Tracked jobs: {job_count}")
        else:
            print_info("No jobs tracked yet")
    
    def list_missing(self):
        """List missing components"""
        print_header("Missing Components")
        
        missing = []
        
        # Check datasets
        datasets = ["ntu_cv", "ntu_cs", "ntu120_cv", "ntu120_cs"]
        for dataset in datasets:
            if not get_dataset_path(dataset).exists():
                missing.append(f"Dataset: {dataset}")
        
        # Check models
        models = [
            ("TMR", self.root / "data" / "models_output" / "model_all.pth"),
            ("Mixformer AR", get_model_path("mixformer", "ntu_cv", "ar")),
            ("Mixformer RI", get_model_path("mixformer", "ntu_cv", "ri")),
            ("SGN AR", get_model_path("sgn", "ntu_cv", "ar")),
            ("SGN RI", get_model_path("sgn", "ntu_cv", "ri")),
            ("PMR", get_model_path("pmr", "ntu_cv", "")),
            ("DMR", get_model_path("dmr", "ntu_cv", "")),
        ]
        
        for model_name, model_path in models:
            if not model_path.exists():
                missing.append(f"Model: {model_name}")
        
        # Print results
        if missing:
            print_section("Missing Items")
            for item in missing:
                print_warning(f"  ✗ {item}")
            print()
            print_info(f"Total missing: {len(missing)} items")
        else:
            print_success("All components available!")
    
    def interactive_menu(self):
        """Interactive repository management menu"""
        while True:
            print_header("Repository Management")
            print("1. Check structure")
            print("2. Check data")
            print("3. Check models")
            print("4. Check results")
            print("5. Validate environment")
            print("6. Show status")
            print("7. List missing components")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.check_structure()
            elif choice == "2":
                self.check_data()
            elif choice == "3":
                self.check_models()
            elif choice == "4":
                self.check_results()
            elif choice == "5":
                self.validate_env()
            elif choice == "6":
                self.show_status()
            elif choice == "7":
                self.list_missing()
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")

