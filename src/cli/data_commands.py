"""
Data Management Commands for TMR
Handles preprocessing, sampling, validation, and statistics
"""

import torch
from pathlib import Path
from typing import Optional, Dict, List
import subprocess

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, check_file_exists, get_dataset_path, create_sample_subset,
    confirm_action, select_from_list
)
from .slurm_manager import SlurmManager


class DataCommands:
    """Handles all data-related operations"""
    
    def __init__(self, slurm_manager: Optional[SlurmManager] = None):
        self.root = get_root_dir()
        self.data_dir = self.root / "data"
        self.slurm = slurm_manager
    
    def list_datasets(self):
        """List available datasets"""
        print_header("Available Datasets")
        
        datasets = {
            "NTU RGB+D": [],
            "NTU RGB+D 120": [],
            "ETRI": [],
            "Processed": []
        }
        
        # Check raw data
        ntu_dir = self.data_dir / "nturgbd_raw" / "nturgb+d_skeletons"
        if ntu_dir.exists():
            datasets["NTU RGB+D"].append(f"Raw data: {ntu_dir}")

        ntu120_dir = self.data_dir / "nturgbd_raw" / "nturgb+d_skeletons120"
        if ntu120_dir.exists():
            datasets["NTU RGB+D 120"].append(f"Raw data: {ntu120_dir}")

        etri_dir = self.data_dir / "etri"
        if etri_dir.exists():
            datasets["ETRI"].append(f"Raw data: {etri_dir}")
        
        # Check processed data
        for pt_file in self.data_dir.glob("*.pt"):
            datasets["Processed"].append(pt_file.name)
        
        # Print results
        for category, items in datasets.items():
            print_section(category)
            if items:
                for item in items:
                    print(f"  ✓ {item}")
            else:
                print(f"  ✗ No data found")
        
        print()
    
    def preprocess_data(
        self,
        dataset: str = "ntu_cv",
        use_slurm: bool = False,
        test_mode: bool = False
    ):
        """Preprocess raw data into paired format"""
        print_header(f"Preprocessing {dataset.upper()}")
        
        # Check if raw data exists
        if dataset.startswith("ntu120"):
            raw_dir = self.data_dir / "nturgbd_raw" / "nturgb+d_skeletons120"
        elif dataset.startswith("ntu"):
            raw_dir = self.data_dir / "nturgbd_raw" / "nturgb+d_skeletons"
        elif dataset == "etri":
            raw_dir = self.data_dir / "etri"
        else:
            print_error(f"Unknown dataset: {dataset}")
            return False

        if not raw_dir.exists():
            print_error(f"Raw data not found: {raw_dir}")
            return False
        
        # Determine output file
        if test_mode:
            output_file = self.data_dir / f"{dataset}_paired_10000_2000.pt"
        else:
            output_file = self.data_dir / f"{dataset}_paired_comprehensive.pt"
        
        # Check if already exists
        if output_file.exists():
            if not confirm_action(f"Output file exists: {output_file.name}. Overwrite?"):
                return False
        
        # Build preprocessing command
        cmd = f"python -u data.py --dataset {dataset}"
        if test_mode:
            cmd += " --max-train 10000 --max-test 2000"
        
        if use_slurm:
            print_info("Submitting preprocessing job to SLURM...")
            job_id = self.slurm.submit_job(
                job_name=f"preprocess_{dataset}",
                command=cmd,
                num_gpus=0,  # CPU only
                time_hours=4,
                mem_gb=64,
                partition="cpu"
            )
            return job_id is not None
        else:
            print_info("Running preprocessing locally...")
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            if result.returncode == 0:
                print_success(f"Preprocessing complete: {output_file}")
                return True
            else:
                print_error("Preprocessing failed")
                return False
    
    def create_test_subset(
        self,
        dataset: str = "ntu_cv",
        num_train: int = 100,
        num_test: int = 20
    ):
        """Create small test subset for debugging"""
        print_header(f"Creating Test Subset ({num_train}/{num_test} samples)")
        
        # Find source file
        source_file = get_dataset_path(dataset)
        if not source_file.exists():
            print_error(f"Source dataset not found: {source_file}")
            return False
        
        # Create output file
        output_file = self.data_dir / f"{dataset}_test_{num_train}_{num_test}.pt"
        
        print_info(f"Loading data from {source_file.name}...")
        try:
            data = torch.load(source_file)
            
            if isinstance(data, dict):
                # Assuming format: {'train': [...], 'test': [...]}
                subset = {}
                if 'train' in data:
                    subset['train'] = data['train'][:num_train]
                if 'test' in data:
                    subset['test'] = data['test'][:num_test]
                
                torch.save(subset, output_file)
                print_success(f"Created test subset: {output_file.name}")
                print_info(f"  Train samples: {len(subset.get('train', []))}")
                print_info(f"  Test samples: {len(subset.get('test', []))}")
                return True
            else:
                print_error(f"Unexpected data format: {type(data)}")
                return False
                
        except Exception as e:
            print_error(f"Failed to create subset: {e}")
            return False
    
    def validate_data(self, dataset: str = "ntu_cv"):
        """Validate dataset integrity"""
        print_header(f"Validating {dataset.upper()}")
        
        data_file = get_dataset_path(dataset)
        if not check_file_exists(data_file, "Dataset"):
            return False
        
        print_info("Loading dataset...")
        try:
            data = torch.load(data_file)
            
            print_section("Dataset Structure")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} samples")
                    elif isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            elif isinstance(data, list):
                print(f"  List with {len(data)} samples")
                if len(data) > 0:
                    sample = data[0]
                    print(f"  Sample type: {type(sample)}")
                    if isinstance(sample, (list, tuple)):
                        print(f"  Sample length: {len(sample)}")
            
            print_section("Sample Inspection")
            if isinstance(data, dict) and 'train' in data:
                sample = data['train'][0]
            elif isinstance(data, list):
                sample = data[0]
            else:
                print_warning("Cannot inspect sample - unknown format")
                return True
            
            # Inspect sample structure
            if isinstance(sample, (list, tuple)):
                print(f"  Sample is {type(sample).__name__} with {len(sample)} elements")
                for i, elem in enumerate(sample):
                    if isinstance(elem, torch.Tensor):
                        print(f"    [{i}] Tensor: {elem.shape}")
                    else:
                        print(f"    [{i}] {type(elem).__name__}: {elem}")
            
            print_success("Dataset validation complete")
            return True
            
        except Exception as e:
            print_error(f"Validation failed: {e}")
            return False
    
    def show_statistics(self, dataset: str = "ntu_cv"):
        """Show dataset statistics"""
        print_header(f"Dataset Statistics: {dataset.upper()}")
        
        data_file = get_dataset_path(dataset)
        if not data_file.exists():
            print_error(f"Dataset not found: {data_file}")
            return False
        
        print_info("Computing statistics...")
        try:
            data = torch.load(data_file)
            
            # Count samples
            if isinstance(data, dict):
                train_count = len(data.get('train', []))
                test_count = len(data.get('test', []))
                print_section("Sample Counts")
                print(f"  Training samples: {train_count:,}")
                print(f"  Test samples: {test_count:,}")
                print(f"  Total samples: {train_count + test_count:,}")
            elif isinstance(data, list):
                print_section("Sample Counts")
                print(f"  Total samples: {len(data):,}")
            
            # Analyze skeleton data
            print_section("Skeleton Statistics")
            if isinstance(data, dict) and 'train' in data:
                sample = data['train'][0]
            elif isinstance(data, list):
                sample = data[0]
            else:
                print_warning("Cannot compute statistics - unknown format")
                return True
            
            # Assuming Cross_Data format: [x1, x2, y1, y2, actors, actions]
            if isinstance(sample, (list, tuple)) and len(sample) >= 6:
                x1 = sample[0]
                if isinstance(x1, torch.Tensor):
                    print(f"  Skeleton shape: {x1.shape}")
                    print(f"  Channels: {x1.shape[0]}")
                    print(f"  Frames: {x1.shape[1]}")
                    print(f"  Joints: {x1.shape[2]}")
                    print(f"  Persons: {x1.shape[3]}")
                    
                    # Compute statistics
                    print(f"  Mean: {x1.mean():.4f}")
                    print(f"  Std: {x1.std():.4f}")
                    print(f"  Min: {x1.min():.4f}")
                    print(f"  Max: {x1.max():.4f}")
            
            print_success("Statistics computed successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to compute statistics: {e}")
            return False
    
    def interactive_menu(self):
        """Interactive data management menu"""
        while True:
            print_header("Data Management")
            print("1. List datasets")
            print("2. Preprocess data")
            print("3. Create test subset")
            print("4. Validate dataset")
            print("5. Show statistics")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.list_datasets()
            elif choice == "2":
                dataset = select_from_list(
                    ["ntu_cv", "ntu_cs", "ntu120_cv", "ntu120_cs", "etri"],
                    "Select dataset to preprocess"
                )
                if dataset:
                    use_slurm = confirm_action("Use SLURM?", default=True)
                    test_mode = confirm_action("Test mode (10k/2k samples)?", default=False)
                    self.preprocess_data(dataset, use_slurm, test_mode)
            elif choice == "3":
                dataset = select_from_list(
                    ["ntu_cv", "ntu_cs", "ntu120_cv", "ntu120_cs"],
                    "Select dataset"
                )
                if dataset:
                    self.create_test_subset(dataset)
            elif choice == "4":
                dataset = select_from_list(
                    ["ntu_cv", "ntu_cs", "ntu120_cv", "ntu120_cs"],
                    "Select dataset to validate"
                )
                if dataset:
                    self.validate_data(dataset)
            elif choice == "5":
                dataset = select_from_list(
                    ["ntu_cv", "ntu_cs", "ntu120_cv", "ntu120_cs"],
                    "Select dataset"
                )
                if dataset:
                    self.show_statistics(dataset)
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")

