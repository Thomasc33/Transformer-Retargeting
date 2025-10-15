"""
Training Commands for TMR
Handles all training operations: MLM, TMR, PMR, DMR, Mixformer, SGN
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, List

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, confirm_action, select_from_list, estimate_training_time
)
from .slurm_manager import SlurmManager


class TrainCommands:
    """Handles all training operations"""
    
    def __init__(self, slurm_manager: Optional[SlurmManager] = None):
        self.root = get_root_dir()
        self.slurm = slurm_manager
    
    def train_mlm_pretrain(
        self,
        dataset: str = "ntu_cv",
        epochs: int = 100,
        batch_size: int = 32,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train MLM pretraining (encoder/decoder)"""
        print_header("MLM Pretraining")
        
        if test_mode:
            epochs = 5
            dataset_arg = f"{dataset}_test_100_20"
            print_warning(f"Test mode: {epochs} epochs, 100 samples")
        else:
            dataset_arg = dataset
        
        # Estimate time
        num_samples = 100 if test_mode else 1600000
        est_time = estimate_training_time(num_samples, epochs, batch_size, num_gpus)
        print_info(f"Estimated training time: {est_time:.1f} hours")
        
        # Build command
        cmd = f"python -u src/training/pretrain.py --dataset {dataset_arg} --epochs {epochs} --batch-size {batch_size}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"mlm_pretrain_{dataset}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=int(est_time * 1.5) + 1,
                mem_gb=64
            )
            return job_id is not None
        else:
            print_info("Running locally...")
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_tmr_stage1(
        self,
        dataset: str = "ntu_cv",
        epochs: int = 3,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train TMR Stage 1 (action-focused)"""
        print_header("TMR Stage 1 Training (Action-Focused)")
        
        if test_mode:
            dataset_arg = f"{dataset}_test_100_20"
            print_warning("Test mode: 100 samples")
        else:
            dataset_arg = dataset
        
        # Check if training script exists
        train_script = self.root / "src" / "training" / "train_stage1.py"
        if not train_script.exists():
            print_error(f"Training script not found: {train_script}")
            print_info("Please implement Stage 1 training first (see RETRAINING_PLAN.md)")
            return False
        
        cmd = f"python -u src/training/train_stage1.py --dataset {dataset_arg} --epochs {epochs}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"tmr_stage1_{dataset}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=48,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_tmr_stage2(
        self,
        dataset: str = "ntu_cv",
        epochs: int = 2,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False,
        stage1_checkpoint: Optional[str] = None
    ):
        """Train TMR Stage 2 (privacy fine-tuning)"""
        print_header("TMR Stage 2 Training (Privacy Fine-Tuning)")
        
        if test_mode:
            dataset_arg = f"{dataset}_test_100_20"
            print_warning("Test mode: 100 samples")
        else:
            dataset_arg = dataset
        
        # Check if training script exists
        train_script = self.root / "src" / "training" / "train_stage2.py"
        if not train_script.exists():
            print_error(f"Training script not found: {train_script}")
            print_info("Please implement Stage 2 training first (see RETRAINING_PLAN.md)")
            return False
        
        cmd = f"python -u src/training/train_stage2.py --dataset {dataset_arg} --epochs {epochs}"
        if stage1_checkpoint:
            cmd += f" --checkpoint {stage1_checkpoint}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"tmr_stage2_{dataset}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=36,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_mixformer(
        self,
        dataset: str = "ntu_cv",
        task: str = "ar",
        setting: str = "cview",
        epochs: int = 120,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train Mixformer model"""
        print_header(f"Mixformer Training ({task.upper()}, {setting})")
        
        if test_mode:
            epochs = 15
            dataset_arg = f"{dataset}_test_100_20"
            print_warning(f"Test mode: {epochs} epochs, 100 samples")
        else:
            dataset_arg = dataset
        
        # Map task names
        task_map = {"ar": "action", "ri": "reid", "gc": "gender"}
        task_full = task_map.get(task, task)
        
        cmd = f"python -u src/training/main.py --dataset {dataset_arg} --task {task_full} --setting {setting} --epochs {epochs}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"mixformer_{task}_{dataset}_{setting}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=72 if not test_mode else 4,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_sgn(
        self,
        dataset: str = "ntu_cv",
        task: str = "ar",
        setting: str = "cview",
        epochs: int = 120,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train SGN model"""
        print_header(f"SGN Training ({task.upper()}, {setting})")
        
        if test_mode:
            epochs = 15
            dataset_arg = f"{dataset}_test_100_20"
            print_warning(f"Test mode: {epochs} epochs, 100 samples")
        else:
            dataset_arg = dataset
        
        cmd = f"python -u src/training/train_sgn.py --dataset {dataset_arg} --task {task} --setting {setting} --epochs {epochs}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"sgn_{task}_{dataset}_{setting}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=72 if not test_mode else 4,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_pmr(
        self,
        dataset: str = "ntu_cv",
        setting: str = "cview",
        epochs: int = 100,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train PMR model"""
        print_header(f"PMR Training ({setting})")
        
        if test_mode:
            epochs = 15
            dataset_arg = f"{dataset}_test_100_20"
            print_warning(f"Test mode: {epochs} epochs, 100 samples")
        else:
            dataset_arg = dataset
        
        # Check if PMR training script exists
        pmr_script = self.root / "src" / "training" / "train_pmr.py"
        if not pmr_script.exists():
            print_error(f"PMR training script not found: {pmr_script}")
            print_info("PMR training not yet implemented")
            return False
        
        cmd = f"python -u src/training/train_pmr.py --dataset {dataset_arg} --setting {setting} --epochs {epochs}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"pmr_{dataset}_{setting}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=96,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def train_dmr(
        self,
        dataset: str = "ntu_cv",
        setting: str = "cview",
        epochs: int = 100,
        num_gpus: int = 4,
        use_slurm: bool = True,
        test_mode: bool = False
    ):
        """Train DMR model"""
        print_header(f"DMR Training ({setting})")
        
        if test_mode:
            epochs = 15
            dataset_arg = f"{dataset}_test_100_20"
            print_warning(f"Test mode: {epochs} epochs, 100 samples")
        else:
            dataset_arg = dataset
        
        # Check if DMR training script exists
        dmr_script = self.root / "src" / "training" / "train_dmr.py"
        if not dmr_script.exists():
            print_error(f"DMR training script not found: {dmr_script}")
            print_info("DMR training not yet implemented")
            return False
        
        cmd = f"python -u src/training/train_dmr.py --dataset {dataset_arg} --setting {setting} --epochs {epochs}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"dmr_{dataset}_{setting}",
                command=cmd,
                num_gpus=num_gpus,
                time_hours=96,
                mem_gb=64
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def interactive_menu(self):
        """Interactive training menu"""
        while True:
            print_header("Training Operations")
            print("1. MLM Pretraining")
            print("2. TMR Stage 1 (Action-Focused)")
            print("3. TMR Stage 2 (Privacy Fine-Tuning)")
            print("4. Train Mixformer")
            print("5. Train SGN")
            print("6. Train PMR")
            print("7. Train DMR")
            print("8. Train All Baseline Models")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self._train_mlm_interactive()
            elif choice == "2":
                self._train_tmr_stage1_interactive()
            elif choice == "3":
                self._train_tmr_stage2_interactive()
            elif choice == "4":
                self._train_mixformer_interactive()
            elif choice == "5":
                self._train_sgn_interactive()
            elif choice == "6":
                self._train_pmr_interactive()
            elif choice == "7":
                self._train_dmr_interactive()
            elif choice == "8":
                self._train_all_baselines()
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")
    
    def _train_mlm_interactive(self):
        """Interactive MLM training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_mlm_pretrain(dataset=dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_tmr_stage1_interactive(self):
        """Interactive TMR Stage 1 training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_tmr_stage1(dataset=dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_tmr_stage2_interactive(self):
        """Interactive TMR Stage 2 training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_tmr_stage2(dataset=dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_mixformer_interactive(self):
        """Interactive Mixformer training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs", "ntu120_cv"], "Select dataset") or "ntu_cv"
        task = select_from_list(["ar", "ri", "gc"], "Select task") or "ar"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_mixformer(dataset=dataset, task=task, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_sgn_interactive(self):
        """Interactive SGN training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs", "ntu120_cv"], "Select dataset") or "ntu_cv"
        task = select_from_list(["ar", "ri", "gc"], "Select task") or "ar"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_sgn(dataset=dataset, task=task, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_pmr_interactive(self):
        """Interactive PMR training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_pmr(dataset=dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_dmr_interactive(self):
        """Interactive DMR training"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode (100 samples)?", default=False)
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.train_dmr(dataset=dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _train_all_baselines(self):
        """Train all baseline models"""
        print_header("Train All Baseline Models")
        print_warning("This will submit multiple SLURM jobs")
        
        if not confirm_action("Continue?", default=False):
            return
        
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        test_mode = confirm_action("Test mode?", default=False)
        
        # Train Mixformer models
        for task in ["ar", "ri"]:
            print_info(f"Submitting Mixformer {task.upper()}...")
            self.train_mixformer(dataset=dataset, task=task, use_slurm=True, test_mode=test_mode)
        
        # Train SGN models
        for task in ["ar", "ri"]:
            print_info(f"Submitting SGN {task.upper()}...")
            self.train_sgn(dataset=dataset, task=task, use_slurm=True, test_mode=test_mode)
        
        print_success("All baseline training jobs submitted!")

