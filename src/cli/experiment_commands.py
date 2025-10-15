"""
Experiment Commands for TMR
Handles experiments and ablations
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, List

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, confirm_action, select_from_list
)
from .slurm_manager import SlurmManager


class ExperimentCommands:
    """Handles experiments and ablations"""
    
    def __init__(self, slurm_manager: Optional[SlurmManager] = None):
        self.root = get_root_dir()
        self.slurm = slurm_manager
    
    def run_masking_ablation(
        self,
        masking_type: str = "temporal",
        ratios: List[float] = [0.1, 0.3, 0.5, 0.7],
        use_slurm: bool = True
    ):
        """Run masking ratio ablation study"""
        print_header(f"{masking_type.capitalize()} Masking Ablation")
        
        job_ids = []
        for ratio in ratios:
            print_info(f"Submitting {masking_type} masking ratio {ratio}...")
            
            cmd = f"python -u src/evaluation/evaluate_masking_results.py --masking-type {masking_type} --ratio {ratio}"
            
            if use_slurm:
                job_id = self.slurm.submit_job(
                    job_name=f"ablation_{masking_type}_{ratio}",
                    command=cmd,
                    num_gpus=1,
                    time_hours=4,
                    mem_gb=32
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                result = subprocess.run(cmd, shell=True, cwd=self.root)
                if result.returncode != 0:
                    print_error(f"Failed for ratio {ratio}")
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} ablation jobs")
        return len(job_ids) > 0
    
    def run_loss_ablation(
        self,
        loss_components: List[str] = ["mse", "ee", "smoothing", "velocity", "bone"],
        use_slurm: bool = True
    ):
        """Run loss component ablation study"""
        print_header("Loss Component Ablation")
        
        print_warning("This requires modified training scripts")
        if not confirm_action("Continue?", default=False):
            return False
        
        job_ids = []
        for component in loss_components:
            print_info(f"Submitting ablation without {component}...")
            
            cmd = f"python -u src/training/retargeting_main.py --ablate-loss {component} --epochs 5"
            
            if use_slurm:
                job_id = self.slurm.submit_job(
                    job_name=f"ablation_loss_{component}",
                    command=cmd,
                    num_gpus=4,
                    time_hours=24,
                    mem_gb=64
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                result = subprocess.run(cmd, shell=True, cwd=self.root)
                if result.returncode != 0:
                    print_error(f"Failed for {component}")
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} ablation jobs")
        return len(job_ids) > 0
    
    def run_teacher_forcing_experiment(
        self,
        schedules: List[str] = ["constant", "linear", "exponential"],
        use_slurm: bool = True
    ):
        """Run teacher forcing schedule experiment"""
        print_header("Teacher Forcing Schedule Experiment")
        
        job_ids = []
        for schedule in schedules:
            print_info(f"Submitting {schedule} schedule...")
            
            cmd = f"python -u src/training/retargeting_main.py --tf-schedule {schedule} --epochs 5"
            
            if use_slurm:
                job_id = self.slurm.submit_job(
                    job_name=f"exp_tf_{schedule}",
                    command=cmd,
                    num_gpus=4,
                    time_hours=24,
                    mem_gb=64
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                result = subprocess.run(cmd, shell=True, cwd=self.root)
                if result.returncode != 0:
                    print_error(f"Failed for {schedule}")
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} experiment jobs")
        return len(job_ids) > 0
    
    def run_architecture_ablation(
        self,
        variants: List[str] = ["no_encoder", "no_decoder", "smaller", "larger"],
        use_slurm: bool = True
    ):
        """Run architecture ablation study"""
        print_header("Architecture Ablation")
        
        print_warning("This requires modified model architectures")
        if not confirm_action("Continue?", default=False):
            return False
        
        job_ids = []
        for variant in variants:
            print_info(f"Submitting {variant} variant...")
            
            cmd = f"python -u src/training/retargeting_main.py --architecture {variant} --epochs 5"
            
            if use_slurm:
                job_id = self.slurm.submit_job(
                    job_name=f"ablation_arch_{variant}",
                    command=cmd,
                    num_gpus=4,
                    time_hours=24,
                    mem_gb=64
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                result = subprocess.run(cmd, shell=True, cwd=self.root)
                if result.returncode != 0:
                    print_error(f"Failed for {variant}")
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} ablation jobs")
        return len(job_ids) > 0
    
    def run_all_experiments(self, use_slurm: bool = True):
        """Run all experiments"""
        print_header("Running All Experiments")
        print_warning("This will submit many SLURM jobs")
        
        if not confirm_action("Continue?", default=False):
            return False
        
        # Masking ablations
        print_section("Temporal Masking Ablation")
        self.run_masking_ablation("temporal", use_slurm=use_slurm)
        
        print_section("Spatial Masking Ablation")
        self.run_masking_ablation("spatial", use_slurm=use_slurm)
        
        # Teacher forcing experiment
        print_section("Teacher Forcing Experiment")
        self.run_teacher_forcing_experiment(use_slurm=use_slurm)
        
        print_success("All experiments submitted!")
        return True
    
    def interactive_menu(self):
        """Interactive experiments menu"""
        while True:
            print_header("Experiments & Ablations")
            print("1. Temporal Masking Ablation")
            print("2. Spatial Masking Ablation")
            print("3. Loss Component Ablation")
            print("4. Teacher Forcing Experiment")
            print("5. Architecture Ablation")
            print("6. Run All Experiments")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_masking_ablation("temporal", use_slurm=use_slurm)
            elif choice == "2":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_masking_ablation("spatial", use_slurm=use_slurm)
            elif choice == "3":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_loss_ablation(use_slurm=use_slurm)
            elif choice == "4":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_teacher_forcing_experiment(use_slurm=use_slurm)
            elif choice == "5":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_architecture_ablation(use_slurm=use_slurm)
            elif choice == "6":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_all_experiments(use_slurm=use_slurm)
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")

