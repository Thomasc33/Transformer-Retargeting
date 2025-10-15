"""
Evaluation Commands for TMR
Handles all evaluation operations: baseline models, anonymization models, experiments
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, List

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, confirm_action, select_from_list, load_results, save_results
)
from .slurm_manager import SlurmManager


class EvalCommands:
    """Handles all evaluation operations"""
    
    def __init__(self, slurm_manager: Optional[SlurmManager] = None):
        self.root = get_root_dir()
        self.slurm = slurm_manager
    
    def eval_baseline(
        self,
        model: str = "mixformer",
        dataset: str = "ntu_cv",
        task: str = "ar",
        num_samples: int = 1000,
        use_slurm: bool = False
    ):
        """Evaluate baseline model (Mixformer, SGN, Raw)"""
        print_header(f"Evaluating {model.upper()} - {task.upper()}")
        
        cmd = f"python -u eval_model.py --model {model} --dataset {dataset} --task {task} --test-samples {num_samples}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"eval_{model}_{task}_{dataset}",
                command=cmd,
                num_gpus=1,
                time_hours=4,
                mem_gb=32
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            if result.returncode == 0:
                print_success("Evaluation complete")
                self._update_results(model, dataset, task, num_samples)
                return True
            else:
                print_error("Evaluation failed")
                return False
    
    def eval_anonymization(
        self,
        model: str = "tmr",
        dataset: str = "ntu_cv",
        num_samples: int = 1000,
        use_slurm: bool = False
    ):
        """Evaluate anonymization model (TMR, PMR, DMR)"""
        print_header(f"Evaluating {model.upper()}")
        
        cmd = f"python -u eval_anonymization_v2.py --model {model} --dataset {dataset} --test-samples {num_samples}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"eval_{model}_{dataset}",
                command=cmd,
                num_gpus=1,
                time_hours=8,
                mem_gb=32
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            if result.returncode == 0:
                print_success("Evaluation complete")
                self._update_results(model, dataset, "anonymization", num_samples)
                return True
            else:
                print_error("Evaluation failed")
                return False
    
    def eval_mlm(
        self,
        dataset: str = "ntu_cv",
        num_samples: int = 1000,
        use_slurm: bool = False
    ):
        """Evaluate MLM pretraining"""
        print_header("Evaluating MLM Pretraining")
        
        cmd = f"python -u src/evaluation/eval_pretrained.py --dataset {dataset} --test-samples {num_samples}"
        
        if use_slurm:
            job_id = self.slurm.submit_job(
                job_name=f"eval_mlm_{dataset}",
                command=cmd,
                num_gpus=1,
                time_hours=4,
                mem_gb=32
            )
            return job_id is not None
        else:
            result = subprocess.run(cmd, shell=True, cwd=self.root)
            return result.returncode == 0
    
    def eval_all_baselines(
        self,
        dataset: str = "ntu_cv",
        num_samples: int = 1000,
        use_slurm: bool = True
    ):
        """Evaluate all baseline models"""
        print_header("Evaluating All Baseline Models")
        
        models_tasks = [
            ("mixformer", "ar"),
            ("mixformer", "ri"),
            ("sgn", "ar"),
            ("sgn", "ri"),
            ("raw", "ar"),
            ("raw", "ri"),
        ]
        
        job_ids = []
        for model, task in models_tasks:
            print_info(f"Submitting {model} {task}...")
            if use_slurm:
                cmd = f"python -u eval_model.py --model {model} --dataset {dataset} --task {task} --test-samples {num_samples}"
                job_id = self.slurm.submit_job(
                    job_name=f"eval_{model}_{task}_{dataset}",
                    command=cmd,
                    num_gpus=1,
                    time_hours=4,
                    mem_gb=32
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                self.eval_baseline(model, dataset, task, num_samples, use_slurm=False)
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} evaluation jobs")
        return len(job_ids) > 0
    
    def eval_all_anonymization(
        self,
        dataset: str = "ntu_cv",
        num_samples: int = 1000,
        use_slurm: bool = True
    ):
        """Evaluate all anonymization models"""
        print_header("Evaluating All Anonymization Models")
        
        models = ["tmr", "pmr", "dmr"]
        
        job_ids = []
        for model in models:
            print_info(f"Submitting {model}...")
            if use_slurm:
                cmd = f"python -u eval_anonymization_v2.py --model {model} --dataset {dataset} --test-samples {num_samples}"
                job_id = self.slurm.submit_job(
                    job_name=f"eval_{model}_{dataset}",
                    command=cmd,
                    num_gpus=1,
                    time_hours=8,
                    mem_gb=32
                )
                if job_id:
                    job_ids.append(job_id)
            else:
                self.eval_anonymization(model, dataset, num_samples, use_slurm=False)
        
        if job_ids:
            print_success(f"Submitted {len(job_ids)} evaluation jobs")
        return len(job_ids) > 0
    
    def eval_comprehensive(
        self,
        dataset: str = "ntu_cv",
        num_samples: int = 1000,
        use_slurm: bool = True
    ):
        """Run comprehensive evaluation (all models)"""
        print_header("Comprehensive Evaluation")
        print_warning("This will evaluate ALL models")
        
        if not confirm_action("Continue?", default=True):
            return False
        
        # Evaluate baselines
        print_section("Baseline Models")
        self.eval_all_baselines(dataset, num_samples, use_slurm)
        
        # Evaluate anonymization
        print_section("Anonymization Models")
        self.eval_all_anonymization(dataset, num_samples, use_slurm)
        
        # Evaluate MLM
        print_section("MLM Pretraining")
        self.eval_mlm(dataset, num_samples, use_slurm)
        
        print_success("Comprehensive evaluation submitted!")
        return True
    
    def _update_results(self, model: str, dataset: str, task: str, num_samples: int):
        """Update results.json with evaluation results"""
        results = load_results()
        
        if model not in results["models"]:
            results["models"][model] = {}
        
        results["models"][model][dataset] = {
            "status": "evaluated",
            "task": task,
            "samples": num_samples,
            "evaluated_at": __import__('datetime').datetime.now().isoformat()
        }
        
        save_results(results)
    
    def interactive_menu(self):
        """Interactive evaluation menu"""
        while True:
            print_header("Evaluation Operations")
            print("1. Evaluate Baseline Model (Mixformer/SGN/Raw)")
            print("2. Evaluate Anonymization Model (TMR/PMR/DMR)")
            print("3. Evaluate MLM Pretraining")
            print("4. Evaluate All Baselines")
            print("5. Evaluate All Anonymization")
            print("6. Comprehensive Evaluation (All Models)")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self._eval_baseline_interactive()
            elif choice == "2":
                self._eval_anonymization_interactive()
            elif choice == "3":
                self._eval_mlm_interactive()
            elif choice == "4":
                self._eval_all_baselines_interactive()
            elif choice == "5":
                self._eval_all_anonymization_interactive()
            elif choice == "6":
                self._eval_comprehensive_interactive()
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")
    
    def _eval_baseline_interactive(self):
        """Interactive baseline evaluation"""
        model = select_from_list(["mixformer", "sgn", "raw"], "Select model") or "mixformer"
        dataset = select_from_list(["ntu_cv", "ntu_cs", "ntu120_cv"], "Select dataset") or "ntu_cv"
        task = select_from_list(["ar", "ri", "gc"], "Select task") or "ar"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=False)
        self.eval_baseline(model, dataset, task, num_samples, use_slurm)
    
    def _eval_anonymization_interactive(self):
        """Interactive anonymization evaluation"""
        model = select_from_list(["tmr", "pmr", "dmr"], "Select model") or "tmr"
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=False)
        self.eval_anonymization(model, dataset, num_samples, use_slurm)
    
    def _eval_mlm_interactive(self):
        """Interactive MLM evaluation"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=False)
        self.eval_mlm(dataset, num_samples, use_slurm)
    
    def _eval_all_baselines_interactive(self):
        """Interactive all baselines evaluation"""
        dataset = select_from_list(["ntu_cv", "ntu_cs", "ntu120_cv"], "Select dataset") or "ntu_cv"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.eval_all_baselines(dataset, num_samples, use_slurm)
    
    def _eval_all_anonymization_interactive(self):
        """Interactive all anonymization evaluation"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.eval_all_anonymization(dataset, num_samples, use_slurm)
    
    def _eval_comprehensive_interactive(self):
        """Interactive comprehensive evaluation"""
        dataset = select_from_list(["ntu_cv", "ntu_cs"], "Select dataset") or "ntu_cv"
        
        num_samples_str = input("Number of samples [1000]: ").strip()
        num_samples = int(num_samples_str) if num_samples_str else 1000
        
        use_slurm = confirm_action("Use SLURM?", default=True)
        self.eval_comprehensive(dataset, num_samples, use_slurm)

