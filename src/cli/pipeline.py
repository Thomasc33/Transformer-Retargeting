"""
Full Pipeline Orchestration for TMR
Runs everything from scratch to paper-ready results
"""

from pathlib import Path
from typing import Optional, List, Dict

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    get_root_dir, confirm_action, get_dataset_path, get_model_path
)
from .slurm_manager import SlurmManager
from .data_commands import DataCommands
from .train_commands import TrainCommands
from .eval_commands import EvalCommands
from .experiment_commands import ExperimentCommands


class Pipeline:
    """Orchestrates full pipeline execution"""
    
    def __init__(self, slurm_manager: Optional[SlurmManager] = None):
        self.root = get_root_dir()
        self.slurm = slurm_manager
        self.data_cmd = DataCommands(slurm_manager)
        self.train_cmd = TrainCommands(slurm_manager)
        self.eval_cmd = EvalCommands(slurm_manager)
        self.exp_cmd = ExperimentCommands(slurm_manager)
    
    def run_full_pipeline(
        self,
        dataset: str = "ntu_cv",
        test_mode: bool = False,
        use_slurm: bool = True
    ):
        """Run complete pipeline from scratch"""
        print_header("Full Pipeline Execution")
        print_warning("This will run the ENTIRE pipeline")
        print_info(f"Dataset: {dataset}")
        print_info(f"Test mode: {test_mode}")
        print_info(f"Use SLURM: {use_slurm}")
        
        if not confirm_action("Continue?", default=False):
            return False
        
        # Phase 1: Data Preparation
        print_section("Phase 1: Data Preparation")
        if not self._phase1_data(dataset, test_mode, use_slurm):
            print_error("Phase 1 failed")
            return False
        
        # Phase 2: Baseline Training
        print_section("Phase 2: Baseline Training")
        if not self._phase2_baselines(dataset, test_mode, use_slurm):
            print_error("Phase 2 failed")
            return False
        
        # Phase 3: MLM Pretraining
        print_section("Phase 3: MLM Pretraining")
        if not self._phase3_mlm(dataset, test_mode, use_slurm):
            print_error("Phase 3 failed")
            return False
        
        # Phase 4: TMR Training
        print_section("Phase 4: TMR Training")
        if not self._phase4_tmr(dataset, test_mode, use_slurm):
            print_error("Phase 4 failed")
            return False
        
        # Phase 5: Evaluation
        print_section("Phase 5: Comprehensive Evaluation")
        if not self._phase5_evaluation(dataset, test_mode, use_slurm):
            print_error("Phase 5 failed")
            return False
        
        # Phase 6: Experiments
        print_section("Phase 6: Experiments & Ablations")
        if not self._phase6_experiments(use_slurm):
            print_error("Phase 6 failed")
            return False
        
        print_success("Full pipeline submitted successfully!")
        return True
    
    def _phase1_data(self, dataset: str, test_mode: bool, use_slurm: bool) -> bool:
        """Phase 1: Data preparation"""
        # Check if data exists
        data_path = get_dataset_path(dataset)
        if data_path.exists():
            print_info(f"Dataset already exists: {data_path.name}")
            return True
        
        # Preprocess data
        print_info("Preprocessing data...")
        return self.data_cmd.preprocess_data(dataset, use_slurm, test_mode)
    
    def _phase2_baselines(self, dataset: str, test_mode: bool, use_slurm: bool) -> bool:
        """Phase 2: Train baseline models"""
        # Check if baselines exist
        mixformer_ar = get_model_path("mixformer", dataset, "ar")
        mixformer_ri = get_model_path("mixformer", dataset, "ri")
        sgn_ar = get_model_path("sgn", dataset, "ar")
        sgn_ri = get_model_path("sgn", dataset, "ri")
        
        all_exist = all([
            mixformer_ar.exists(),
            mixformer_ri.exists(),
            sgn_ar.exists(),
            sgn_ri.exists()
        ])
        
        if all_exist:
            print_info("All baseline models already exist")
            return True
        
        # Train baselines
        print_info("Training baseline models...")
        
        # Mixformer AR
        if not mixformer_ar.exists():
            print_info("Training Mixformer AR...")
            self.train_cmd.train_mixformer(dataset, "ar", "cview", use_slurm=use_slurm, test_mode=test_mode)
        
        # Mixformer RI
        if not mixformer_ri.exists():
            print_info("Training Mixformer RI...")
            self.train_cmd.train_mixformer(dataset, "ri", "cview", use_slurm=use_slurm, test_mode=test_mode)
        
        # SGN AR
        if not sgn_ar.exists():
            print_info("Training SGN AR...")
            self.train_cmd.train_sgn(dataset, "ar", "cview", use_slurm=use_slurm, test_mode=test_mode)
        
        # SGN RI
        if not sgn_ri.exists():
            print_info("Training SGN RI...")
            self.train_cmd.train_sgn(dataset, "ri", "cview", use_slurm=use_slurm, test_mode=test_mode)
        
        return True
    
    def _phase3_mlm(self, dataset: str, test_mode: bool, use_slurm: bool) -> bool:
        """Phase 3: MLM pretraining"""
        # Check if MLM models exist
        mlm_encoder = self.root / "data" / "models_output" / "encoder_pretrained.pth"
        mlm_decoder = self.root / "data" / "models_output" / "decoder_pretrained.pth"
        
        if mlm_encoder.exists() and mlm_decoder.exists():
            print_info("MLM models already exist")
            return True
        
        # Train MLM
        print_info("Training MLM pretraining...")
        return self.train_cmd.train_mlm_pretrain(dataset, use_slurm=use_slurm, test_mode=test_mode)
    
    def _phase4_tmr(self, dataset: str, test_mode: bool, use_slurm: bool) -> bool:
        """Phase 4: TMR training"""
        # Check if TMR model exists
        tmr_model = self.root / "data" / "models_output" / "model_all.pth"
        
        if tmr_model.exists():
            print_warning("TMR model exists but may be broken")
            if not confirm_action("Retrain TMR?", default=True):
                return True
        
        # Train TMR Stage 1
        print_info("Training TMR Stage 1...")
        if not self.train_cmd.train_tmr_stage1(dataset, use_slurm=use_slurm, test_mode=test_mode):
            print_error("TMR Stage 1 training failed")
            return False
        
        # Train TMR Stage 2
        print_info("Training TMR Stage 2...")
        if not self.train_cmd.train_tmr_stage2(dataset, use_slurm=use_slurm, test_mode=test_mode):
            print_error("TMR Stage 2 training failed")
            return False
        
        return True
    
    def _phase5_evaluation(self, dataset: str, test_mode: bool, use_slurm: bool) -> bool:
        """Phase 5: Comprehensive evaluation"""
        num_samples = 100 if test_mode else 1000
        
        # Evaluate baselines
        print_info("Evaluating baseline models...")
        self.eval_cmd.eval_all_baselines(dataset, num_samples, use_slurm)
        
        # Evaluate anonymization
        print_info("Evaluating anonymization models...")
        self.eval_cmd.eval_all_anonymization(dataset, num_samples, use_slurm)
        
        # Evaluate MLM
        print_info("Evaluating MLM...")
        self.eval_cmd.eval_mlm(dataset, num_samples, use_slurm)
        
        return True
    
    def _phase6_experiments(self, use_slurm: bool) -> bool:
        """Phase 6: Experiments and ablations"""
        # Masking ablations
        print_info("Running masking ablations...")
        self.exp_cmd.run_masking_ablation("temporal", use_slurm=use_slurm)
        self.exp_cmd.run_masking_ablation("spatial", use_slurm=use_slurm)
        
        # Teacher forcing experiment
        print_info("Running teacher forcing experiment...")
        self.exp_cmd.run_teacher_forcing_experiment(use_slurm=use_slurm)
        
        return True
    
    def run_quick_test(self, dataset: str = "ntu_cv"):
        """Run quick test pipeline (100 samples, local execution)"""
        print_header("Quick Test Pipeline")
        print_info("This will run a quick test with 100 samples locally")
        
        if not confirm_action("Continue?", default=True):
            return False
        
        # Create test subset
        print_section("Creating test subset...")
        self.data_cmd.create_test_subset(dataset, num_train=100, num_test=20)
        
        # Train one baseline
        print_section("Training Mixformer AR (test)...")
        self.train_cmd.train_mixformer(dataset, "ar", use_slurm=False, test_mode=True)
        
        # Evaluate
        print_section("Evaluating...")
        self.eval_cmd.eval_baseline("mixformer", dataset, "ar", num_samples=20, use_slurm=False)
        
        print_success("Quick test complete!")
        return True
    
    def run_baseline_pipeline(self, dataset: str = "ntu_cv", use_slurm: bool = True):
        """Run baseline-only pipeline"""
        print_header("Baseline Pipeline")
        print_info("This will train and evaluate all baseline models")
        
        if not confirm_action("Continue?", default=True):
            return False
        
        # Train baselines
        print_section("Training baselines...")
        self._phase2_baselines(dataset, test_mode=False, use_slurm=use_slurm)
        
        # Evaluate baselines
        print_section("Evaluating baselines...")
        self.eval_cmd.eval_all_baselines(dataset, 1000, use_slurm)
        
        print_success("Baseline pipeline complete!")
        return True
    
    def run_tmr_pipeline(self, dataset: str = "ntu_cv", use_slurm: bool = True):
        """Run TMR-only pipeline"""
        print_header("TMR Pipeline")
        print_info("This will train and evaluate TMR")
        
        if not confirm_action("Continue?", default=True):
            return False
        
        # MLM pretraining
        print_section("MLM pretraining...")
        self._phase3_mlm(dataset, test_mode=False, use_slurm=use_slurm)
        
        # TMR training
        print_section("TMR training...")
        self._phase4_tmr(dataset, test_mode=False, use_slurm=use_slurm)
        
        # TMR evaluation
        print_section("TMR evaluation...")
        self.eval_cmd.eval_anonymization("tmr", dataset, 1000, use_slurm)
        
        print_success("TMR pipeline complete!")
        return True
    
    def interactive_menu(self):
        """Interactive pipeline menu"""
        while True:
            print_header("Pipeline Orchestration")
            print("1. Run Full Pipeline (Everything)")
            print("2. Run Quick Test (100 samples, local)")
            print("3. Run Baseline Pipeline")
            print("4. Run TMR Pipeline")
            print("5. Run Evaluation Only")
            print("6. Run Experiments Only")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
                test_mode = confirm_action("Test mode?", default=False)
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_full_pipeline(dataset, test_mode, use_slurm)
            elif choice == "2":
                dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
                self.run_quick_test(dataset)
            elif choice == "3":
                dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_baseline_pipeline(dataset, use_slurm)
            elif choice == "4":
                dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
                use_slurm = confirm_action("Use SLURM?", default=True)
                self.run_tmr_pipeline(dataset, use_slurm)
            elif choice == "5":
                dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
                use_slurm = confirm_action("Use SLURM?", default=True)
                self._phase5_evaluation(dataset, False, use_slurm)
            elif choice == "6":
                use_slurm = confirm_action("Use SLURM?", default=True)
                self._phase6_experiments(use_slurm)
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")

