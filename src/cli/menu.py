"""
Interactive Menu System for TMR
Main entry point for interactive mode
"""

import sys
from pathlib import Path

from .utils import (
    print_header, print_success, print_error, print_warning, print_info, print_section,
    Colors
)
from .slurm_manager import SlurmManager
from .data_commands import DataCommands
from .train_commands import TrainCommands
from .eval_commands import EvalCommands
from .experiment_commands import ExperimentCommands
from .repo_manager import RepoManager
from .pipeline import Pipeline


class InteractiveMenu:
    """Main interactive menu system"""
    
    def __init__(self):
        self.slurm = SlurmManager()
        self.data_cmd = DataCommands(self.slurm)
        self.train_cmd = TrainCommands(self.slurm)
        self.eval_cmd = EvalCommands(self.slurm)
        self.exp_cmd = ExperimentCommands(self.slurm)
        self.repo_mgr = RepoManager()
        self.pipeline = Pipeline(self.slurm)
    
    def show_banner(self):
        """Show welcome banner"""
        banner = f"""
{Colors.HEADER}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           Transformer Motion Retargeting (TMR)                ║
║                  Command-Line Interface                       ║
║                                                               ║
║                    Version 2.0 - 2025                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.BLUE}Welcome to the TMR CLI!{Colors.ENDC}
This is your one-stop interface for all TMR operations.

{Colors.GREEN}Quick Start:{Colors.ENDC}
  • First time? Try option 9 (Repository Status)
  • Want to run everything? Try option 7 (Full Pipeline)
  • Need help? Check README.md and STATUS_REPORT.md
"""
        print(banner)
    
    def main_menu(self):
        """Display main menu"""
        while True:
            print()
            print_header("TMR Main Menu")
            print()
            print(f"{Colors.BLUE}═══ Data Management ═══{Colors.ENDC}")
            print("  1. Data Operations (preprocess, validate, statistics)")
            print()
            print(f"{Colors.BLUE}═══ Training ═══{Colors.ENDC}")
            print("  2. Training Operations (MLM, TMR, Mixformer, SGN, PMR, DMR)")
            print()
            print(f"{Colors.BLUE}═══ Evaluation ═══{Colors.ENDC}")
            print("  3. Evaluation Operations (baselines, anonymization, MLM)")
            print()
            print(f"{Colors.BLUE}═══ Experiments ═══{Colors.ENDC}")
            print("  4. Experiments & Ablations (masking, loss, architecture)")
            print()
            print(f"{Colors.BLUE}═══ Pipeline ═══{Colors.ENDC}")
            print("  5. Pipeline Orchestration (run everything)")
            print()
            print(f"{Colors.BLUE}═══ Repository ═══{Colors.ENDC}")
            print("  6. Repository Management (check structure, models, data)")
            print()
            print(f"{Colors.BLUE}═══ SLURM ═══{Colors.ENDC}")
            print("  7. SLURM Job Management (list, cancel, status)")
            print()
            print(f"{Colors.BLUE}═══ Quick Actions ═══{Colors.ENDC}")
            print("  8. Quick Test (100 samples, local)")
            print("  9. Repository Status")
            print(" 10. Open Dashboard (index.html)")
            print()
            print(f"{Colors.RED}  0. Exit{Colors.ENDC}")
            print()
            
            choice = input(f"{Colors.YELLOW}Enter choice: {Colors.ENDC}").strip()
            
            if choice == "0":
                print_info("Goodbye!")
                sys.exit(0)
            elif choice == "1":
                self.data_cmd.interactive_menu()
            elif choice == "2":
                self.train_cmd.interactive_menu()
            elif choice == "3":
                self.eval_cmd.interactive_menu()
            elif choice == "4":
                self.exp_cmd.interactive_menu()
            elif choice == "5":
                self.pipeline.interactive_menu()
            elif choice == "6":
                self.repo_mgr.interactive_menu()
            elif choice == "7":
                self.slurm_menu()
            elif choice == "8":
                self.quick_test()
            elif choice == "9":
                self.repo_mgr.show_status()
                input("\nPress Enter to continue...")
            elif choice == "10":
                self.open_dashboard()
            else:
                print_error("Invalid choice")
    
    def slurm_menu(self):
        """SLURM job management menu"""
        while True:
            print()
            print_header("SLURM Job Management")
            print("1. List tracked jobs")
            print("2. Check job status")
            print("3. Cancel job")
            print("4. Show queue status")
            print("0. Back to main menu")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                status_filter = input("Filter by status (RUNNING/PENDING/COMPLETED/FAILED) [all]: ").strip()
                self.slurm.list_jobs(status_filter if status_filter else None)
            elif choice == "2":
                job_id = input("Enter job ID: ").strip()
                if job_id:
                    status = self.slurm.check_job_status(job_id)
                    if status:
                        print_info(f"Job {job_id} status: {status}")
                    else:
                        print_warning(f"Job {job_id} not found")
            elif choice == "3":
                job_id = input("Enter job ID to cancel: ").strip()
                if job_id:
                    if self.slurm.cancel_job(job_id):
                        print_success(f"Job {job_id} cancelled")
                    else:
                        print_error(f"Failed to cancel job {job_id}")
            elif choice == "4":
                self.slurm.get_queue_status()
            else:
                print_error("Invalid choice")
            
            input("\nPress Enter to continue...")
    
    def quick_test(self):
        """Run quick test"""
        print_header("Quick Test")
        print_info("This will run a quick test with 100 samples locally")
        print_warning("This is for testing purposes only")
        
        dataset = input("Dataset [ntu_cv]: ").strip() or "ntu_cv"
        self.pipeline.run_quick_test(dataset)
        
        input("\nPress Enter to continue...")
    
    def open_dashboard(self):
        """Open results dashboard"""
        from pathlib import Path
        import webbrowser
        
        dashboard_path = Path(__file__).parent.parent.parent / "index.html"
        if dashboard_path.exists():
            print_info(f"Opening dashboard: {dashboard_path}")
            webbrowser.open(f"file://{dashboard_path.absolute()}")
            print_success("Dashboard opened in browser")
        else:
            print_error(f"Dashboard not found: {dashboard_path}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run interactive menu"""
        try:
            self.show_banner()
            self.main_menu()
        except KeyboardInterrupt:
            print()
            print_info("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def run_interactive():
    """Entry point for interactive mode"""
    menu = InteractiveMenu()
    menu.run()

