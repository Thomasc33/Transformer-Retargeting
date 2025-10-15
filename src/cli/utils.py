"""
Shared utilities for TMR CLI
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")

def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(text)}{Colors.ENDC}")

def get_root_dir() -> Path:
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent.parent

def get_config_path() -> Path:
    """Get main config file path"""
    return get_root_dir() / "configs" / "main_config.yaml"

def get_jobs_file() -> Path:
    """Get jobs tracking file"""
    return get_root_dir() / "jobs.json"

def get_results_file() -> Path:
    """Get results tracking file"""
    return get_root_dir() / "results.json"

def load_jobs() -> Dict:
    """Load jobs tracking data"""
    jobs_file = get_jobs_file()
    if jobs_file.exists():
        with open(jobs_file, 'r') as f:
            return json.load(f)
    return {"jobs": [], "last_updated": None}

def save_jobs(jobs_data: Dict):
    """Save jobs tracking data"""
    jobs_data["last_updated"] = datetime.now().isoformat()
    with open(get_jobs_file(), 'w') as f:
        json.dump(jobs_data, f, indent=2)

def load_results() -> Dict:
    """Load results tracking data"""
    results_file = get_results_file()
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {
        "project": {
            "name": "Transformer Motion Retargeting",
            "description": "Privacy-preserving motion retargeting using transformers",
            "status": "retraining_required",
            "last_updated": ""
        },
        "models": {},
        "experiments": {},
        "visualizations": {},
        "status": {}
    }

def save_results(results_data: Dict):
    """Save results tracking data"""
    results_data["project"]["last_updated"] = datetime.now().isoformat()
    with open(get_results_file(), 'w') as f:
        json.dump(results_data, f, indent=2)

def run_command(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return result"""
    if cwd is None:
        cwd = get_root_dir()
    
    print_info(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, cwd=cwd)
    
    return result

def check_file_exists(path: Path, description: str = "File") -> bool:
    """Check if file exists and print status"""
    if path.exists():
        print_success(f"{description} found: {path}")
        return True
    else:
        print_error(f"{description} not found: {path}")
        return False

def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_dataset_path(dataset: str, subset: Optional[str] = None) -> Path:
    """Get path to dataset file"""
    root = get_root_dir()
    data_dir = root / "data"
    
    if subset:
        return data_dir / f"{dataset}_{subset}.pt"
    else:
        return data_dir / f"{dataset}_paired_comprehensive.pt"

def get_model_path(model_type: str, dataset: str, task: str, setting: str = "cview") -> Path:
    """Get path to model file"""
    root = get_root_dir()
    
    if model_type == "tmr":
        return root / "data" / "models_output" / "model_all.pth"
    elif model_type in ["pmr", "dmr"]:
        return root / "trained_models" / f"{model_type}_{dataset}_{setting}_final.pth"
    elif model_type in ["mixformer", "sgn"]:
        model_dir = f"{dataset}_{model_type}_{task}_{setting}"
        return root / "output" / model_dir / "model_best.pth.tar"
    else:
        return root / "output" / f"{model_type}_{dataset}_{task}_{setting}" / "model_best.pth.tar"

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"

def confirm_action(prompt: str, default: bool = False) -> bool:
    """Ask user to confirm an action"""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{Colors.YELLOW}❓ {prompt} [{default_str}]: {Colors.ENDC}").strip().lower()
    
    if not response:
        return default
    return response in ['y', 'yes']

def select_from_list(options: List[str], prompt: str = "Select an option") -> Optional[str]:
    """Let user select from a list of options"""
    print(f"\n{Colors.CYAN}{prompt}:{Colors.ENDC}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print(f"  0. Cancel")
    
    while True:
        try:
            choice = input(f"{Colors.YELLOW}Enter choice [0-{len(options)}]: {Colors.ENDC}").strip()
            choice_num = int(choice)
            if choice_num == 0:
                return None
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            print_error(f"Invalid choice. Please enter 0-{len(options)}")
        except ValueError:
            print_error("Invalid input. Please enter a number")
        except KeyboardInterrupt:
            print("\n")
            return None

def get_slurm_status(job_id: str) -> Optional[str]:
    """Get SLURM job status"""
    try:
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def parse_slurm_output(output: str) -> Optional[str]:
    """Parse SLURM job ID from sbatch output"""
    # Expected format: "Submitted batch job 12345678"
    if "Submitted batch job" in output:
        parts = output.strip().split()
        if len(parts) >= 4:
            return parts[-1]
    return None

def create_sample_subset(input_file: Path, output_file: Path, num_samples: int = 100):
    """Create a small subset of data for testing"""
    import torch
    
    print_info(f"Creating {num_samples}-sample subset from {input_file.name}")
    
    data = torch.load(input_file)
    
    if isinstance(data, list):
        subset = data[:num_samples]
    elif isinstance(data, dict):
        subset = {k: v[:num_samples] if isinstance(v, (list, torch.Tensor)) else v 
                  for k, v in data.items()}
    else:
        print_error(f"Unknown data format: {type(data)}")
        return False
    
    torch.save(subset, output_file)
    print_success(f"Created subset: {output_file}")
    return True

def get_available_gpus() -> int:
    """Get number of available GPUs"""
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0

def estimate_training_time(num_samples: int, num_epochs: int, batch_size: int = 32, num_gpus: int = 1) -> float:
    """Estimate training time in hours"""
    # Rough estimate: 0.1 seconds per sample per epoch on 1 GPU
    time_per_sample = 0.1 / num_gpus
    total_time = (num_samples * num_epochs * time_per_sample) / 3600
    return total_time

def validate_environment() -> Dict[str, bool]:
    """Validate environment setup"""
    checks = {}
    
    # Check Python version
    checks['python'] = sys.version_info >= (3, 7)
    
    # Check PyTorch
    try:
        import torch
        checks['pytorch'] = True
        checks['cuda'] = torch.cuda.is_available()
    except ImportError:
        checks['pytorch'] = False
        checks['cuda'] = False
    
    # Check data directory
    checks['data_dir'] = (get_root_dir() / "data").exists()
    
    # Check config file
    checks['config'] = get_config_path().exists()
    
    return checks

