"""
TMR CLI Package
Comprehensive command-line interface for TMR project
"""

from .utils import *
from .slurm_manager import SlurmManager
from .data_commands import DataCommands
from .train_commands import TrainCommands
from .eval_commands import EvalCommands
from .experiment_commands import ExperimentCommands
from .pipeline import Pipeline
from .repo_manager import RepoManager
from .menu import InteractiveMenu, run_interactive

__all__ = [
    'SlurmManager',
    'DataCommands',
    'TrainCommands',
    'EvalCommands',
    'ExperimentCommands',
    'Pipeline',
    'RepoManager',
    'InteractiveMenu',
    'run_interactive',
]

