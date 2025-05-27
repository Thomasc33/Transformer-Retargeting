"""
Experiment definitions for the Transformer Retargeting evaluation suite.
"""

try:
    from .primary import PrimaryExperiments
except ImportError:
    PrimaryExperiments = None

try:
    from .ablation import AblationExperiments
except ImportError:
    AblationExperiments = None

try:
    from .pretraining import PretrainingExperiments
except ImportError:
    PretrainingExperiments = None

try:
    from .robustness import RobustnessExperiments
except ImportError:
    RobustnessExperiments = None

try:
    from .efficiency import EfficiencyExperiments
except ImportError:
    EfficiencyExperiments = None

try:
    from .generalization import GeneralizationExperiments
except ImportError:
    GeneralizationExperiments = None

try:
    from .visualization import VisualizationExperiments
except ImportError:
    VisualizationExperiments = None

try:
    from .qualitative import QualitativeExperiments
except ImportError:
    QualitativeExperiments = None

__all__ = [
    'PrimaryExperiments',
    'AblationExperiments',
    'PretrainingExperiments',
    'RobustnessExperiments',
    'EfficiencyExperiments',
    'GeneralizationExperiments',
    'VisualizationExperiments',
    'QualitativeExperiments'
]
