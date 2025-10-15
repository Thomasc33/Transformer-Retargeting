"""
Analysis and visualization modules for the evaluation suite.
"""

try:
    from .visualizer import ComprehensiveVisualizer
except ImportError:
    ComprehensiveVisualizer = None

try:
    from .comparator import ResultComparator
except ImportError:
    ResultComparator = None

__all__ = ['ComprehensiveVisualizer', 'ResultComparator']
