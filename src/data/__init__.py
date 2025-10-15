"""src.data package

- Re-exports legacy datasets and helpers from src.data.datasets
- Preferred import going forward: `from src.data import datasets, load_data, get_cross_data`
"""
from .datasets import *  # noqa: F401,F403

