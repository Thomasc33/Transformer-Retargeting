"""
Compatibility shim: data has moved to src/data/datasets.py

Preferred imports:
    from src.data import datasets, load_data, get_cross_data

This shim will be removed in a future cleanup pass.
"""
from src.data.datasets import *  # noqa: F401,F403

if __name__ == '__main__':
    # Simple sanity: list available attributes
    import inspect
    mod = __import__('src.data.datasets', fromlist=['*'])
    names = [n for n,_ in inspect.getmembers(mod)]
    print("src.data.datasets exported symbols:", len(names))
