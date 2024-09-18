"""
Compatibility shim for loading pickled data files.

This file exists ONLY to support loading legacy pickled data files that were
created when Cross_Data was defined in the root-level data.py module.

When torch.load() unpickles these files, it tries to import 'data.Cross_Data'.
This shim re-exports Cross_Data from its new location (src.data.datasets) so
that unpickling works correctly.

DO NOT REMOVE THIS FILE unless you regenerate all pickled data files.
"""

from src.data.datasets import Cross_Data

__all__ = ['Cross_Data']

