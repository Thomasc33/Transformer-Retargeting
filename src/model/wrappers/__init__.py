"""
Model wrappers for external SOTA action recognition models.

This module provides wrapper classes that import models directly from
external_repo_for_reference directories and adapt them to our unified interface.
"""

from .base_wrapper import BaseModelWrapper
from .ctrgcn_wrapper import CTRGCNWrapper
from .infogcn_wrapper import InfoGCNWrapper
from .skateformer_wrapper import SkateFormerWrapper
from .hdgcn_wrapper import HDGCNWrapper
from .mamp_wrapper import MAMPWrapper
from .hgformer_wrapper import HGformerWrapper

__all__ = [
    'BaseModelWrapper',
    'CTRGCNWrapper',
    'InfoGCNWrapper',
    'SkateFormerWrapper',
    'HDGCNWrapper',
    'MAMPWrapper',
    'HGformerWrapper',
]
