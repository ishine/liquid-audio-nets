"""Liquid Audio Neural Networks - Edge-efficient audio processing.

This package provides Liquid Neural Network implementations optimized for
ultra-low-power audio processing on edge devices.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.dev"

from typing import Optional

try:
    from ._core import *  # Rust/C++ core bindings
except ImportError:
    # Fallback for development without compiled bindings
    pass

from .lnn import LNN, AdaptiveConfig
from .training import LNNTrainer
from .tools import profiler, compression

__all__ = [
    "LNN",
    "AdaptiveConfig", 
    "LNNTrainer",
    "profiler",
    "compression",
]