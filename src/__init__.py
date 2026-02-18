"""
Medical Healthcare Assistant - Source Package

This package contains utilities for fine-tuning and deploying a medical LLM assistant.
"""

__version__ = "1.0.0"
__author__ = "ML Assignment Team"

from . import data_preprocessing
from . import model_training
from . import evaluation
from . import inference

__all__ = [
    'data_preprocessing',
    'model_training',
    'evaluation',
    'inference'
]
