"""Excess-Mass and Mass-Volume scores for unsupervised ML AD models."""

import importlib.metadata

from . import metrics
from .metrics import emmv_scores, excess_mass, mass_volume

__version__ = importlib.metadata.version('emmv')

__all__ = [
    'excess_mass',
    'emmv_scores',
    'mass_volume',
    'metrics',
]
