"""Public API for hierarchical uncertainty-aware spatial allocation."""

from ._objective import Instance
from ._synthetic import make_synthetic
from .instances import load_instance, save_instance
from .solver import Allocation, load_example_policy, load_policy, save_policy, solve
from .training import train_policy

__all__ = [
    "Allocation",
    "Instance",
    "load_example_policy",
    "load_instance",
    "load_policy",
    "make_synthetic",
    "save_instance",
    "save_policy",
    "solve",
    "train_policy",
]
