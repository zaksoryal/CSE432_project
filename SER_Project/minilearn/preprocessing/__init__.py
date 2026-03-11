"""
minilearn.preprocessing
=======================
Feature standardization and data-splitting utilities, implemented from
scratch using only NumPy.
"""

from .scaler import StandardScaler
from .splitter import train_test_split

__all__ = ["StandardScaler", "train_test_split"]
