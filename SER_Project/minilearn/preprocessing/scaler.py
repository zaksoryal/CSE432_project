"""
StandardScaler — z-score normalization fitted on training data only.

For each feature j, the transformation is:
    x_scaled = (x - mean_j) / std_j

where mean_j and std_j are computed from the training set.
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Fit on training data only; use the stored mean/std to transform any
    subsequent dataset (train, validation, test) to prevent data leakage
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Compute mean and std from training data X (shape: n_samples × n_features)."""
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero: features with zero variance keep their value
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """Apply z-score scaling using the fitted mean and std."""
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.array(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Convenience: fit on X then return the scaled version."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse the scaling: x = x_scaled * std + mean."""
        if self.mean_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        return np.array(X_scaled, dtype=float) * self.std_ + self.mean_
