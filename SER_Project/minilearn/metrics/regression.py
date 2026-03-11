"""
Regression evaluation metrics.

All functions accept 1-D array-likes y_true and y_pred.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error: MSE = (1/n) * sum( (y_true - y_pred)^2 )

    Penalizes large errors more than small ones due to squaring.
    Lower is better; 0 means perfect prediction.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error: RMSE = sqrt(MSE)

    Has the same units as the target variable, making it easier to interpret.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error: MAE = (1/n) * sum( |y_true - y_pred| )

    Robust to outliers relative to MSE; interpretable in the original units.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Coefficient of Determination: R² = 1 - SS_res / SS_tot

    Fraction of variance in y_true explained by the model.
    R² = 1  → perfect fit
    R² = 0  → model no better than predicting the mean
    R² < 0  → model worse than predicting the mean
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot
