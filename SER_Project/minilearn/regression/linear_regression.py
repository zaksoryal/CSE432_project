"""
Linear Regression — implemented from scratch using NumPy.

Two solvers are provided:
1. 'normal_equation' (default) — closed-form solution via the normal equations:
       w = (X^T X)^{-1} X^T y
   Fast and exact for moderate n_features, but may be ill-conditioned when
   features are highly correlated or n_features is very large.

2. 'gradient_descent' — iterative batch gradient descent:
       w := w - lr * (1/n) * X^T (X w - y)
   Useful for understanding the optimization perspective and for large datasets
   where the normal equation becomes too expensive to invert.

Both solvers fit an intercept term by augmenting X with a column of ones.
"""

import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares linear regression.

    Parameters
    ----------
    solver : {'normal_equation', 'gradient_descent'}
        Optimisation strategy. Default: 'normal_equation'.
    learning_rate : float
        Step size for gradient descent. Ignored for normal equation.
    n_iterations : int
        Number of gradient-descent steps. Ignored for normal equation.
    fit_intercept : bool
        Whether to add a bias/intercept term. Default True.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated regression coefficients (excludes intercept).
    intercept_ : float
        Estimated intercept (bias) term.
    loss_history_ : list of float
        MSE per iteration — only populated when solver='gradient_descent'.
    """

    def __init__(
        self,
        solver="normal_equation",
        learning_rate=0.01,
        n_iterations=1000,
        fit_intercept=True,
    ):
        if solver not in ("normal_equation", "gradient_descent"):
            raise ValueError("solver must be 'normal_equation' or 'gradient_descent'.")
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_intercept(self, X):
        """Prepend a column of ones to X for the intercept term."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()

        if self.fit_intercept:
            X_aug = self._add_intercept(X)
        else:
            X_aug = X

        if self.solver == "normal_equation":
            self._fit_normal(X_aug, y)
        else:
            self._fit_gradient_descent(X_aug, y)

        # Separate intercept from coefficients
        if self.fit_intercept:
            self.intercept_ = self._weights[0]
            self.coef_ = self._weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights

        return self

    def _fit_normal(self, X, y):
        """
        Closed-form solution:  w = (X^T X)^{-1} X^T y

        np.linalg.lstsq is used instead of explicit inversion to handle
        near-singular matrices more gracefully (it computes the minimum-norm
        least-squares solution via SVD internally).
        """
        # lstsq solves ||X w - y||^2 in the least-squares sense
        self._weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    def _fit_gradient_descent(self, X, y):
        """
        Batch gradient descent update rule:
            gradient = (1/n) * X^T (X w - y)
            w := w - lr * gradient
        """
        n_samples, n_params = X.shape
        self._weights = np.zeros(n_params)
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            y_pred = X @ self._weights
            residuals = y_pred - y
            gradient = (1 / n_samples) * X.T @ residuals
            self._weights -= self.learning_rate * gradient
            mse = np.mean(residuals ** 2)
            self.loss_history_.append(mse)

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return the R² coefficient of determination on (X, y)."""
        from minilearn.metrics import r2_score
        return r2_score(y, self.predict(X))
