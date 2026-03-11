"""
Logistic Regression — implemented from scratch using NumPy.

Binary case
-----------
The model learns a weight vector w and bias b such that:
    z = X @ w + b
    p = sigmoid(z) = 1 / (1 + exp(-z))

The binary cross-entropy (log-loss) is minimised via batch gradient descent:
    dL/dw = (1/n) * X^T (p - y)
    dL/db = (1/n) * sum(p - y)

Multi-class case
----------------
For K > 2 classes, one-vs-rest (OvR) is used: K binary classifiers are
trained, one per class. At prediction time the class with the highest
sigmoid score is returned.

An L2 regularisation term λ||w||² can be added to the loss to reduce
overfitting (λ = 0 disables it).
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters
    ----------
    learning_rate : float, default 0.1
        Gradient-descent step size.
    n_iterations : int, default 1000
        Number of gradient-descent updates.
    lambda_ : float, default 0.0
        L2 regularisation strength (0 = no regularisation).
    fit_intercept : bool, default True
        Whether to learn a bias term.
    tol : float, default 1e-6
        Early stopping: if the change in loss across consecutive iterations
        is below this threshold, training stops.
    verbose : bool, default False
        Print loss every 100 iterations.

    Attributes
    ----------
    classes_ : ndarray
        Sorted unique class labels seen during fit.
    coef_ : ndarray, shape (n_classes_binary, n_features)
        Learned weight vectors (one row per OvR binary classifier).
    intercept_ : ndarray, shape (n_classes_binary,)
        Learned bias terms.
    loss_history_ : list of float
        Mean loss per iteration across all OvR classifiers.
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=1000,
        lambda_=0.0,
        fit_intercept=True,
        tol=1e-6,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.verbose = verbose

        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    # Static helpers

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid: clips z to [-500, 500] before exp."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y, p):
        """Binary cross-entropy: -[y log(p) + (1-y) log(1-p)], clipped for stability."""
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Core training for one OvR binary problem

    def _fit_binary(self, X, y_bin):
        """
        Fit one binary classifier (class=1 vs rest) using gradient descent.

        Returns (weights, bias, loss_history).
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        losses = []
        prev_loss = np.inf

        for iteration in range(self.n_iterations):
            z = X @ w + b
            p = self._sigmoid(z)
            error = p - y_bin                          # shape (n_samples,)

            # Gradients (with optional L2 regularisation on weights, not bias)
            dw = (1 / n_samples) * (X.T @ error) + (self.lambda_ / n_samples) * w
            db = (1 / n_samples) * error.sum()

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            # Loss (without regularisation term for comparability)
            loss = self._binary_cross_entropy(y_bin, p)
            losses.append(loss)

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"  iter {iteration + 1:5d}  loss={loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return w, b, losses

    # Public API

    def fit(self, X, y):
        """
        Train the model using one-vs-rest logistic regression.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)  — integer or string class labels
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)

        self.coef_ = np.zeros((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        all_losses = []

        for i, cls in enumerate(self.classes_):
            y_bin = (y == cls).astype(float)          # 1 for this class, 0 for all others
            w, b, losses = self._fit_binary(X, y_bin)
            self.coef_[i] = w
            self.intercept_[i] = b
            all_losses.append(losses)

        # Store mean loss across classifiers per iteration
        min_len = min(len(l) for l in all_losses)
        self.loss_history_ = [
            np.mean([all_losses[c][t] for c in range(n_classes)])
            for t in range(min_len)
        ]
        return self

    def predict_proba(self, X):
        """
        Probability estimates for each class (OvR, not calibrated softmax).

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Raw sigmoid scores. Rows do NOT necessarily sum to 1.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.array(X, dtype=float)
        scores = X @ self.coef_.T + self.intercept_   # (n_samples, n_classes)
        return self._sigmoid(scores)

    def predict(self, X):
        """
        Predict class labels for X.

        Assigns each sample to the class with the highest OvR sigmoid score.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        """Accuracy on (X, y)."""
        from minilearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
