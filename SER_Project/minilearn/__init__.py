"""
MiniLearn — A minimal scikit-learn-style machine learning library.

Built from scratch for educational purposes as part of the CSE 432/532
Speech Emotion Recognition project.

Subpackages
-----------
minilearn.preprocessing   — StandardScaler, train_test_split
minilearn.metrics         — regression and classification metrics, k-fold CV
minilearn.regression      — LinearRegression (normal equations + gradient descent)
minilearn.classifiers     — LogisticRegression (and more to come)

Usage examples
--------------
    from minilearn.preprocessing import StandardScaler, train_test_split
    from minilearn.regression import LinearRegression
    from minilearn.classifiers import LogisticRegression
    from minilearn.metrics import r2_score, accuracy_score, k_fold_cv
"""

__version__ = "0.2.0"
