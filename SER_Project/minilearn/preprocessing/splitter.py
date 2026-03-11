"""
train_test_split — randomly partition arrays into train and test subsets.

Mirrors the scikit-learn signature for the common use case:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=0.2,
                                                         random_state=42,
                                                         stratify=y)
"""

import numpy as np


def train_test_split(*arrays, test_size=0.2, random_state=None, stratify=None):
    """
    Split one or more arrays into random train and test subsets
    """
    if not arrays:
        raise ValueError("At least one array is required.")

    n_samples = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same number of samples.")

    rng = np.random.default_rng(random_state)

    if stratify is not None:
        # Stratified split: sample each class proportionally
        labels = np.array(stratify)
        classes = np.unique(labels)
        train_idx, test_idx = [], []
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            n_test = max(1, int(np.ceil(len(cls_idx) * test_size)))
            test_idx.append(cls_idx[:n_test])
            train_idx.append(cls_idx[n_test:])
        train_idx = rng.permutation(np.concatenate(train_idx))
        test_idx = rng.permutation(np.concatenate(test_idx))
    else:
        indices = rng.permutation(n_samples)
        n_test = int(np.ceil(n_samples * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result
