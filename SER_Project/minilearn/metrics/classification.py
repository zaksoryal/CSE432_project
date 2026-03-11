"""
Classification evaluation metrics — all from scratch with NumPy.

Supports binary and multi-class targets via the 'macro' and 'weighted'
averaging strategies.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Fraction of correctly classified samples."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix C where C[i, j] is the number of samples
    with true label i predicted as label j.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    labels = np.array(labels)
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def _per_class_precision_recall_f1(y_true, y_pred, labels):
    """Return arrays of per-class precision, recall, f1, and support."""
    cm, _ = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = cm.sum(axis=1)           # row sums = true counts per class
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp      # predicted as i but not i
        fn = cm[i, :].sum() - tp      # actually i but predicted as other
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0
    return precision, recall, f1, support


def precision_score(y_true, y_pred, average="macro", labels=None):
    """
    Precision for multi-class classification.

    average : 'macro' (unweighted mean across classes)
              'weighted' (mean weighted by class support)
              'per_class' (returns array)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    prec, _, _, support = _per_class_precision_recall_f1(y_true, y_pred, labels)
    if average == "per_class":
        return prec
    elif average == "weighted":
        return np.average(prec, weights=support)
    else:  # macro
        return prec.mean()


def recall_score(y_true, y_pred, average="macro", labels=None):
    """Recall for multi-class classification. Same averaging options as precision_score."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    _, rec, _, support = _per_class_precision_recall_f1(y_true, y_pred, labels)
    if average == "per_class":
        return rec
    elif average == "weighted":
        return np.average(rec, weights=support)
    else:
        return rec.mean()


def f1_score(y_true, y_pred, average="macro", labels=None):
    """F1 score for multi-class classification. Same averaging options as precision_score."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    _, _, f1, support = _per_class_precision_recall_f1(y_true, y_pred, labels)
    if average == "per_class":
        return f1
    elif average == "weighted":
        return np.average(f1, weights=support)
    else:
        return f1.mean()


def classification_report(y_true, y_pred, target_names=None):
    """
    Build a text summary of per-class and averaged metrics.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    prec, rec, f1, support = _per_class_precision_recall_f1(y_true, y_pred, labels)

    if target_names is None:
        target_names = [str(lbl) for lbl in labels]

    header = f"{'':>15s}  {'precision':>9s}  {'recall':>9s}  {'f1-score':>9s}  {'support':>9s}\n\n"
    rows = ""
    for i, name in enumerate(target_names):
        rows += (
            f"{name:>15s}  {prec[i]:>9.4f}  {rec[i]:>9.4f}  {f1[i]:>9.4f}  {support[i]:>9d}\n"
        )

    n_total = support.sum()
    acc = accuracy_score(y_true, y_pred)
    macro_p = prec.mean()
    macro_r = rec.mean()
    macro_f = f1.mean()
    weighted_p = np.average(prec, weights=support)
    weighted_r = np.average(rec, weights=support)
    weighted_f = np.average(f1, weights=support)

    rows += f"\n{'accuracy':>15s}  {'':>9s}  {'':>9s}  {acc:>9.4f}  {n_total:>9d}\n"
    rows += f"{'macro avg':>15s}  {macro_p:>9.4f}  {macro_r:>9.4f}  {macro_f:>9.4f}  {n_total:>9d}\n"
    rows += f"{'weighted avg':>15s}  {weighted_p:>9.4f}  {weighted_r:>9.4f}  {weighted_f:>9.4f}  {n_total:>9d}\n"

    return header + rows


def k_fold_cv(model, X, y, k=5, random_state=None, stratify=True):
    """
    Stratified (or regular) k-fold cross-validation.
    """
    import copy

    X = np.array(X, dtype=float)
    y = np.array(y)
    n = len(y)
    rng = np.random.default_rng(random_state)

    if stratify:
        # Build fold indices preserving class proportions
        classes = np.unique(y)
        fold_indices = [[] for _ in range(k)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            cls_idx = rng.permutation(cls_idx)
            for fold_i, chunk in enumerate(np.array_split(cls_idx, k)):
                fold_indices[fold_i].extend(chunk.tolist())
        fold_indices = [np.array(fi) for fi in fold_indices]
    else:
        all_idx = rng.permutation(n)
        fold_indices = [chunk for chunk in np.array_split(all_idx, k)]

    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold_i in range(k):
        test_idx = fold_indices[fold_i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != fold_i])

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        scores["accuracy"].append(accuracy_score(y_te, y_pred))
        scores["precision"].append(precision_score(y_te, y_pred, average="macro"))
        scores["recall"].append(recall_score(y_te, y_pred, average="macro"))
        scores["f1"].append(f1_score(y_te, y_pred, average="macro"))

    return scores
