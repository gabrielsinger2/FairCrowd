import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as f1,
)
from ..core.metrics import MatrixLike

safe_divide = lambda num, den: num / den if den != 0 and np.isfinite(den) else np.nan


def accuracy(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return accuracy_score(y_true, y_pred)


def precision(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return precision_score(y_true, y_pred) if y_pred.sum() > 0 else np.nan


def recall(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return recall_score(y_true, y_pred) if y_true.sum() > 0 else np.nan


def f1_score(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return f1(y_true, y_pred) if y_true.sum() > 0 and y_pred.sum() > 0 else np.nan


def false_positive_rate(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return safe_divide(((y_true == 0) & (y_pred == 1)).sum(), len(y_true[y_true == 0]))


def false_negative_rate(y_true: MatrixLike, y_pred: MatrixLike) -> float:
    return safe_divide(((y_true == 1) & (y_pred == 0)).sum(), len(y_true[y_true == 1]))
