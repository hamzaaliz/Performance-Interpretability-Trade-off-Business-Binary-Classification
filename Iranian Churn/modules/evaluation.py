import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CALIBRATION_EPSILON, RECALL_FLOOR


def positive_scores(estimator, X):
    return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)


def calibration_statistics(y_true, y_score):
    clipped = np.clip(np.asarray(y_score, dtype=float), CALIBRATION_EPSILON, 1 - CALIBRATION_EPSILON)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(logits, y_true)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Calibration regression did not converge")
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def reliability_bins(y_true, y_score, n_bins=10):
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_score": np.asarray(y_score)})
    frame["bin"] = pd.qcut(frame["y_score"], q=n_bins, duplicates="drop")
    result = (
        frame.groupby("bin", observed=True)
        .agg(mean_score=("y_score", "mean"), observed_rate=("y_true", "mean"), n=("y_true", "size"))
        .reset_index(drop=True)
    )
    result.insert(0, "bin", np.arange(1, len(result) + 1))
    return result


def metrics_at_threshold(y_true, y_score, threshold):
    pred = (np.asarray(y_score, dtype=float) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "predicted_positive": int(pred.sum()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_table(y_true, y_score):
    score = np.asarray(y_score, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if not np.isfinite(score).all():
        raise RuntimeError("Threshold input contains non-finite scores")
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    group_ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(score) - 1]
    predicted_positive = group_ends + 1
    tp = np.cumsum(sorted_y)[group_ends]
    fp = predicted_positive - tp
    total_positive = int(y.sum())
    total_negative = int(len(y) - total_positive)
    frame = pd.DataFrame(
        {
            "threshold": sorted_score[group_ends],
            "predicted_positive": predicted_positive,
            "precision": tp / predicted_positive,
            "recall": tp / total_positive,
            "tp": tp,
            "fp": fp,
            "tn": total_negative - fp,
            "fn": total_positive - tp,
        }
    )
    return frame


def select_threshold(table, recall_floor=RECALL_FLOOR):
    eligible = table.loc[table["recall"] >= recall_floor].copy()
    if not eligible.empty:
        selected = eligible.sort_values(
            ["precision", "recall", "threshold"], ascending=[False, False, False]
        ).iloc[0].to_dict()
        selected.update({"floor_attained": True, "selection_mode": "constrained"})
        return selected
    selected = table.sort_values(
        ["recall", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0].to_dict()
    selected.update({"floor_attained": False, "selection_mode": "max_recall_fallback"})
    return selected


def discrimination_metrics(y_true, y_score):
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
