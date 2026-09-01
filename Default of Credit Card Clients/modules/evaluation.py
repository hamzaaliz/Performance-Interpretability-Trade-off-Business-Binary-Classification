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
    clipped = np.clip(
        np.asarray(y_score, dtype=float),
        CALIBRATION_EPSILON,
        1 - CALIBRATION_EPSILON,
    )
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
        .agg(
            mean_score=("y_score", "mean"),
            observed_rate=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index(drop=True)
    )
    result.insert(0, "bin", np.arange(1, len(result) + 1))
    return result


def threshold_table(y_true, y_score):
    candidates = np.unique(np.concatenate(([0.0], np.asarray(y_score, dtype=float), [1.0])))
    rows = []
    for threshold in candidates:
        pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table, recall_floor=RECALL_FLOOR):
    eligible = table.loc[table["recall"] >= recall_floor].copy()
    if eligible.empty:
        raise RuntimeError(f"No threshold attains recall floor {recall_floor}")
    eligible = eligible.sort_values(
        ["precision", "recall", "threshold"], ascending=[False, False, False]
    )
    return eligible.iloc[0].to_dict()


def metrics_at_threshold(y_true, y_score, threshold):
    pred = (np.asarray(y_score) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def discrimination_metrics(y_true, y_score):
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
