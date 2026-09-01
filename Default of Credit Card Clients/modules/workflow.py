import hashlib
import importlib.util
import itertools
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import shap
import sklearn
import xgboost
from scipy.stats import spearmanr

from config import (
    BILL_FEATURES,
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_BINS,
    CODE_VERSION_LABEL,
    DATA_PATH,
    DISPLAY_NAMES,
    INNER_FOLDS,
    LR_PRUNED_FEATURES,
    LR_USED_FEATURES,
    MODEL_DISPLAY,
    MODEL_ORDER,
    NOMINAL,
    ORDINAL,
    OUTER_FOLDS,
    PAYMENT_FEATURES,
    PREDICTORS,
    PROJECT_ROOT,
    RAW_COLUMNS,
    RAW_TO_FEATURE,
    RECALL_FLOOR,
    ROOT,
    SEARCH_ITERATIONS,
    SEED,
    SHARED_WORKFLOW_PATH,
    TARGET_RAW,
    TEST_SIZE,
)
from modules import model_dt, model_lr, model_rf, model_xgb
from modules.evaluation import calibration_statistics, reliability_bins
from modules.interpretability import run_interpretability
from modules.preprocessing import apply_locked_rulings


_shared_spec = importlib.util.spec_from_file_location("_shared_sgc_workflow", SHARED_WORKFLOW_PATH)
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)

MODEL_MODULES = {"lr": model_lr, "dt": model_dt, "rf": model_rf, "xgb": model_xgb}
_shared.MODEL_MODULES = MODEL_MODULES


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version():
    paths = [ROOT / "config.py", ROOT / "run_workflow.py"]
    paths.extend(sorted((ROOT / "modules").glob("*.py")))
    paths.extend(
        [
            PROJECT_ROOT / "South German Credit" / "modules" / "workflow.py",
            PROJECT_ROOT / "South German Credit" / "modules" / "interpretability.py",
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return f"{CODE_VERSION_LABEL}:{digest.hexdigest()[:16]}"


def _duplicate_profile_statistics(raw):
    feature_columns = [column for column in RAW_COLUMNS if column not in {"ID", TARGET_RAW}]
    feature_target = raw.drop(columns="ID")
    feature_only = raw[feature_columns]
    feature_target_repeated = feature_target.duplicated(keep=False)
    grouped = raw.groupby(feature_columns, dropna=False, sort=False)
    sizes = grouped.size()
    repeated_groups = sizes[sizes > 1]
    conflicts = grouped[TARGET_RAW].nunique().gt(1)
    return {
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "redundant_feature_target_copies": int(feature_target.duplicated().sum()),
        "feature_target_repeated_rows": int(feature_target_repeated.sum()),
        "redundant_feature_only_copies": int(feature_only.duplicated().sum()),
        "repeated_feature_groups": int(len(repeated_groups)),
        "repeated_feature_rows": int(repeated_groups.sum()),
        "conflicting_label_groups": int(conflicts.sum()),
    }


def _load_and_assert():
    raw = pd.read_csv(DATA_PATH)
    if list(raw.columns) != RAW_COLUMNS:
        raise RuntimeError("Source column list differs from the locked specification")
    if raw.shape != (30000, 25):
        raise RuntimeError(f"Unexpected shape: {raw.shape}")
    if _sha256(DATA_PATH) != "4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb":
        raise RuntimeError("Source SHA-256 mismatch")
    if not all(str(dtype) == "int64" for dtype in raw.dtypes):
        raise RuntimeError("All source columns must parse as int64")
    if int(raw.isna().sum().sum()) != 0:
        raise RuntimeError("Unexpected missing values")
    if raw["ID"].nunique() != 30000 or raw["ID"].tolist() != list(range(1, 30001)):
        raise RuntimeError("Identifier assertion failed")
    if set(raw[TARGET_RAW]) != {0, 1} or int(raw[TARGET_RAW].sum()) != 6636:
        raise RuntimeError("Target polarity or count assertion failed")
    if (raw[[f"PAY_AMT{i}" for i in range(1, 7)]] < 0).any().any():
        raise RuntimeError("Unexpected negative payment amount")

    duplicate_stats = _duplicate_profile_statistics(raw)
    expected_duplicates = {
        "exact_duplicate_rows": 0,
        "redundant_feature_target_copies": 35,
        "feature_target_repeated_rows": 70,
        "redundant_feature_only_copies": 56,
        "repeated_feature_groups": 52,
        "repeated_feature_rows": 108,
        "conflicting_label_groups": 21,
    }
    if duplicate_stats != expected_duplicates:
        raise RuntimeError(f"Repeated-profile assertions failed: {duplicate_stats}")

    expected_code_sets = {
        "PAY_0": set(range(-2, 9)),
        "PAY_2": set(range(-2, 9)),
        "PAY_3": set(range(-2, 9)),
        "PAY_4": set(range(-2, 9)),
        "PAY_5": {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8},
        "PAY_6": {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8},
    }
    for column, expected in expected_code_sets.items():
        if set(raw[column]) != expected or 9 in set(raw[column]):
            raise RuntimeError(f"Repayment-status code assertion failed for {column}")

    off_grid = raw.loc[raw["LIMIT_BAL"] % 10000 != 0, ["ID", "LIMIT_BAL"]]
    expected_off_grid = [(5477, 16000), (12526, 327680), (27004, 16000)]
    if list(off_grid.itertuples(index=False, name=None)) != expected_off_grid:
        raise RuntimeError(f"Off-grid credit-limit assertion failed: {off_grid.to_dict('records')}")

    y = raw[TARGET_RAW].astype(int).copy()
    X = apply_locked_rulings(raw.drop(columns=["ID", TARGET_RAW]))
    if X.shape != (30000, 23) or list(X.columns) != PREDICTORS:
        raise RuntimeError("Locked predictor matrix assertion failed")
    if X["education_level"].value_counts().sort_index().to_dict() != {
        1: 10585, 2: 14030, 3: 4917, 4: 468
    }:
        raise RuntimeError("Education folding assertion failed")
    if X["marital_status"].value_counts().sort_index().to_dict() != {
        1: 13659, 2: 15964, 3: 377
    }:
        raise RuntimeError("Marital-status folding assertion failed")
    if set(X["sex"]) != {0, 1} or int(X["sex"].sum()) != int((raw["SEX"] == 2).sum()):
        raise RuntimeError("Sex indicator assertion failed")
    return raw, X, y, duplicate_stats


def _problem_and_audit(raw, X, y, duplicate_stats, output_root):
    problem = """# Problem definition

- **Target:** Default next month; default is the positive class.
- **Observation:** One existing Taiwanese bank credit-card account observed from April to September 2005, with the October 2005 repayment outcome recorded afterwards.
- **Primary decision:** Route an existing account for pre-delinquency intervention, such as collections outreach, credit-line review or a payment-plan offer.
- **Stakeholders:** The card-issuing bank and the account holder who may be contacted.
- **False positive:** An account that would have paid is routed for intervention.
- **False negative:** An account that defaults is not routed for intervention.
- **Evaluation stance:** Models are evaluated as ranking and classification systems. Probability statements are limited to calibration against the delivered sample and are made only where calibration evidence supports them.
- **Holdout status:** The test result is a one-time held-out confirmation under a pre-inspected public dataset, not blind external validation. Development evidence governs comparative claims.
"""
    _write_text(output_root / "audit" / "problem_definition.md", problem)

    raw_feature_columns = [column for column in RAW_COLUMNS if column not in {"ID", TARGET_RAW}]
    bills_raw = [f"BILL_AMT{i}" for i in range(1, 7)]
    payments_raw = [f"PAY_AMT{i}" for i in range(1, 7)]
    zero_segment = raw[bills_raw + payments_raw].eq(0).all(axis=1)
    bills_exceed_limit = raw[bills_raw].gt(raw["LIMIT_BAL"], axis=0).any(axis=1)
    categorical_raw = ["SEX", "EDUCATION", "MARRIAGE", *["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]]
    largest_majority = max(float(raw[column].value_counts(normalize=True).max()) for column in categorical_raw)
    undocumented_cells = int(raw["EDUCATION"].isin([0, 5, 6]).sum() + raw["MARRIAGE"].eq(0).sum())
    repayment_undocumented_cells = int(raw[["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]].isin([-2, 0]).sum().sum())

    rows = [
        ("source_filename", DATA_PATH.name, "Locked converted CSV"),
        ("retrieval_date", "2026-08-20", "Recorded retrieval date"),
        ("licence", "CC BY 4.0", "UCI dataset licence"),
        ("sha256", _sha256(DATA_PATH), "Matches locked converted-file hash"),
        ("file_size_bytes", DATA_PATH.stat().st_size, "Expected 2,867,208"),
        ("rows", len(raw), "Expected 30,000"),
        ("columns", raw.shape[1], "Expected 25"),
        ("predictors", len(raw_feature_columns), "Expected 23"),
        ("positive_count", int(y.sum()), "Default next month"),
        ("negative_count", int((1 - y).sum()), "No default next month"),
        ("positive_rate", float(y.mean()), "Delivered-sample rate; sampling design undocumented"),
        ("missing_values", int(raw.isna().sum().sum()), "Expected zero"),
        ("constant_columns", int((raw.nunique() == 1).sum()), "Expected zero"),
        ("largest_categorical_majority_rate", largest_majority, "No automatic exclusion"),
        ("duplicate_identifiers", int(raw["ID"].duplicated().sum()), "Expected zero"),
        *[(key, value, "Scope follows the locked duplicate ruling") for key, value in duplicate_stats.items()],
        ("undocumented_education_or_marriage_cells", undocumented_cells, "Folded under locked rulings"),
        ("undocumented_repayment_status_cells", repayment_undocumented_cells, "Retained; labels remain inferential"),
        ("negative_bill_cells", int(raw[bills_raw].lt(0).sum().sum()), "Retained unchanged"),
        ("rows_with_bill_above_credit_limit", int(bills_exceed_limit.sum()), "Retained unchanged"),
        ("maximum_bill_to_limit_ratio", float(raw[bills_raw].div(raw["LIMIT_BAL"], axis=0).max().max()), "No hard-cap inference"),
        ("maximum_bill_amount", int(raw[bills_raw].max().max()), "Retained unchanged"),
        ("maximum_payment_amount", int(raw[payments_raw].max().max()), "Retained unchanged"),
        ("zero_recorded_bill_and_payment_rows", int(zero_segment.sum()), "Pre-model segment; no indicator engineered"),
        ("off_grid_credit_limit_rows", int((raw["LIMIT_BAL"] % 10000 != 0).sum()), "Three locked anomalies retained"),
        ("identifier_columns_dropped", 1, "Identifier excluded before modelling"),
        ("candidate_leakage_variables", 0, "No predictor post-dates the decision point"),
    ]
    audit = pd.DataFrame(rows, columns=["audit_item", "observed_value", "verification_note"])
    audit.to_csv(output_root / "audit" / "data_audit.csv", index=False)

    pd.DataFrame(columns=["display_name", "availability_issue", "action"]).to_csv(
        output_root / "audit" / "leakage_register.csv", index=False
    )
    deciles = pd.qcut(np.arange(len(raw)), 10, labels=False)
    target_position = (
        pd.DataFrame({"position_decile": deciles + 1, "positive": y.to_numpy()})
        .groupby("position_decile", as_index=False)
        .agg(
            rows=("positive", "size"),
            positive_count=("positive", "sum"),
            positive_rate=("positive", "mean"),
        )
    )
    target_position.to_csv(output_root / "audit" / "target_rate_by_position.csv", index=False)

    raw_pay_columns = ["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]
    expected_pay_sets = {
        "PAY_0": set(range(-2, 9)),
        "PAY_2": set(range(-2, 9)),
        "PAY_3": set(range(-2, 9)),
        "PAY_4": set(range(-2, 9)),
        "PAY_5": {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8},
        "PAY_6": {-2, -1, 0, 2, 3, 4, 5, 6, 7, 8},
    }
    rulings = [
        ("R1", "Default next month", "Target polarity", "Retain source polarity; positive equals default", int(y.sum()) == 6636),
        ("R2", "All variables", "Missing values", "Assert zero; no imputation", int(raw.isna().sum().sum()) == 0),
        ("R3", "All variables", "Conventional missing tokens", "No replacement; undocumented codes handled explicitly", all(str(dtype) == "int64" for dtype in raw.dtypes)),
        ("R4", "Identifier", "Identifier column", "Drop before every model", "ID" not in X.columns),
        ("R5", "Repeated profiles", "Duplicate scopes", "Retain all rows; do not deduplicate", duplicate_stats["conflicting_label_groups"] == 21),
        ("R6", "Education level", "Undocumented levels", "Fold source levels 0, 5 and 6 into other or undocumented", X["education_level"].value_counts().sort_index().to_dict() == {1: 10585, 2: 14030, 3: 4917, 4: 468}),
        ("R7", "Marital status", "Undocumented level", "Fold source level 0 into other or undocumented", X["marital_status"].value_counts().sort_index().to_dict() == {1: 13659, 2: 15964, 3: 377}),
        ("R8", "Repayment status", "Undocumented codes", "Retain raw codes for trees and apply locked LR buckets", all(set(raw[column]) == expected_pay_sets[column] and 9 not in set(raw[column]) for column in raw_pay_columns)),
        ("R9", "Bill amounts", "Negative balances", "Retain without clipping or flooring", int(raw[bills_raw].lt(0).sum().sum()) > 0),
        ("R10", "Bill amounts", "Collinear block", "LR retains latest bill and prunes five older bills; trees retain all six", set(LR_PRUNED_FEATURES).isdisjoint(LR_USED_FEATURES)),
        ("R11", "Payment amounts", "Right skew", "Fold-contained log1p and robust scaling for LR only", not raw[payments_raw].lt(0).any().any()),
        ("R12", "Payment amounts", "Low mutual dependence", "Retain all six in every model", all(feature in X.columns for feature in PAYMENT_FEATURES)),
        ("R13", "Row position", "Possible ordering", "Never use position; shuffle and stratify", True),
        ("R14", "All variables", "Extreme observations", "Remove no rows", len(X) == 30000),
        ("R15", "Credit limit", "Three off-grid values", "Retain unchanged", int((raw["LIMIT_BAL"] % 10000 != 0).sum()) == 3),
        ("R16", "Sex", "Sensitive demographic attribute", "Retain as explicit binary indicator; make no fairness or legal claim", "sex" in X.columns),
    ]
    applied = pd.DataFrame(
        rulings, columns=["ruling", "display_name", "issue", "action", "verified"]
    )
    if not applied["verified"].all():
        failed = applied.loc[~applied["verified"], "ruling"].tolist()
        raise RuntimeError(f"Locked ruling verification failed: {failed}")
    applied.to_csv(output_root / "audit" / "applied_rulings.csv", index=False)

    types = {}
    for feature in PREDICTORS:
        if feature in NOMINAL:
            types[feature] = "Nominal categorical"
        elif feature in ORDINAL:
            types[feature] = "Ordinal"
        elif feature == "sex":
            types[feature] = "Binary categorical"
        else:
            types[feature] = "Quantitative"
    display_frame = pd.DataFrame(
        {
            "feature": [*PREDICTORS, "default"],
            "display_name": [*[DISPLAY_NAMES[item] for item in PREDICTORS], DISPLAY_NAMES["default"]],
            "feature_type": [*[types[item] for item in PREDICTORS], "Target (binary)"],
        }
    )
    display_frame.to_csv(output_root / "audit" / "feature_display_names.csv", index=False)
    return audit, target_position


def _development_eda(X_dev, y_dev, raw, y_full, output_root, plot_root):
    renamed = X_dev[PREDICTORS].rename(columns=DISPLAY_NAMES)
    corr = renamed.corr(method="spearman")
    corr.to_csv(output_root / "eda" / "correlation_matrix.csv")
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    max_pair = upper.abs().idxmax()
    max_value = float(corr.loc[max_pair[0], max_pair[1]])

    bill_corr = X_dev[BILL_FEATURES].corr(method="pearson")
    bill_upper = bill_corr.where(np.triu(np.ones(bill_corr.shape), 1).astype(bool)).stack()
    repayment_corr = X_dev[ORDINAL].corr(method="pearson")
    repayment_upper = repayment_corr.where(
        np.triu(np.ones(repayment_corr.shape), 1).astype(bool)
    ).stack()

    credit_quintile = pd.qcut(X_dev["credit_limit"], q=5, duplicates="drop")
    credit_rates = (
        pd.DataFrame({"bin": credit_quintile, "positive": y_dev})
        .groupby("bin", observed=True)["positive"]
        .mean()
        .tolist()
    )
    age_quintile = pd.qcut(X_dev["age"], q=5, duplicates="drop")
    age_rates = (
        pd.DataFrame({"bin": age_quintile, "positive": y_dev})
        .groupby("bin", observed=True)["positive"]
        .mean()
        .tolist()
    )
    average_ratio = X_dev[BILL_FEATURES].mean(axis=1) / X_dev["credit_limit"]
    ratio_decile = pd.qcut(average_ratio, q=10, duplicates="drop")
    ratio_rates = (
        pd.DataFrame({"bin": ratio_decile, "positive": y_dev})
        .groupby("bin", observed=True)["positive"]
        .mean()
        .tolist()
    )
    zero_segment = X_dev[BILL_FEATURES + PAYMENT_FEATURES].eq(0).all(axis=1)
    zero_count = int(zero_segment.sum())
    zero_rate = float(y_dev.loc[zero_segment].mean())
    nonzero_rate = float(y_dev.loc[~zero_segment].mean())

    latest_status = X_dev["repayment_status_september"]
    status_buckets = pd.Series(
        np.select(
            [latest_status <= 0, latest_status == 1, latest_status == 2, latest_status >= 3],
            ["non-positive status", "one-month delay", "two-month delay", "delay of three months or more"],
            default="unexpected",
        ),
        index=X_dev.index,
    )
    status_rates = (
        pd.DataFrame({"bucket": status_buckets, "positive": y_dev})
        .groupby("bucket", observed=True)["positive"]
        .agg(["size", "mean"])
        .reindex([
            "non-positive status", "one-month delay", "two-month delay",
            "delay of three months or more",
        ])
    )
    naive = X_dev[ORDINAL].ge(1).any(axis=1)
    naive_tp = int((naive & y_dev.eq(1)).sum())
    naive_flagged = int(naive.sum())
    naive_recall = naive_tp / int(y_dev.sum())
    naive_precision = naive_tp / naive_flagged
    last_decile_rate = float(y_full.iloc[-3000:].mean())

    metrics = {
        "development_rows": float(len(X_dev)),
        "development_positives": float(y_dev.sum()),
        "development_positive_rate": float(y_dev.mean()),
        "largest_absolute_spearman": float(abs(max_value)),
        "largest_spearman_signed": max_value,
        "bill_block_mean_absolute_pearson": float(bill_upper.abs().mean()),
        "bill_block_max_absolute_pearson": float(bill_upper.abs().max()),
        "repayment_block_mean_absolute_pearson": float(repayment_upper.abs().mean()),
        "repayment_block_max_absolute_pearson": float(repayment_upper.abs().max()),
        "zero_segment_rows": float(zero_count),
        "zero_segment_default_rate": zero_rate,
        "nonzero_segment_default_rate": nonzero_rate,
        "naive_any_delay_flagged": float(naive_flagged),
        "naive_any_delay_recall": naive_recall,
        "naive_any_delay_precision": naive_precision,
        "source_last_position_decile_default_rate": last_decile_rate,
    }
    for index, rate in enumerate(credit_rates, 1):
        metrics[f"credit_limit_quintile_{index}_default_rate"] = float(rate)
    for index, rate in enumerate(age_rates, 1):
        metrics[f"age_quintile_{index}_default_rate"] = float(rate)
    for index, rate in enumerate(ratio_rates, 1):
        metrics[f"audit_bill_to_limit_decile_{index}_default_rate"] = float(rate)
    for bucket, row in status_rates.iterrows():
        key = bucket.replace("-", "_").replace(" ", "_")
        metrics[f"latest_repayment_{key}_rows"] = float(row["size"])
        metrics[f"latest_repayment_{key}_default_rate"] = float(row["mean"])

    status_text = "; ".join(
        f"{bucket}: {int(row['size']):,} rows, rate {row['mean']:.4f}"
        for bucket, row in status_rates.iterrows()
    )
    findings = f"""# Development-set EDA findings

All statements below describe observed development or source structure. Hypotheses are labelled and were locked before model results were examined.

- The development set contains {int(y_dev.sum()):,} defaults among {len(y_dev):,} accounts ({y_dev.mean():.4f}).
- The largest absolute pairwise Spearman correlation is {abs(max_value):.4f}, between **{max_pair[0]}** and **{max_pair[1]}** (signed value {max_value:.4f}).
- Across the six bill amounts, mean absolute Pearson correlation is {bill_upper.abs().mean():.4f} and the maximum is {bill_upper.abs().max():.4f}. This verifies the locked LR pruning decision; trees retain the full block.
- Across the six raw repayment-status variables, mean absolute Pearson correlation is {repayment_upper.abs().mean():.4f} and the maximum is {repayment_upper.abs().max():.4f}.
- Latest-month repayment buckets show a threshold pattern: {status_text}.
- The audit-only average bill-to-limit decile default-rate sequence is {json.dumps([round(float(value), 6) for value in ratio_rates])}. It is not an engineered model feature.
- The zero-recorded bill-and-payment segment contains {zero_count:,} development accounts, with default rate {zero_rate:.4f}, against {nonzero_rate:.4f} outside the segment. No segment indicator is engineered.
- The development-only rule flagging any positive repayment-status code marks {naive_flagged:,} accounts, with recall {naive_recall:.4f} and precision {naive_precision:.4f}. It is contextual evidence only and cannot alter the 0.65 recall floor.
- The final source-position decile has default rate {last_decile_rate:.4f}. Row position is excluded from all models; the split is shuffled and stratified.
- No observations were removed. Negative bills and extreme bill/payment amounts remain in the model inputs specified by the decisions file.
- **Pre-model hypothesis:** the repayment-status cliff can provide strong monotonic or threshold signal to all four pipelines, with LR expressing the locked buckets and trees using raw codes.
- **Pre-model hypothesis:** the tree family may use historical bill trajectory, bill-to-limit interactions or the zero-recorded segment that are not explicitly represented by the LR pipeline.
- **Pre-model hypothesis:** correlated bill variables may distribute tree native importance differently from grouped permutation importance.

Development default-rate sequences by quintile were: Credit limit {json.dumps([round(float(value), 6) for value in credit_rates])}; Age {json.dumps([round(float(value), 6) for value in age_rates])}.
"""
    _write_text(output_root / "eda" / "eda_findings.md", findings)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_dev.value_counts().sort_index()
    ax.bar(["No default", "Default"], counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Development observations")
    ax.set_title("Development-set class balance")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman correlation"},
    )
    ax.set_title("Development-set predictor correlations")
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sequences = [
        ("Credit-limit quintile", credit_rates),
        ("Age quintile", age_rates),
        ("Average bill-to-limit decile (audit only)", ratio_rates),
    ]
    for ax, (label, values) in zip(axes, sequences):
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", color="#4472C4")
        ax.set_xlabel(label)
        ax.set_ylabel("Default rate")
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle("Development-set quantitative predictor–outcome patterns")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "quantitative_by_outcome.png", dpi=180)
    plt.close(fig)

    position = pd.DataFrame(
        {
            "position_decile": pd.qcut(np.arange(len(raw)), 10, labels=False) + 1,
            "positive": y_full.to_numpy(),
        }
    ).groupby("position_decile", as_index=False)["positive"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(position["position_decile"], position["positive"], marker="o")
    ax.axhline(y_full.mean(), color="grey", linestyle="--", label="Overall delivered-sample rate")
    ax.set_xlabel("Source-row position decile")
    ax.set_ylabel("Default rate")
    ax.set_title("Target rate by source-row position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "target_rate_by_position.png", dpi=180)
    plt.close(fig)
    return corr, metrics


def _evaluation_and_features(output_root):
    text = f"""# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Default next month.
- **Operating rule:** Maximise precision subject to recall of at least {RECALL_FLOOR:.2f}; ties use maximum recall and then the highest threshold.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Below-baseline handling:** A valid PR-AUC at or below the matched prevalence/dummy baseline is warned about and investigated, not suppressed.
- **Sampling caveat:** PR-AUC, precision and calibration describe the delivered sample. Its sampling design is undocumented, so no population-prevalence claim is made.
- **Test status:** The test set is a one-time held-out confirmation under a pre-inspected public dataset. Development evidence governs comparative claims.
"""
    _write_text(output_root / "audit" / "evaluation_framework.md", text)

    rows = []
    for model in MODEL_ORDER:
        if model == "lr":
            treatments = {
                "Credit limit, age and latest bill": "Robust-scaled",
                "Five older bill amounts": "Pruned by the locked chronology rule",
                "Six payment amounts": "Fold-contained log1p, then robust-scaled",
                "Repayment status": "Locked behavioural buckets; drop-reference one-hot",
                "Sex": "Explicit binary indicator",
                "Education and marital status": "Drop-reference one-hot with locked references",
            }
        else:
            treatments = {
                "Fourteen quantitative variables": "Raw values; no scaling or transformation",
                "Repayment status": "Raw integer codes",
                "Sex": "Explicit binary indicator",
                "Education and marital status": "Full one-hot encoding; no dropped level",
            }
        for group, treatment in treatments.items():
            rows.append(
                {
                    "model": MODEL_DISPLAY[model],
                    "feature_group": group,
                    "treatment": treatment,
                    "expected_encoded_terms": 28,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "preprocessing" / "feature_sets_by_model.csv", index=False)
    return frame


def _calibration(oof_scores, y_dev, output_root, plot_root):
    rows = []
    bin_rows = []
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Ideal")
    for model_name in MODEL_ORDER:
        stats = calibration_statistics(y_dev, oof_scores[model_name])
        rows.append(
            {"model": model_name, **stats, "input_rows": len(y_dev), "score_source": "nested OOF"}
        )
        bins = reliability_bins(y_dev, oof_scores[model_name], CALIBRATION_BINS)
        bins.insert(0, "model", model_name)
        bin_rows.append(bins)
        ax.plot(
            bins["mean_score"],
            bins["observed_rate"],
            marker="o",
            label=MODEL_DISPLAY[model_name],
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "modelling" / "calibration.csv", index=False)
    reliability = pd.concat(bin_rows, ignore_index=True)
    reliability.to_csv(output_root / "modelling" / "reliability_bins.csv", index=False)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed default rate")
    ax.set_title("OOF reliability curves (equal-frequency bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "calibration" / "reliability_curves.png", dpi=180)
    plt.close(fig)
    return frame, reliability


def _structural_findings(interpretability, X_dev, y_dev, output_root):
    permutation = interpretability["permutation"]
    shap_summaries = interpretability["shap"]
    summary = interpretability["summary"].set_index("model")
    metrics = {}
    lines = [
        "# Structural observations",
        "",
        "These statements report measured development-set evidence. They do not explain performance differences, recommend a model or answer a research question.",
        "",
        f"- The development evidence contains {len(y_dev):,} accounts and {int(y_dev.sum()):,} default outcomes.",
    ]
    for model_name in MODEL_ORDER:
        perm_frame = permutation[model_name]
        shap_frame = shap_summaries[model_name]
        top_perm = perm_frame.iloc[0]
        top_shap = shap_frame.iloc[0]
        positive = perm_frame["importance_mean"].clip(lower=0)
        concentration = float(positive.head(3).sum() / positive.sum()) if positive.sum() > 0 else np.nan
        metrics[f"{model_name}_top3_positive_permutation_share"] = concentration
        lines.append(
            f"- **{MODEL_DISPLAY[model_name]}:** the highest cross-fitted permutation rank was "
            f"{top_perm['display_name']} (mean average-precision decrease {top_perm['importance_mean']:.6f}); "
            f"the highest variable-level SHAP rank was {top_shap['display_name']} "
            f"(mean absolute contribution {top_shap['mean_absolute_shap']:.6f}, "
            f"direction {top_shap['direction']}, scale: {top_shap['output_scale']}). "
            f"The top three variables account for {concentration:.4f} of summed positive permutation importance."
        )

    dt = summary.loc["dt"]
    lines.append(
        f"- The final decision tree had realised depth {dt['tree_depth']:.0f} and "
        f"{dt['n_leaves']:.0f} leaves. Stability across resamples was not measured, so stable "
        "thresholds are not asserted; the full fitted rules are recorded in the structure file."
    )
    overlaps = []
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        value = int(summary.loc[model_a, f"top10_overlap_{model_b}"])
        overlaps.append(f"{model_a.upper()}–{model_b.upper()} {value}/10")
    lines.append("- Pairwise permutation top-ten overlap counts were: " + "; ".join(overlaps) + ".")

    older_bills = set(LR_PRUNED_FEATURES)
    for model_name in ["dt", "rf", "xgb"]:
        frame = permutation[model_name].set_index("feature")
        best_feature = min(older_bills, key=lambda feature: int(frame.loc[feature, "rank"]))
        best_rank = int(frame.loc[best_feature, "rank"])
        metrics[f"{model_name}_best_older_bill_permutation_rank"] = float(best_rank)
        lines.append(
            f"- **{MODEL_DISPLAY[model_name]}:** the highest-ranked older bill omitted from LR was "
            f"{DISPLAY_NAMES[best_feature]} at permutation rank {best_rank}. This records model use "
            "of an available historical input; it does not establish that the input caused a score difference."
        )

    lines.extend(
        [
            "- Nominal-variable SHAP directions are category-dependent; no single direction is assigned.",
            "- Quantitative, ordinal and binary directions use the locked Spearman-rho convention in each model's SHAP summary.",
            "- The zero-recorded segment, bill trajectory and bill-to-limit curvature remain pre-model hypotheses. Standard EDA and importance evidence do not by themselves isolate a causal performance explanation.",
            "- The 5,309 development defaults provide substantial positive-class support for the fitted rankings; this is an observed count, not an explanation for any model ordering.",
            "",
        ]
    )
    _write_text(output_root / "findings" / "structural_findings.md", "\n".join(lines))
    return metrics


def _registry_entries(context, code_version):
    rows = []

    def add(stage, model, scope, metric, value, source_file, seed_applicable):
        numeric = float(value) if value is not None else np.nan
        rows.append(
            {
                "dataset": "Default of Credit Card Clients",
                "stage": int(stage),
                "model": model,
                "scope": scope,
                "metric": metric,
                "value": numeric,
                "source_file": source_file,
                "seed_applicable": bool(seed_applicable),
                "seed": SEED if seed_applicable else np.nan,
                "code_version": code_version,
            }
        )

    for _, row in context["audit"].iterrows():
        value = pd.to_numeric(pd.Series([row["observed_value"]]), errors="coerce").iloc[0]
        if pd.notna(value):
            add(2, "", "full-file structural audit", row["audit_item"], value, "outputs/audit/data_audit.csv", False)
    for _, row in context["target_position"].iterrows():
        for metric in ["rows", "positive_count", "positive_rate"]:
            add(2, "", f"source position decile {int(row['position_decile'])}", metric, row[metric], "outputs/audit/target_rate_by_position.csv", False)

    split = context["split_spec"]
    for metric in [
        "development_rows", "development_positives", "test_rows", "test_positives",
        "outer_folds", "inner_folds",
    ]:
        add(4, "", "split", metric, split[metric], "outputs/split/split_spec.json", True)
    for metric, value in context["eda_metrics"].items():
        add(5, "", "development EDA", metric, value, "outputs/eda/eda_findings.md", True)

    for _, row in context["baselines"].iterrows():
        add(8, row["model"], "outer-fold CV", "pr_auc_mean", row["pr_auc_mean"], "outputs/modelling/baseline_cv.csv", True)
        add(8, row["model"], "outer-fold CV", "pr_auc_sd", row["pr_auc_sd"], "outputs/modelling/baseline_cv.csv", True)
    for _, row in context["nested"].iterrows():
        for metric in [
            "pr_auc_mean", "pr_auc_sd", *[f"pr_auc_fold_{i}" for i in range(1, 6)],
            "search_score", "final_fit_search_score",
        ]:
            add(9, row["model"], "nested development CV", metric, row[metric], "outputs/modelling/nested_cv.csv", True)
    for _, row in context["calibration"].iterrows():
        for metric in ["slope", "intercept", "brier", "log_loss"]:
            add(11, row["model"], "nested OOF development", metric, row[metric], "outputs/modelling/calibration.csv", True)
    for _, row in context["reliability"].iterrows():
        for metric in ["mean_score", "observed_rate", "n"]:
            add(11, row["model"], f"OOF reliability bin {int(row['bin'])}", metric, row[metric], "outputs/modelling/reliability_bins.csv", True)
    for _, row in context["thresholds"].iterrows():
        for metric in ["threshold", "precision", "recall", "fp", "fn", "precision_at_050", "recall_at_050"]:
            add(12, row["model"], "nested OOF development", metric, row[metric], "outputs/modelling/thresholds_selected.csv", True)
    for _, row in context["test_results"].iterrows():
        for metric in ["pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc", "calibration_slope", "precision", "recall", "threshold"]:
            add(13, row["model"], "locked holdout test", metric, row[metric], "outputs/modelling/test_results.csv", True)
    for _, row in context["confusion"].iterrows():
        for metric in ["tp", "fp", "tn", "fn"]:
            add(13, row["model"], "locked holdout test", metric, row[metric], "outputs/modelling/confusion_matrices.csv", True)
    for _, row in context["paired"].iterrows():
        pair = f"{row['model_a']}_vs_{row['model_b']}"
        for metric in ["mean_diff", "sd_diff", "folds_agreeing_in_sign", "n_folds"]:
            add(14, pair, "paired outer folds", metric, row[metric], "outputs/analysis/paired_fold_differences.csv", True)
        for metric in [column for column in row.index if column.startswith("fold_diff_")]:
            add(14, pair, "paired outer folds", metric, row[metric], "outputs/analysis/paired_fold_differences.csv", True)
    for _, row in context["metric_sensitivity"].iterrows():
        values = json.loads(row["values_json"])
        for model_name, value in values.items():
            add(15, model_name, "pooled nested OOF development", row["metric"], value, "outputs/analysis/metric_sensitivity.csv", True)

    interpretability = context["interpretability"]
    for model_name in MODEL_ORDER:
        for _, row in interpretability["permutation"][model_name].iterrows():
            for metric_name, column in [("mean", "importance_mean"), ("sd", "importance_sd"), ("rank", "rank")]:
                add(16, model_name, "cross-fitted development", f"permutation::{row['feature']}::{metric_name}", row[column], f"outputs/interpretability/permutation_importance_{model_name}.csv", True)
        for _, row in interpretability["shap"][model_name].iterrows():
            for metric_name, column in [
                ("mean_absolute", "mean_absolute_shap"),
                ("mean_signed", "mean_signed_shap"),
                ("direction_rho", "direction_rho"),
                ("rank", "rank"),
            ]:
                add(16, model_name, "final-refit development", f"shap::{row['feature']}::{metric_name}", row[column], f"outputs/interpretability/shap_summary_{model_name}.csv", True)
        if model_name != "lr":
            for _, row in interpretability["native"][model_name].iterrows():
                add(16, model_name, "final-refit development", f"native::{row['feature']}::importance", row["importance"], f"outputs/interpretability/native_importance_{model_name}.csv", True)

    coefficients = pd.read_csv(context["output_root"] / "interpretability" / "lr_coefficients.csv")
    for _, row in coefficients.iterrows():
        for metric_name, column in [
            ("coefficient", "coefficient"),
            ("odds_ratio", "odds_ratio"),
            ("absolute_coefficient", "absolute_coefficient"),
        ]:
            add(16, "lr", "final-refit development", f"encoded::{row['encoded_term']}::{metric_name}", row[column], "outputs/interpretability/lr_coefficients.csv", True)

    for _, row in interpretability["summary"].iterrows():
        for metric in [
            "n_features_used", "n_nonzero_coef", "tree_depth", "n_leaves",
            "native_vs_perm_rank_corr", "perm_vs_shap_rank_corr", "calibration_slope",
            *[f"top10_overlap_{model}" for model in MODEL_ORDER],
        ]:
            add(16, row["model"], "interpretability summary", metric, row[metric], "outputs/interpretability/interpretability_summary.csv", True)
    for metric, value in context["structural_metrics"].items():
        model_name = metric.split("_", 1)[0]
        add(17, model_name, "structural findings", metric, value, "outputs/findings/structural_findings.md", True)
    return pd.DataFrame(rows)


def _manifest(output_root, plot_root, expected_relative):
    expected = set(expected_relative)
    actual = set()
    for path in output_root.rglob("*"):
        if path.is_file():
            actual.add(str(Path("outputs") / path.relative_to(output_root)))
    for path in plot_root.rglob("*"):
        if path.is_file():
            actual.add(str(Path("plots") / path.relative_to(plot_root)))
    actual.add("outputs/manifest.csv")
    rows = []
    for relative in sorted(expected | actual):
        if relative.startswith("outputs/"):
            actual_path = output_root / Path(relative).relative_to("outputs")
        else:
            actual_path = plot_root / Path(relative).relative_to("plots")
        rows.append(
            {
                "path": relative,
                "expected": relative in expected,
                "produced": relative in actual,
                "unexpected": relative in actual and relative not in expected,
                "bytes": actual_path.stat().st_size if actual_path.exists() else 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "manifest.csv", index=False)
    return frame


def _environment(output_root, start_time, code_version, split_spec, interpretability):
    payload = {
        "dataset": "Default of Credit Card Clients",
        "started_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_time_seconds": time.time() - start_time,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "seed": SEED,
        "single_thread_requested": True,
        "code_version": code_version,
        "source_sha256": _sha256(DATA_PATH),
        "fold_indices_sha256": split_spec["fold_indices_sha256"],
        "shap_rows_by_model": {
            model: int(interpretability["shap"][model]["n_rows"].iloc[0]) for model in MODEL_ORDER
        },
        "shap_background_by_model": {
            model: int(interpretability["shap"][model]["background_n"].iloc[0]) for model in MODEL_ORDER
        },
        "permutation_repeats_per_fold": 30,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "shared_sgc_workflow_sha256": _sha256(SHARED_WORKFLOW_PATH),
    }
    _write_json(output_root / "audit" / "environment.json", payload)
    return payload


def expected_artifacts():
    return _shared.expected_artifacts()


def run_workflow(output_root=None, plot_root=None, verification=False):
    start_time = time.time()
    output_root = Path(output_root) if output_root else ROOT / "outputs"
    plot_root = Path(plot_root) if plot_root else ROOT / "plots"
    _shared._ensure_directories(output_root, plot_root)
    code_version = _code_version()
    print(f"Starting Default of Credit Card Clients workflow ({code_version})", flush=True)

    raw, X, y, duplicate_stats = _load_and_assert()
    audit, target_position = _problem_and_audit(raw, X, y, duplicate_stats, output_root)
    X_dev, X_test, y_dev, y_test, outer_splits, split_spec = _shared._split_and_folds(
        X, y, output_root
    )
    _, eda_metrics = _development_eda(X_dev, y_dev, raw, y, output_root, plot_root)
    _evaluation_and_features(output_root)

    print("Stage 8: untuned constrained baselines", flush=True)
    baselines = _shared._untuned_baselines(X_dev, y_dev, outer_splits, output_root)
    nested, oof_scores, outer_models, final_models, fold_scores = _shared._nested_cv(
        X_dev, y_dev, outer_splits, output_root
    )
    calibration, reliability = _calibration(oof_scores, y_dev, output_root, plot_root)
    thresholds, _ = _shared._thresholds(oof_scores, y_dev, output_root, plot_root)
    _shared._below_baseline_warnings(baselines, nested, oof_scores, y_dev, output_root)

    test_results, confusion, _ = _shared._test_evaluation(
        final_models, X_test, y_test, thresholds, output_root
    )
    paired = _shared._paired_comparison(fold_scores, output_root)
    metric_sensitivity = _shared._metric_sensitivity(
        oof_scores, y_dev, calibration, thresholds, output_root
    )
    print("Stage 16: cross-fitted permutation and SHAP evidence", flush=True)
    interpretability = run_interpretability(
        final_models,
        outer_models,
        outer_splits,
        X_dev,
        y_dev,
        calibration,
        output_root / "interpretability",
        plot_root / "interpretability",
    )
    structural_metrics = _structural_findings(interpretability, X_dev, y_dev, output_root)

    context = {
        "output_root": output_root,
        "audit": audit,
        "target_position": target_position,
        "split_spec": split_spec,
        "eda_metrics": eda_metrics,
        "baselines": baselines,
        "nested": nested,
        "calibration": calibration,
        "reliability": reliability,
        "thresholds": thresholds,
        "test_results": test_results,
        "confusion": confusion,
        "paired": paired,
        "metric_sensitivity": metric_sensitivity,
        "interpretability": interpretability,
        "structural_metrics": structural_metrics,
    }
    registry = _registry_entries(context, code_version)
    registry.to_csv(output_root / "registry.csv", index=False)
    registry[["stage", "model", "scope", "metric"]].drop_duplicates().sort_values(
        ["stage", "model", "scope", "metric"]
    ).to_csv(output_root / "registry_expected_keys.csv", index=False)
    environment = _environment(output_root, start_time, code_version, split_spec, interpretability)

    if not verification:
        _write_json(
            output_root / "audit" / "reproducibility.json",
            {"status": "pending", "note": "Written after the separate exact reproduction run"},
        )
        _write_json(
            output_root / "audit" / "invariant_results.json",
            {"status": "pending", "note": "Written by tests/test_invariants.py"},
        )
    else:
        _write_json(output_root / "audit" / "reproducibility.json", {"status": "verification-run"})
        _write_json(output_root / "audit" / "invariant_results.json", {"status": "verification-run"})

    manifest = _manifest(output_root, plot_root, expected_artifacts())
    missing = manifest.loc[manifest["expected"] & ~manifest["produced"], "path"].tolist()
    unexpected = manifest.loc[manifest["unexpected"], "path"].tolist()
    if missing or unexpected:
        raise RuntimeError(f"Manifest mismatch; missing={missing}, unexpected={unexpected}")
    print(f"Workflow complete in {environment['run_time_seconds']:.1f} seconds", flush=True)
    return {
        "code_version": code_version,
        "output_root": str(output_root),
        "plot_root": str(plot_root),
        "numeric_summary": {
            "nested": nested.to_dict(orient="records"),
            "calibration": calibration.to_dict(orient="records"),
            "thresholds": thresholds.to_dict(orient="records"),
            "test_results": test_results.to_dict(orient="records"),
            "confusion": confusion.to_dict(orient="records"),
            "paired": paired.to_dict(orient="records"),
            "interpretability_summary": interpretability["summary"].to_dict(orient="records"),
        },
    }
