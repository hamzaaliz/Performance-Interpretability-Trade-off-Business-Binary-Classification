import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from config import (
    BINARY, CALIBRATION_BINS, COLUMN_MAP, DATA_PATH, DISPLAY_NAMES,
    EXPECTED_LEVELS, MODEL_DISPLAY, MODEL_ORDER, NOMINAL, ORDINAL,
    PREDICTORS, QUANTITATIVE, RAW_COLUMNS, RECALL_FLOOR,
)
from modules.evaluation import (
    calibration_statistics, metrics_at_threshold, reliability_bins,
    select_threshold, threshold_table,
)
from modules.preprocessing import apply_locked_rulings


SOURCE_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
SOURCE_BYTES = 131728
AGE_FACTORS = {1: 0.055, 2: 0.045, 3: 0.040, 4: 0.025, 5: 0.015}
ZERO_COUNTS = {
    "call_failure": 702,
    "seconds_of_use": 154,
    "frequency_of_use": 154,
    "frequency_of_sms": 603,
    "distinct_called_numbers": 154,
    "customer_value": 132,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _customer_value_residual(frame):
    factor = frame["age_group"].map(AGE_FACTORS).to_numpy(dtype=float)
    expected = factor * (
        frame["seconds_of_use"].to_numpy(dtype=float)
        + frame["frequency_of_use"].to_numpy(dtype=float)
        + 100.0 * frame["frequency_of_sms"].to_numpy(dtype=float)
    )
    return np.abs(frame["customer_value"].to_numpy(dtype=float) - expected)


def _conflicting_profiles(frame):
    feature_columns = [column for column in frame.columns if column != "churn"]
    grouped = frame.groupby(feature_columns, dropna=False)["churn"].agg(["size", "nunique"])
    conflicting = grouped.loc[grouped["nunique"] > 1]
    dormant = conflicting.reset_index()
    dormant = dormant.loc[
        dormant["seconds_of_use"].eq(0)
        & dormant["frequency_of_use"].eq(0)
        & dormant["distinct_called_numbers"].eq(0)
    ]
    return conflicting, dormant


def load_and_assert():
    source = pd.read_csv(DATA_PATH)
    if list(source.columns) != RAW_COLUMNS:
        raise RuntimeError("Source column list differs from the locked specification")
    if source.shape != (3150, 14):
        raise RuntimeError(f"Unexpected shape: {source.shape}")
    if _sha256(DATA_PATH) != SOURCE_SHA256 or DATA_PATH.stat().st_size != SOURCE_BYTES:
        raise RuntimeError("Source identity assertion failed")
    raw = source.rename(columns=COLUMN_MAP)
    if int((raw.dtypes == "int64").sum()) != 13 or int((raw.dtypes == "float64").sum()) != 1:
        raise RuntimeError("Source dtype assertion failed")
    if int(raw.isna().sum().sum()) != 0 or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise RuntimeError("Unexpected missing or non-finite value")
    if int((raw < 0).sum().sum()) != 0:
        raise RuntimeError("Unexpected negative value")
    for column, expected in EXPECTED_LEVELS.items():
        if set(raw[column]) != set(expected):
            raise RuntimeError(f"Category-level assertion failed for {column}")
    if int(raw["churn"].sum()) != 495:
        raise RuntimeError("Target-polarity assertion failed")
    if int(raw.duplicated().sum()) != 300:
        raise RuntimeError("Exact-duplicate assertion failed")
    if int(raw.drop(columns="churn").duplicated().sum()) != 314:
        raise RuntimeError("Feature-duplicate assertion failed")
    if int(raw.drop(columns="churn").duplicated(keep=False).sum()) != 476:
        raise RuntimeError("Repeated-feature-profile assertion failed")
    conflicting, dormant_conflicting = _conflicting_profiles(raw)
    if len(conflicting) != 14 or int(conflicting["size"].sum()) != 64:
        raise RuntimeError("Conflicting-profile assertion failed")
    if len(dormant_conflicting) != 11 or int(dormant_conflicting["size"].sum()) != 56:
        raise RuntimeError("Dormant conflicting-profile assertion failed")
    if {column: int(raw[column].eq(0).sum()) for column in ZERO_COUNTS} != ZERO_COUNTS:
        raise RuntimeError("Zero-mass assertion failed")
    age_map = raw.groupby("age_group")["age"].unique().apply(list).to_dict()
    if age_map != {1: [15], 2: [25], 3: [30], 4: [45], 5: [55]}:
        raise RuntimeError("Age functional-duplicate assertion failed")
    if int((raw["distinct_called_numbers"] > raw["frequency_of_use"]).sum()) != 120:
        raise RuntimeError("Call-count semantic-ambiguity assertion failed")
    residual = _customer_value_residual(raw)
    if float(residual.max()) > 5e-10:
        raise RuntimeError("Customer Value formula assertion failed")
    if int(raw["charge_amount"].max()) != 10:
        raise RuntimeError("Charge Amount source range assertion failed")
    if raw["charge_amount"].value_counts().sort_index().to_dict() != {
        0: 1768, 1: 617, 2: 395, 3: 199, 4: 76, 5: 30,
        6: 11, 7: 14, 8: 19, 9: 14, 10: 7,
    }:
        raise RuntimeError("Charge Amount source-level count assertion failed")

    y = raw["churn"].astype(int)
    X = apply_locked_rulings(raw.drop(columns=["churn", "age", "customer_value"]))
    if X.shape != (3150, 12) or list(X.columns) != PREDICTORS:
        raise RuntimeError("Locked shared input-frame assertion failed")
    if X["charge_amount"].value_counts().sort_index().to_dict() != {0: 1768, 1: 617, 2: 395, 3: 199, 4: 76, 5: 95}:
        raise RuntimeError("Charge Amount collapse assertion failed")
    if int(X["mean_call_duration"].eq(0).sum()) != 154:
        raise RuntimeError("Mean-duration zero-denominator assertion failed")
    if not np.isfinite(X["mean_call_duration"]).all():
        raise RuntimeError("Non-finite mean call duration")
    return raw, X, y


def problem_and_audit(raw, X, y, output_root):
    problem = """# Problem definition

- **Target:** Customer churn; churn is the positive class.
- **Observation:** One Iranian telecommunications customer, described by aggregates over months 1–9; churn is recorded at month 12.
- **Primary decision:** Whether to target a customer with a retention intervention before the churn outcome.
- **Stakeholders:** The telecommunications provider and the customer considered for retention contact.
- **False positive:** A customer who stayed, targeted with a retention offer.
- **False negative:** A customer who churned, not targeted.
- **Evaluation stance:** Models are evaluated as ranking and classification systems; probability statements are limited to the supplied sample.
- **Holdout status:** The stratified test set is a one-time held-out confirmation; development evidence governs comparison.
"""
    _write_text(output_root / "audit" / "problem_definition.md", problem)

    position = (
        pd.DataFrame({
            "position_decile": pd.qcut(np.arange(len(raw)), 10, labels=False) + 1,
            "positive": y.to_numpy(),
        })
        .groupby("position_decile", as_index=False)
        .agg(rows=("positive", "size"), positive_count=("positive", "sum"), positive_rate=("positive", "mean"))
    )
    position.to_csv(output_root / "audit" / "target_rate_by_position.csv", index=False)

    conflicting, dormant_conflicting = _conflicting_profiles(raw)
    residual = _customer_value_residual(raw)
    zero_call = raw["frequency_of_use"].eq(0)
    zero_sets_equal = (
        zero_call.equals(raw["seconds_of_use"].eq(0))
        and zero_call.equals(raw["distinct_called_numbers"].eq(0))
    )
    customer_zero_expected = zero_call & raw["frequency_of_sms"].eq(0)
    audit_rows = [
        ("source_filename", DATA_PATH.name, "Locked supplied CSV"),
        ("retrieval_date", "2026-08-25", "Recorded project date"),
        ("licence", "CC BY 4.0", "UCI dataset licence"),
        ("sha256", _sha256(DATA_PATH), "Matches locked source hash"),
        ("file_size_bytes", DATA_PATH.stat().st_size, "Expected 131,728"),
        ("rows", len(raw), "Expected 3,150"),
        ("columns", raw.shape[1], "Expected 14"),
        ("source_predictors", 13, "Before two locked exclusions"),
        ("model_predictors_per_family", 11, "Age and Customer Value excluded; Seconds/mean-duration asymmetry"),
        ("positive_count", int(y.sum()), "Churn"),
        ("negative_count", int((1 - y).sum()), "Stayed"),
        ("positive_rate", float(y.mean()), "Supplied-sample rate"),
        ("missing_values", int(raw.isna().sum().sum()), "Expected zero"),
        ("non_finite_values", 0, "Asserted"),
        ("negative_values", int((raw < 0).sum().sum()), "Expected zero"),
        ("exact_duplicate_rows", int(raw.duplicated().sum()), "Retained"),
        ("feature_identical_duplicates", int(raw.drop(columns="churn").duplicated().sum()), "Retained"),
        ("rows_in_repeated_feature_profiles", int(raw.drop(columns="churn").duplicated(keep=False).sum()), "Retained"),
        ("conflicting_feature_profiles", len(conflicting), "64 rows in total; retained"),
        ("rows_in_conflicting_feature_profiles", int(conflicting["size"].sum()), "Retained"),
        ("dormant_conflicting_profiles", len(dormant_conflicting), "Corrected pre-execution count"),
        ("rows_in_dormant_conflicting_profiles", int(dormant_conflicting["size"].sum()), "Corrected pre-execution count"),
        ("charge_amount_source_max", int(raw["charge_amount"].max()), "Source documents 0-9; observed 10"),
        ("charge_amount_collapsed_max", int(X["charge_amount"].max()), "Levels 5-10 collapsed to 5"),
        ("age_exact_functional_duplicate", True, "Age Group maps to 15,25,30,45,55"),
        ("distinct_numbers_exceeds_frequency_rows", int((raw["distinct_called_numbers"] > raw["frequency_of_use"]).sum()), "Retained; no ratio derived"),
        ("customer_value_max_abs_formula_residual", float(residual.max()), "Tolerance 5e-10"),
        ("zero_call_usage_sets_equal", zero_sets_equal, "Seconds, frequency and distinct numbers"),
        ("customer_value_zero_set_verified", raw["customer_value"].eq(0).equals(customer_zero_expected), "Zero calls and zero SMS"),
        ("call_failure_zero_count", ZERO_COUNTS["call_failure"], "Substantive zero"),
        ("seconds_of_use_zero_count", ZERO_COUNTS["seconds_of_use"], "Substantive zero"),
        ("frequency_of_use_zero_count", ZERO_COUNTS["frequency_of_use"], "Substantive zero"),
        ("frequency_of_sms_zero_count", ZERO_COUNTS["frequency_of_sms"], "Substantive zero"),
        ("distinct_called_numbers_zero_count", ZERO_COUNTS["distinct_called_numbers"], "Substantive zero"),
        ("customer_value_zero_count", ZERO_COUNTS["customer_value"], "Audit-only derived field"),
        ("identifier_columns", 0, "No identifier column supplied"),
        ("candidate_antecedent_indicators", 2, "Status and Complains retained with timing caveat"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["audit_item", "observed_value", "verification_note"])
    audit.to_csv(output_root / "audit" / "data_audit.csv", index=False)

    pd.DataFrame(
        columns=["feature", "display_name", "availability_issue", "action"]
    ).to_csv(output_root / "audit" / "leakage_register.csv", index=False)

    rulings = [
        ("R1", "Churn", "Target polarity", "Use raw churn code; positive is one", int(y.sum()) == 495),
        ("R2", "All variables", "Missing values", "Assert zero; no imputation", int(raw.isna().sum().sum()) == 0),
        ("R3", "All variables", "Disguised missingness", "Retain substantive zeros", True),
        ("R4", "Age", "Exact functional duplicate", "Drop from all models", "age" not in X.columns),
        ("R5", "All variables", "Identifier", "No identifier exclusion", True),
        ("R6", "Charge Amount", "Undocumented code 10", "Resolve through R7 collapse", int(raw["charge_amount"].max()) == 10),
        ("R7", "Charge Amount", "Sparse upper levels", "Collapse 5-10 to 5", int(X["charge_amount"].max()) == 5),
        ("R8", "All variables", "Duplicate rows", "Retain all rows", len(X) == 3150),
        ("R9", "Feature profiles", "Conflicting targets", "Retain all 14 patterns / 64 rows", len(conflicting) == 14 and int(conflicting["size"].sum()) == 64),
        ("R10", "All variables", "Outliers", "Remove no observations", len(X) == 3150),
        ("R11", "Six zero-mass fields", "Exact zeros", "Retain without indicators", True),
        ("R12", "Row position", "Ordering", "Exclude and shuffle split", True),
        ("R13", "All variables", "Negative values", "Assert none", int((raw < 0).sum().sum()) == 0),
        ("R14", "Mean call duration", "Zero denominator", "Set to 0.0", int(X.loc[raw["frequency_of_use"].eq(0), "mean_call_duration"].ne(0).sum()) == 0),
        ("R15", "Preprocessing", "Fold containment", "Only deterministic row-wise recodes precede folds", True),
        ("R16", "Call counts", "Source-semantic ambiguity", "Retain; derive no ratio", int((raw["distinct_called_numbers"] > raw["frequency_of_use"]).sum()) == 120),
        ("R17", "Customer Value", "Exact undocumented composite", "Drop from all models", "customer_value" not in X.columns and float(residual.max()) <= 5e-10),
    ]
    applied = pd.DataFrame(rulings, columns=["ruling", "display_name", "issue", "action", "verified"])
    if not applied["verified"].all():
        raise RuntimeError("Locked ruling verification failed")
    applied.to_csv(output_root / "audit" / "applied_rulings.csv", index=False)

    types = {feature: "Quantitative" for feature in QUANTITATIVE}
    types.update({feature: "Binary categorical" for feature in BINARY})
    types.update({feature: "Ordinal" for feature in ORDINAL})
    types.update({feature: "Categorical, five levels" for feature in NOMINAL})
    display_rows = []
    for feature in PREDICTORS:
        feature_type = "Derived quantitative (LR only)" if feature == "mean_call_duration" else types.get(feature, "Quantitative")
        model_use = "LR only" if feature == "mean_call_duration" else ("Tree family only" if feature == "seconds_of_use" else "All four models")
        display_rows.append((feature, DISPLAY_NAMES[feature], feature_type, model_use))
    display_rows.extend(
        [
            ("age", DISPLAY_NAMES["age"], "Dropped exact duplicate", "Audit only"),
            ("customer_value", DISPLAY_NAMES["customer_value"], "Dropped exact composite", "Audit only"),
            ("churn", DISPLAY_NAMES["churn"], "Target (binary)", "Positive class = 1"),
        ]
    )
    pd.DataFrame(display_rows, columns=["feature", "display_name", "feature_type", "model_use"]).to_csv(
        output_root / "audit" / "feature_display_names.csv", index=False
    )
    return audit, position


def development_eda(X_dev, y_dev, raw, y_full, output_root, plot_root):
    corr = X_dev[PREDICTORS].corr(method="spearman")
    corr.to_csv(output_root / "eda" / "correlation_matrix.csv")
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    max_pair = upper.abs().idxmax()
    max_value = float(corr.loc[max_pair[0], max_pair[1]])

    rate_features = [*QUANTITATIVE, "mean_call_duration"]
    quintile_rates = {}
    for feature in rate_features:
        bins = pd.qcut(X_dev[feature], q=5, duplicates="drop")
        rates = pd.DataFrame({"bin": bins, "positive": y_dev}).groupby("bin", observed=True)["positive"].mean()
        quintile_rates[feature] = [float(value) for value in rates]
    dormant = X_dev["frequency_of_use"].eq(0)
    age_rates = pd.DataFrame({"age_group": X_dev["age_group"], "positive": y_dev}).groupby("age_group")["positive"].agg(["size", "mean"])
    charge_rates = pd.DataFrame({"charge_amount": X_dev["charge_amount"], "positive": y_dev}).groupby("charge_amount")["positive"].agg(["size", "mean"])

    findings = f"""# Development-set EDA findings

All predictor–target statements below use development data only. Hypotheses were recorded before model results were examined.

- The development set contains {int(y_dev.sum()):,} churn outcomes among {len(y_dev):,} rows ({y_dev.mean():.6f}).
- The largest absolute development-set Spearman association in the shared modelling input frame is {abs(max_value):.6f}, for `{max_pair[0]}` and `{max_pair[1]}` (signed value {max_value:.6f}).
- Development-set churn rates by age band are {json.dumps({int(index): round(float(row['mean']), 6) for index, row in age_rates.iterrows()}, sort_keys=True)}.
- Development-set churn rates by collapsed charge band are {json.dumps({int(index): round(float(row['mean']), 6) for index, row in charge_rates.iterrows()}, sort_keys=True)}.
- The development set contains {int(dormant.sum())} rows with zero call usage; their churn rate is {float(y_dev.loc[dormant].mean()):.6f}.
- Quantitative churn-rate sequences by development-set quintile are {json.dumps({key: [round(value, 6) for value in values] for key, values in quintile_rates.items()}, sort_keys=True)}.
- No observations were removed; substantive zero values are retained.
- `Distinct Called Numbers` and `Frequency of use` are retained without a derived ratio because their source semantics are unresolved in 120 source rows.
- **Pre-model hypothesis:** the zero-usage block may be isolated by a tree threshold, whereas the additive logistic specification represents it through several lower-tail values.
- **Pre-model hypothesis:** subscription length may exhibit a threshold or U-shaped relationship that a single linear logistic term cannot express directly.
- **Pre-model hypothesis:** age-band one-hot terms permit non-monotone logistic effects, while repeated tree splits can also represent them.
"""
    _write_text(output_root / "eda" / "eda_findings.md", findings)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_dev.value_counts().sort_index()
    ax.bar(["Stayed", "Churned"], counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Development observations")
    ax.set_title("Development-set class balance")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman"})
    ax.set_title("Development-set predictor associations")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, feature in zip(axes.ravel(), rate_features):
        values = quintile_rates[feature]
        ax.plot(range(1, len(values) + 1), values, marker="o")
        ax.set_title(DISPLAY_NAMES[feature])
        ax.set_xlabel("Development quantile")
        ax.set_ylabel("Churn rate")
    for ax in axes.ravel()[len(rate_features):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "quantitative_by_outcome.png", dpi=180)
    plt.close(fig)
    return corr, (str(max_pair[0]), str(max_pair[1])), max_value


def evaluation_and_features(output_root):
    text = f"""# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Customer churn.
- **Operating rule:** Require recall of at least {RECALL_FLOOR:.2f}, maximise precision, and break ties by recall and then the highest threshold. The guaranteed-floor fallback is labelled explicitly if ever invoked.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **False positive:** A customer who stayed, targeted with a retention offer.
- **False negative:** A customer who churned, not targeted.
- **Test status:** The test set is evaluated once after the development-only gate passes.
"""
    _write_text(output_root / "audit" / "evaluation_framework.md", text)
    rows = []
    treatments = {
        "lr": {
            "Quantitative": "Six standardised terms: five retained raw quantities plus mean call duration; Seconds of Use excluded",
            "Ordinal": "Collapsed Charge Amount score 0-5, standardised",
            "Binary": "Complaint registered, contractual plan and account inactive as explicit 0/1 indicators",
            "Categorical": "Age Group one-hot encoded with explicit levels 1-5; band 3 dropped",
            "Excluded": "Age and Customer Value; raw Seconds of Use replaced by mean call duration",
        },
        "tree": {
            "Quantitative": "Six raw quantities including Seconds of Use; no mean call duration",
            "Ordinal": "Collapsed Charge Amount code 0-5 retained raw",
            "Binary": "Complaint registered, contractual plan and account inactive as explicit 0/1 indicators",
            "Categorical": "Age Group retained as raw integer code 1-5",
            "Excluded": "Age and Customer Value",
        },
    }
    for model in MODEL_ORDER:
        key = "lr" if model == "lr" else "tree"
        expected_count = 14 if model == "lr" else 11
        for group, treatment in treatments[key].items():
            rows.append({
                "model": MODEL_DISPLAY[model],
                "feature_group": group,
                "treatment": treatment,
                "expected_encoded_terms": expected_count,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "preprocessing" / "feature_sets_by_model.csv", index=False)
    return frame


def calibration_stage(oof_scores, y_dev, output_root, plot_root):
    rows = []
    bin_rows = []
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Ideal")
    for model_name in MODEL_ORDER:
        stats = calibration_statistics(y_dev, oof_scores[model_name])
        rows.append({"model": model_name, **stats, "input_rows": len(y_dev), "score_source": "nested OOF"})
        bins = reliability_bins(y_dev, oof_scores[model_name], CALIBRATION_BINS)
        bins.insert(0, "model", model_name)
        bin_rows.append(bins)
        ax.plot(bins["mean_score"], bins["observed_rate"], marker="o", label=MODEL_DISPLAY[model_name])
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "modelling" / "calibration.csv", index=False)
    pd.concat(bin_rows, ignore_index=True).to_csv(output_root / "modelling" / "reliability_bins.csv", index=False)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("OOF reliability curves (equal-frequency bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "calibration" / "reliability_curves.png", dpi=180)
    plt.close(fig)
    return frame


def thresholds_stage(oof_scores, y_dev, output_root, plot_root):
    selected_rows = []
    sweeps = {}
    for model_name in MODEL_ORDER:
        table = threshold_table(y_dev.to_numpy(), oof_scores[model_name])
        table.to_csv(output_root / "modelling" / f"threshold_sweep_{model_name}.csv", index=False)
        selected = select_threshold(table, RECALL_FLOOR)
        reference = metrics_at_threshold(y_dev, oof_scores[model_name], 0.50)
        selected_rows.append(
            {
                "model": model_name,
                "floored_metric": "recall",
                "floor_value": RECALL_FLOOR,
                "optimised_metric": "precision",
                "selection_mode": selected["selection_mode"],
                "threshold": selected["threshold"],
                "floor_attained": bool(selected["floor_attained"]),
                "predicted_positive": int(selected["predicted_positive"]),
                "precision": selected["precision"],
                "recall": selected["recall"],
                "tp": int(selected["tp"]),
                "fp": int(selected["fp"]),
                "tn": int(selected["tn"]),
                "fn": int(selected["fn"]),
                "precision_at_050": reference["precision"],
                "recall_at_050": reference["recall"],
                "predicted_positive_at_050": int(reference["predicted_positive"]),
                "selection_input_rows": len(y_dev),
            }
        )
        sweeps[model_name] = table
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ordered = table.sort_values("threshold")
        ax.plot(ordered["threshold"], ordered["precision"], label="Precision")
        ax.plot(ordered["threshold"], ordered["recall"], label="Recall")
        ax.axhline(RECALL_FLOOR, color="grey", linestyle="--", label="Recall floor")
        ax.axvline(selected["threshold"], color="black", linestyle=":", label="Selected threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{MODEL_DISPLAY[model_name]}: OOF threshold sweep")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_root / "thresholds" / f"threshold_sweep_{model_name}.png", dpi=180)
        plt.close(fig)
    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(output_root / "modelling" / "thresholds_selected.csv", index=False)
    return selected_frame, sweeps
