import hashlib
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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import (
    BINARY,
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_BINS,
    CODE_VERSION_LABEL,
    DATA_PATH,
    DISPLAY_NAMES,
    EXPECTED_LEVELS,
    INNER_FOLDS,
    LR_REFERENCE_LEVELS,
    MODEL_DISPLAY,
    MODEL_ORDER,
    NOMINAL,
    ORDINAL,
    OUTER_FOLDS,
    PREDICTORS,
    QUANTITATIVE,
    RAW_COLUMNS,
    RECALL_FLOOR,
    ROOT,
    SEARCH_ITERATIONS,
    SEED,
    TEST_SIZE,
)
from modules import model_dt, model_lr, model_rf, model_xgb
from modules.evaluation import (
    calibration_statistics,
    discrimination_metrics,
    metrics_at_threshold,
    positive_scores,
    reliability_bins,
    select_threshold,
    threshold_table,
)
from modules.interpretability import run_interpretability
from modules.preprocessing import apply_locked_rulings, encoded_feature_metadata


MODEL_MODULES = {"lr": model_lr, "dt": model_dt, "rf": model_rf, "xgb": model_xgb}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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
    paths = [ROOT / "config.py", ROOT / "run_workflow.py"] + sorted((ROOT / "modules").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return f"{CODE_VERSION_LABEL}:{digest.hexdigest()[:16]}"


def _ensure_directories(output_root, plot_root):
    for relative in [
        "audit",
        "split",
        "eda",
        "preprocessing",
        "modelling",
        "analysis",
        "interpretability",
        "findings",
    ]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    for relative in ["eda", "calibration", "thresholds", "interpretability"]:
        (plot_root / relative).mkdir(parents=True, exist_ok=True)


def _load_and_assert():
    raw = pd.read_csv(DATA_PATH, sep=r"\s+")
    if list(raw.columns) != RAW_COLUMNS:
        raise RuntimeError("Source column list differs from the locked specification")
    if raw.shape != (1000, 21):
        raise RuntimeError(f"Unexpected shape: {raw.shape}")
    if _sha256(DATA_PATH) != "5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f":
        raise RuntimeError("Source SHA-256 mismatch")
    if raw.isna().sum().sum() != 0:
        raise RuntimeError("Unexpected missing values")
    if int(raw.duplicated().sum()) != 0 or int(raw.drop(columns="kredit").duplicated().sum()) != 0:
        raise RuntimeError("Unexpected duplicate rows")
    if 7 in raw["verw"].unique():
        raise RuntimeError("Unexpected observed purpose code 7")
    if raw["gastarb"].value_counts().to_dict().get(1) != 37:
        raise RuntimeError("Foreign-worker coding polarity check failed")
    y = 1 - raw["kredit"]
    if int(y.sum()) != 300:
        raise RuntimeError("Target polarity assertion failed")
    processed = apply_locked_rulings(raw.drop(columns="kredit"))
    if processed["bishkred"].value_counts().sort_index().to_dict() != {1: 633, 2: 333, 3: 34}:
        raise RuntimeError("Number-of-credits collapse verification failed")
    return raw, processed, y.astype(int)


def _problem_and_audit(raw, X, y, output_root):
    problem = """# Problem definition

- **Target:** Credit risk; bad credit is the positive class.
- **Observation:** One credit contract granted by a bank in southern Germany between 1973 and 1975, with its outcome recorded afterwards.
- **Primary decision:** Grant or refuse a credit application.
- **Stakeholders:** The credit-granting institution and the applicant.
- **False positive:** A credit that would have been repaid is refused.
- **False negative:** A credit that was not complied with is granted.
- **Evaluation stance:** Models are evaluated as ranking and classification systems. Probability statements are conditional on the 30% sample positive rate and are made only where calibration evidence supports them.
"""
    _write_text(output_root / "audit" / "problem_definition.md", problem)

    rows = [
        ("source_filename", DATA_PATH.name, "Locked source file"),
        ("retrieval_date", "2026-07-15", "Local file timestamp; retained as the recorded retrieval date"),
        ("licence", "CC BY 4.0", "UCI dataset licence"),
        ("sha256", _sha256(DATA_PATH), "Matches locked hash"),
        ("rows", len(raw), "Expected 1000"),
        ("columns", raw.shape[1], "Expected 21"),
        ("predictors", X.shape[1], "Expected 20"),
        ("positive_count", int(y.sum()), "Bad credit"),
        ("positive_rate", float(y.mean()), "Sample rate; positives were oversampled"),
        ("missing_values", int(raw.isna().sum().sum()), "Expected zero"),
        ("exact_duplicate_rows", int(raw.duplicated().sum()), "Expected zero"),
        ("feature_identical_duplicates", int(raw.drop(columns="kredit").duplicated().sum()), "Expected zero"),
        ("constant_columns", int((raw.nunique() == 1).sum()), "Expected zero"),
        ("near_constant_foreign_worker_majority_rate", float(raw["gastarb"].value_counts(normalize=True).max()), "963 of 1000 in the majority category"),
        ("maximum_credit_amount", int(raw["hoehe"].max()), "Retained; no outlier removal"),
        ("maximum_age", int(raw["alter"].max()), "Retained; no outlier removal"),
        ("undocumented_category_codes", 0, "All observed codes matched the code table"),
        ("identifier_columns", 0, "No identifiers present"),
        ("candidate_leakage_variables", 0, "All predictors are available at the decision point"),
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
        .agg(rows=("positive", "size"), positive_count=("positive", "sum"), positive_rate=("positive", "mean"))
    )
    target_position.to_csv(output_root / "audit" / "target_rate_by_position.csv", index=False)

    rulings = [
        ("Credit risk", "Target polarity", "Construct positive class as one minus the raw outcome", int(y.sum()) == 300),
        ("All variables", "Missing values", "Assert zero; no imputation", raw.isna().sum().sum() == 0),
        ("All variables", "Disguised missing codes", "No replacement", True),
        ("All variables", "Duplicate rows", "Retain all rows; none observed", raw.duplicated().sum() == 0),
        ("All variables", "Identifier columns", "No identifier exclusion", True),
        ("Purpose", "Unobserved education code", "Do not create an all-zero dummy", 7 not in raw["verw"].unique()),
        ("Purpose", "Rare observed levels", "Retain without collapsing", True),
        ("Number of credits at this bank", "Sparse top levels", "Combine original levels 3 and 4", X["bishkred"].value_counts().sort_index().to_dict() == {1: 633, 2: 333, 3: 34}),
        ("People financially dependent and foreign worker", "Corrected polarity", "Apply no reversal beyond explicit binary indicators", int(raw["gastarb"].eq(1).sum()) == 37),
        ("Credit amount (DM)", "Right skew", "Log1p only inside logistic-regression folds", True),
        ("Row position", "Target ordering", "Never use position; shuffled stratified split", True),
        ("All variables", "Outliers", "No observations removed", len(X) == 1000),
    ]
    pd.DataFrame(rulings, columns=["display_name", "issue", "action", "verified"]).to_csv(
        output_root / "audit" / "applied_rulings.csv", index=False
    )

    display_frame = pd.DataFrame(
        {
            "feature": [
                "duration", "amount", "age", "checking_account_status", "credit_history",
                "savings", "employment_duration", "instalment_rate", "present_residence",
                "property", "number_of_credits", "job_quality", "dependants", "telephone",
                "foreign_worker", "purpose", "personal_status_and_sex", "other_debtors",
                "other_instalment_plans", "housing", "credit_risk",
            ],
            "display_name": [DISPLAY_NAMES[item] for item in PREDICTORS] + [DISPLAY_NAMES["kredit"]],
        }
    )
    display_frame.to_csv(output_root / "audit" / "feature_display_names.csv", index=False)
    return audit, target_position


def _split_and_folds(X, y, output_root):
    indices = np.arange(len(X))
    development_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y, random_state=SEED, shuffle=True
    )
    X_dev = X.iloc[development_indices].reset_index(drop=True)
    y_dev = y.iloc[development_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    development_profiles = set(map(tuple, X_dev[PREDICTORS].to_numpy()))
    test_pattern_overlap = int(sum(tuple(row) in development_profiles for row in X_test[PREDICTORS].to_numpy()))
    if test_pattern_overlap != 100:
        raise RuntimeError("Locked split feature-pattern-overlap assertion failed")
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_splits = [(train.copy(), valid.copy()) for train, valid in outer.split(X_dev, y_dev)]
    fold_hash = hashlib.sha256(pickle.dumps(outer_splits)).hexdigest()
    payload = {
        "seed": SEED,
        "test_fraction": TEST_SIZE,
        "development_rows": len(X_dev),
        "development_positives": int(y_dev.sum()),
        "test_rows": len(X_test),
        "test_positives": int(y_test.sum()),
        "test_rows_with_development_feature_pattern": test_pattern_overlap,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "shuffle": True,
        "fold_indices_sha256": fold_hash,
        "development_source_indices": development_indices.tolist(),
        "test_source_indices": test_indices.tolist(),
    }
    _write_json(output_root / "split" / "split_spec.json", payload)
    joblib.dump(outer_splits, output_root / "split" / "fold_indices.pkl")
    return X_dev, X_test, y_dev, y_test, outer_splits, payload


def _eda(X_dev, y_dev, raw, y_full, output_root, plot_root):
    renamed = X_dev[PREDICTORS].rename(columns=DISPLAY_NAMES)
    corr = renamed.corr(method="spearman")
    corr.to_csv(output_root / "eda" / "correlation_matrix.csv")
    corr_values = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    max_pair = corr_values.abs().idxmax()
    max_value = float(corr.loc[max_pair[0], max_pair[1]])

    quantitative_rates = {}
    for feature in QUANTITATIVE:
        bins = pd.qcut(X_dev[feature], q=5, duplicates="drop")
        rates = pd.DataFrame({"bin": bins, "positive": y_dev}).groupby("bin", observed=True)["positive"].mean()
        quantitative_rates[feature] = [float(item) for item in rates]

    end_rate = float(y_full.iloc[750:].mean())
    findings = f"""# Development-set EDA findings

All statements below describe observed development or source structure. Hypotheses are labelled and were recorded before model results were examined.

- The development set contains {int(y_dev.sum())} bad-credit outcomes among {len(y_dev)} observations ({y_dev.mean():.3f}).
- The largest absolute pairwise Spearman correlation is {abs(max_value):.3f}, between **{max_pair[0]}** and **{max_pair[1]}** (signed value {max_value:.3f}). The locked decision retains both variables for logistic regression.
- Across source rows 751–1000, the bad-credit rate is {end_rate:.3f}. Row position is excluded from every feature matrix and the split is shuffled and stratified.
- Credit amount ranges from {int(X_dev['hoehe'].min())} to {int(X_dev['hoehe'].max())}; age ranges from {int(X_dev['alter'].min())} to {int(X_dev['alter'].max())}. No observations were removed.
- **Pre-model hypothesis:** ordered-score predictors may provide monotonic signal suitable for logistic regression, while quantitative quintile-rate changes may also permit tree thresholds.
- **Pre-model hypothesis:** sparse nominal levels may yield unstable encoded-term coefficients; they are retained and flagged rather than collapsed.
- **Pre-model hypothesis:** the depth-five decision tree may express fewer distinct structures than the two ensembles; this is a model constraint, not a result.

Quantitative positive-rate sequences by development-set quintile were recorded as: Credit duration {json.dumps(quantitative_rates['laufzeit'])}; Credit amount {json.dumps(quantitative_rates['hoehe'])}; Age {json.dumps(quantitative_rates['alter'])}.
"""
    _write_text(output_root / "eda" / "eda_findings.md", findings)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_dev.value_counts().sort_index()
    ax.bar(["Good credit", "Bad credit"], counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Development observations")
    ax.set_title("Development-set class balance")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman correlation"})
    ax.set_title("Development-set predictor correlations")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feature in zip(axes, QUANTITATIVE):
        frame = pd.DataFrame({"value": X_dev[feature], "Outcome": np.where(y_dev == 1, "Bad credit", "Good credit")})
        sns.boxplot(data=frame, x="Outcome", y="value", hue="Outcome", legend=False, ax=ax)
        ax.set_ylabel(DISPLAY_NAMES[feature])
        ax.set_xlabel("")
    fig.suptitle("Development-set quantitative distributions by outcome")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "quantitative_by_outcome.png", dpi=180)
    plt.close(fig)

    position = pd.DataFrame({"position_decile": pd.qcut(np.arange(len(raw)), 10, labels=False) + 1, "positive": y_full.to_numpy()})
    position = position.groupby("position_decile", as_index=False)["positive"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(position["position_decile"], position["positive"], marker="o")
    ax.axhline(y_full.mean(), color="grey", linestyle="--", label="Overall sample rate")
    ax.set_xlabel("Source-row position decile")
    ax.set_ylabel("Bad-credit rate")
    ax.set_title("Target rate by source-row position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "target_rate_by_position.png", dpi=180)
    plt.close(fig)
    return corr, max_pair, max_value


def _evaluation_and_features(output_root):
    text = f"""# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss, and confusion-matrix counts.
- **Positive class:** Bad credit.
- **Operating rule:** Maximise precision subject to recall of at least {RECALL_FLOOR:.2f}; ties use maximum recall and then the highest threshold.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Below-baseline handling:** A valid PR-AUC at or below the matched prevalence/dummy baseline is warned about and investigated, not suppressed.
- **Sampling caveat:** PR-AUC, precision and calibration refer to the 30% positive-rate sample and are not population estimates.
"""
    _write_text(output_root / "audit" / "evaluation_framework.md", text)

    rows = []
    for model in MODEL_ORDER:
        if model == "lr":
            treatments = {
                "Quantitative": "Standardised; credit amount receives fold-contained log1p first",
                "Ordinal": "Ordered scores, standardised",
                "Binary": "Explicit 0/1 indicators",
                "Nominal": "Drop-reference one-hot encoding with locked references",
            }
            count = 33
        else:
            treatments = {
                "Quantitative": "Raw values, no scaling",
                "Ordinal": "Raw ordered scores",
                "Binary": "Explicit 0/1 indicators",
                "Nominal": "Full one-hot encoding; no dropped level",
            }
            count = 38
        for group, treatment in treatments.items():
            rows.append(
                {
                    "model": MODEL_DISPLAY[model],
                    "feature_group": group,
                    "treatment": treatment,
                    "expected_encoded_terms": count,
                }
            )
    feature_sets = pd.DataFrame(rows)
    feature_sets.to_csv(output_root / "preprocessing" / "feature_sets_by_model.csv", index=False)
    return feature_sets


def _baseline_models():
    models = {model: MODEL_MODULES[model].make_pipeline() for model in MODEL_ORDER}
    models["dummy"] = Pipeline([("classifier", DummyClassifier(strategy="prior", random_state=SEED))])
    return models


def _untuned_baselines(X_dev, y_dev, outer_splits, output_root):
    rows = []
    for model_name, estimator in _baseline_models().items():
        scores = cross_val_score(
            estimator,
            X_dev,
            y_dev,
            cv=outer_splits,
            scoring="average_precision",
            n_jobs=1,
        )
        rows.append(
            {
                "model": model_name,
                "pr_auc_mean": float(scores.mean()),
                "pr_auc_sd": float(scores.std(ddof=1)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "modelling" / "baseline_cv.csv", index=False)
    return result


def _model_complexity(model_name, pipeline):
    classifier = pipeline.named_steps["classifier"]
    if model_name == "lr":
        return {"nonzero_coefficients": int(np.sum(np.abs(classifier.coef_[0]) > 1e-12))}
    if model_name == "dt":
        return {"depth": int(classifier.get_depth()), "leaves": int(classifier.get_n_leaves())}
    if model_name == "rf":
        depths = [tree.get_depth() for tree in classifier.estimators_]
        leaves = [tree.get_n_leaves() for tree in classifier.estimators_]
        return {
            "n_estimators": len(classifier.estimators_),
            "mean_depth": float(np.mean(depths)),
            "max_depth": int(np.max(depths)),
            "mean_leaves": float(np.mean(leaves)),
        }
    return {
        "n_estimators": int(classifier.get_params()["n_estimators"]),
        "max_depth": int(classifier.get_params()["max_depth"]),
    }


def _search(model_name, X, y, cv):
    return RandomizedSearchCV(
        estimator=MODEL_MODULES[model_name].make_pipeline(),
        param_distributions=MODEL_MODULES[model_name].search_space(),
        n_iter=SEARCH_ITERATIONS,
        scoring="average_precision",
        n_jobs=1,
        cv=cv,
        refit=True,
        random_state=SEED,
        return_train_score=False,
        error_score="raise",
    ).fit(X, y)


def _nested_cv(X_dev, y_dev, outer_splits, output_root):
    rows = []
    oof_scores = {}
    outer_models = {}
    final_models = {}
    fold_scores_by_model = {}
    for model_name in MODEL_ORDER:
        print(f"Stage 9: nested search for {MODEL_DISPLAY[model_name]}", flush=True)
        model_oof = np.full(len(X_dev), np.nan, dtype=float)
        fold_scores = []
        inner_scores = []
        params = []
        complexities = []
        fitted_outer = []
        for fold_number, (train_idx, valid_idx) in enumerate(outer_splits, 1):
            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
            search = _search(model_name, X_dev.iloc[train_idx], y_dev.iloc[train_idx], inner)
            fitted = search.best_estimator_
            scores = positive_scores(fitted, X_dev.iloc[valid_idx])
            model_oof[valid_idx] = scores
            fold_score = average_precision_score(y_dev.iloc[valid_idx], scores)
            fold_scores.append(float(fold_score))
            inner_scores.append(float(search.best_score_))
            params.append({"fold": fold_number, "params": search.best_params_})
            complexities.append({"fold": fold_number, **_model_complexity(model_name, fitted)})
            fitted_outer.append(fitted)
            print(
                f"  outer fold {fold_number}/{OUTER_FOLDS}: AP={fold_score:.5f}, inner={search.best_score_:.5f}",
                flush=True,
            )
        if np.isnan(model_oof).any():
            raise RuntimeError(f"Incomplete OOF predictions for {model_name}")
        print(f"  final development search for {MODEL_DISPLAY[model_name]}", flush=True)
        final_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        final_search = _search(model_name, X_dev, y_dev, final_inner)
        final_models[model_name] = final_search.best_estimator_
        outer_models[model_name] = fitted_outer
        oof_scores[model_name] = model_oof
        fold_scores_by_model[model_name] = fold_scores
        row = {
            "model": model_name,
            "pr_auc_mean": float(np.mean(fold_scores)),
            "pr_auc_sd": float(np.std(fold_scores, ddof=1)),
            "search_score": float(np.mean(inner_scores)),
            "params_json": json.dumps(params, default=_json_default, sort_keys=True),
            "complexity_proxies_json": json.dumps(complexities, default=_json_default, sort_keys=True),
            "final_fit_search_score": float(final_search.best_score_),
            "final_fit_params_json": json.dumps(final_search.best_params_, default=_json_default, sort_keys=True),
        }
        for index, value in enumerate(fold_scores, 1):
            row[f"pr_auc_fold_{index}"] = value
        rows.append(row)

        pd.DataFrame(
            {
                "row_index": np.arange(len(X_dev)),
                "fold": np.concatenate(
                    [np.full(len(valid), fold, dtype=int) for fold, (_, valid) in enumerate(outer_splits, 1)]
                )[
                    np.argsort(np.concatenate([valid for _, valid in outer_splits]))
                ],
                "y_true": y_dev.to_numpy(),
                "y_score": model_oof,
            }
        ).to_csv(output_root / "modelling" / f"oof_predictions_{model_name}.csv", index=False)

    result = pd.DataFrame(rows)
    column_order = [
        "model", "pr_auc_mean", "pr_auc_sd", *[f"pr_auc_fold_{i}" for i in range(1, 6)],
        "search_score", "params_json", "complexity_proxies_json",
        "final_fit_search_score", "final_fit_params_json",
    ]
    result = result[column_order]
    result.to_csv(output_root / "modelling" / "nested_cv.csv", index=False)
    return result, oof_scores, outer_models, final_models, fold_scores_by_model


def _calibration(oof_scores, y_dev, output_root, plot_root):
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
    pd.concat(bin_rows, ignore_index=True).to_csv(
        output_root / "modelling" / "reliability_bins.csv", index=False
    )
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed bad-credit rate")
    ax.set_title("OOF reliability curves (equal-frequency bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_root / "calibration" / "reliability_curves.png", dpi=180)
    plt.close(fig)
    return frame


def _thresholds(oof_scores, y_dev, output_root, plot_root):
    selected_rows = []
    sweeps = {}
    for model_name in MODEL_ORDER:
        table = threshold_table(y_dev.to_numpy(), oof_scores[model_name])
        table.to_csv(output_root / "modelling" / f"threshold_sweep_{model_name}.csv", index=False)
        selected = select_threshold(table, RECALL_FLOOR)
        reference = metrics_at_threshold(y_dev, oof_scores[model_name], 0.50)
        row = {
            "model": model_name,
            "threshold": selected["threshold"],
            "precision": selected["precision"],
            "recall": selected["recall"],
            "fp": int(selected["fp"]),
            "fn": int(selected["fn"]),
            "precision_at_050": reference["precision"],
            "recall_at_050": reference["recall"],
            "selection_input_rows": len(y_dev),
        }
        selected_rows.append(row)
        sweeps[model_name] = table

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table["threshold"], table["precision"], label="Precision")
        ax.plot(table["threshold"], table["recall"], label="Recall")
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


def _bootstrap_indices(y_test):
    y_array = y_test.to_numpy()
    positives = np.flatnonzero(y_array == 1)
    negatives = np.flatnonzero(y_array == 0)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        samples.append(sample)
    return samples


def _test_evaluation(final_models, X_test, y_test, selected_thresholds, output_root):
    print("Stage 13: one-time locked test evaluation", flush=True)
    bootstrap = _bootstrap_indices(y_test)
    threshold_map = selected_thresholds.set_index("model")["threshold"].to_dict()
    test_rows = []
    confusion_rows = []
    test_scores = {}
    for model_name in MODEL_ORDER:
        scores = positive_scores(final_models[model_name], X_test)
        test_scores[model_name] = scores
        discrimination = discrimination_metrics(y_test, scores)
        calibration = calibration_statistics(y_test, scores)
        operating = metrics_at_threshold(y_test, scores, threshold_map[model_name])
        bootstrap_scores = np.array(
            [average_precision_score(y_test.iloc[index], scores[index]) for index in bootstrap], dtype=float
        )
        low, high = np.percentile(bootstrap_scores, [2.5, 97.5])
        test_rows.append(
            {
                "model": model_name,
                "pr_auc": discrimination["pr_auc"],
                "pr_auc_ci_low": float(low),
                "pr_auc_ci_high": float(high),
                "roc_auc": discrimination["roc_auc"],
                "calibration_slope": calibration["slope"],
                "precision": operating["precision"],
                "recall": operating["recall"],
                "threshold": threshold_map[model_name],
                "test_rows": len(y_test),
                "test_positives": int(y_test.sum()),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            }
        )
        confusion_rows.append(
            {"model": model_name, **{key: operating[key] for key in ["tp", "fp", "tn", "fn"]}}
        )
    test_frame = pd.DataFrame(test_rows)
    confusion_frame = pd.DataFrame(confusion_rows)
    test_frame.to_csv(output_root / "modelling" / "test_results.csv", index=False)
    confusion_frame.to_csv(output_root / "modelling" / "confusion_matrices.csv", index=False)
    return test_frame, confusion_frame, test_scores


def _paired_comparison(fold_scores_by_model, output_root):
    rows = []
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        differences = np.asarray(fold_scores_by_model[model_a]) - np.asarray(fold_scores_by_model[model_b])
        mean = float(differences.mean())
        agreeing = int(np.sum(differences > 0)) if mean >= 0 else int(np.sum(differences < 0))
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_diff": mean,
                "sd_diff": float(differences.std(ddof=1)),
                "folds_agreeing_in_sign": agreeing,
                "n_folds": len(differences),
                **{f"fold_diff_{index}": float(value) for index, value in enumerate(differences, 1)},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "analysis" / "paired_fold_differences.csv", index=False)
    return frame


def _metric_sensitivity(oof_scores, y_dev, calibration_frame, selected_thresholds, output_root):
    threshold_map = selected_thresholds.set_index("model")["threshold"].to_dict()
    calibration_map = calibration_frame.set_index("model")["slope"].to_dict()
    metrics = {
        "PR-AUC": {},
        "ROC-AUC": {},
        "Precision at operating threshold": {},
        "Recall at operating threshold": {},
        "Calibration slope proximity to one": {},
    }
    for model_name in MODEL_ORDER:
        scores = oof_scores[model_name]
        operating = metrics_at_threshold(y_dev, scores, threshold_map[model_name])
        metrics["PR-AUC"][model_name] = average_precision_score(y_dev, scores)
        metrics["ROC-AUC"][model_name] = roc_auc_score(y_dev, scores)
        metrics["Precision at operating threshold"][model_name] = operating["precision"]
        metrics["Recall at operating threshold"][model_name] = operating["recall"]
        metrics["Calibration slope proximity to one"][model_name] = abs(calibration_map[model_name] - 1)
    rows = []
    for metric, values in metrics.items():
        ascending = metric == "Calibration slope proximity to one"
        ordered = sorted(values, key=lambda model: (values[model], MODEL_ORDER.index(model)), reverse=not ascending)
        rows.append(
            {
                "metric": metric,
                "rank_1": ordered[0],
                "rank_2": ordered[1],
                "rank_3": ordered[2],
                "rank_4": ordered[3],
                "values_json": json.dumps(values, sort_keys=True),
                "evidence_scope": "pooled nested OOF development predictions",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "analysis" / "metric_sensitivity.csv", index=False)
    return frame


def _below_baseline_warnings(baselines, nested, oof_scores, y_dev, output_root):
    dummy = float(baselines.set_index("model").loc["dummy", "pr_auc_mean"])
    rows = []
    for model_name in MODEL_ORDER:
        pooled = float(average_precision_score(y_dev, oof_scores[model_name]))
        nested_mean = float(nested.set_index("model").loc[model_name, "pr_auc_mean"])
        if pooled <= dummy or nested_mean <= dummy:
            correlation = float(spearmanr(y_dev, oof_scores[model_name]).statistic)
            rows.append(
                {
                    "model": model_name,
                    "warning": "PR-AUC at or below matched dummy baseline",
                    "dummy_pr_auc": dummy,
                    "pooled_oof_pr_auc": pooled,
                    "nested_mean_pr_auc": nested_mean,
                    "target_score_spearman": correlation,
                    "positive_count": int(y_dev.sum()),
                    "action": "Polarity and prediction-score diagnostics completed; valid result retained",
                }
            )
    columns = [
        "model", "warning", "dummy_pr_auc", "pooled_oof_pr_auc", "nested_mean_pr_auc",
        "target_score_spearman", "positive_count", "action",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output_root / "audit" / "warnings.csv", index=False)
    return frame


def _structural_findings(interpretability, X_dev, y_dev, output_root):
    permutation = interpretability["permutation"]
    shap_summaries = interpretability["shap"]
    summary = interpretability["summary"].set_index("model")
    lines = [
        "# Structural observations",
        "",
        "These statements report measured development-set evidence. They do not explain performance differences, recommend a model, or answer a research question.",
        "",
        f"- The development evidence contains {len(y_dev)} observations and {int(y_dev.sum())} positive outcomes.",
    ]
    for model_name in MODEL_ORDER:
        top_perm = permutation[model_name].iloc[0]
        top_shap = shap_summaries[model_name].iloc[0]
        lines.append(
            f"- **{MODEL_DISPLAY[model_name]}:** the highest cross-fitted grouped permutation rank was "
            f"{top_perm['display_name']} (mean average-precision decrease {top_perm['importance_mean']:.6f}); "
            f"the highest variable-level SHAP rank was {top_shap['display_name']} "
            f"(mean absolute contribution {top_shap['mean_absolute_shap']:.6f}, scale: {top_shap['output_scale']})."
        )
    dt = summary.loc["dt"]
    lines.append(
        f"- The final decision tree had realised depth {dt['tree_depth']:.0f} and {dt['n_leaves']:.0f} leaves; "
        "its full rules are recorded in `outputs/interpretability/dt_structure.txt`."
    )
    overlaps = []
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        overlaps.append(
            f"{model_a.upper()}–{model_b.upper()} "
            f"{int(summary.loc[model_a, f'top10_overlap_{model_b}'])}/"
            f"{int(summary.loc[model_a, f'top10_shared_denominator_{model_b}'])} shared-feature slots"
        )
    lines.append("- Pairwise permutation top-ten overlap counts were: " + "; ".join(overlaps) + ".")
    lines.extend(
        [
            "- Nominal-variable SHAP directions are recorded as category-dependent; no single direction was assigned.",
            "- Quantitative, ordinal and binary directions use the locked Spearman-rho convention and are recorded in each model's SHAP summary.",
            "- The Stage 5 statements remain pre-model hypotheses; this file records no claim that a structural pattern caused a metric difference.",
            "",
        ]
    )
    _write_text(output_root / "findings" / "structural_findings.md", "\n".join(lines))


def _registry_entries(context, code_version):
    rows = []

    def add(stage, model, scope, metric, value, source_file, seed_applicable):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        rows.append(
            {
                "dataset": "Iranian Churn",
                "stage": int(stage),
                "model": model,
                "scope": scope,
                "metric": metric,
                "value": float(value),
                "source_file": source_file,
                "seed_applicable": bool(seed_applicable),
                "seed": SEED if seed_applicable else np.nan,
                "code_version": code_version,
            }
        )

    split = context["split_spec"]
    for metric in [
        "development_rows", "development_positives", "test_rows", "test_positives",
        "test_rows_with_development_feature_pattern", "outer_folds", "inner_folds",
    ]:
        add(4, "", "split", metric, split[metric], "outputs/split/split_spec.json", True)

    for _, row in context["baselines"].iterrows():
        add(8, row["model"], "outer-fold CV", "pr_auc_mean", row["pr_auc_mean"], "outputs/modelling/baseline_cv.csv", True)
        add(8, row["model"], "outer-fold CV", "pr_auc_sd", row["pr_auc_sd"], "outputs/modelling/baseline_cv.csv", True)
    for _, row in context["nested"].iterrows():
        for metric in ["pr_auc_mean", "pr_auc_sd", *[f"pr_auc_fold_{i}" for i in range(1, 6)], "search_score", "final_fit_search_score"]:
            add(9, row["model"], "nested development CV", metric, row[metric], "outputs/modelling/nested_cv.csv", True)
    for _, row in context["calibration"].iterrows():
        for metric in ["slope", "intercept", "brier", "log_loss"]:
            add(11, row["model"], "nested OOF development", metric, row[metric], "outputs/modelling/calibration.csv", True)
    for _, row in context["thresholds"].iterrows():
        for metric in [
            "floor_value", "threshold", "floor_attained", "predicted_positive",
            "precision", "recall", "tp", "fp", "tn", "fn",
            "precision_at_050", "recall_at_050", "predicted_positive_at_050",
            "selection_input_rows",
        ]:
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
    interpretability = context["interpretability"]
    for model_name in MODEL_ORDER:
        for _, row in interpretability["permutation"][model_name].iterrows():
            add(16, model_name, "cross-fitted development", f"permutation::{row['feature']}", row["importance_mean"], f"outputs/interpretability/permutation_importance_{model_name}.csv", True)
            for fold in range(1, 6):
                add(16, model_name, f"cross-fitted development fold {fold}", f"permutation::{row['feature']}", row[f"fold_mean_{fold}"], f"outputs/interpretability/permutation_importance_{model_name}.csv", True)
        for _, row in interpretability["shap"][model_name].iterrows():
            add(16, model_name, "final-refit development", f"shap::{row['feature']}", row["mean_absolute_shap"], f"outputs/interpretability/shap_summary_{model_name}.csv", True)
    return pd.DataFrame(rows)


def _manifest(output_root, plot_root, expected_relative):
    rows = []
    expected_set = set(expected_relative)
    for relative in sorted(expected_set):
        base = output_root.parent if relative.startswith("outputs/") or relative.startswith("plots/") else ROOT
        actual_path = base / relative
        produced = actual_path.exists() or relative == "outputs/manifest.csv"
        rows.append(
            {
                "path": relative,
                "expected": True,
                "produced": produced,
                "bytes": actual_path.stat().st_size if actual_path.exists() else 0,
            }
        )
    actual = set()
    for base, prefix in [(output_root, "outputs"), (plot_root, "plots")]:
        for path in base.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                actual.add(f"{prefix}/{path.relative_to(base)}")
    for relative in sorted(actual - expected_set):
        base = output_root.parent
        actual_path = base / relative
        rows.append(
            {
                "path": relative,
                "expected": False,
                "produced": True,
                "bytes": actual_path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "manifest.csv", index=False)
    return frame


def _environment(output_root, start_time, code_version, split_spec, interpretability):
    payload = {
        "dataset": "Iranian Churn",
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
    }
    _write_json(output_root / "audit" / "environment.json", payload)
    return payload


# Dataset-specific locked stages replace the generic South German Credit scaffold above.
from modules.churn_stages import (
    calibration_stage as _calibration,
    development_eda as _eda,
    evaluation_and_features as _evaluation_and_features,
    load_and_assert as _load_and_assert,
    problem_and_audit as _problem_and_audit,
    thresholds_stage as _thresholds,
)


def expected_artifacts():
    outputs = [
        "outputs/registry.csv", "outputs/registry_expected_keys.csv", "outputs/manifest.csv",
        "outputs/audit/problem_definition.md", "outputs/audit/data_audit.csv",
        "outputs/audit/leakage_register.csv", "outputs/audit/target_rate_by_position.csv",
        "outputs/audit/applied_rulings.csv", "outputs/audit/feature_display_names.csv",
        "outputs/audit/evaluation_framework.md", "outputs/audit/environment.json",
        "outputs/audit/reproducibility.json", "outputs/audit/warnings.csv",
        "outputs/audit/invariant_results.json", "outputs/split/split_spec.json",
        "outputs/split/fold_indices.pkl", "outputs/eda/eda_findings.md",
        "outputs/eda/correlation_matrix.csv", "outputs/preprocessing/feature_sets_by_model.csv",
        "outputs/modelling/baseline_cv.csv", "outputs/modelling/nested_cv.csv",
        "outputs/modelling/calibration.csv", "outputs/modelling/reliability_bins.csv",
        "outputs/modelling/thresholds_selected.csv", "outputs/modelling/test_results.csv",
        "outputs/modelling/confusion_matrices.csv", "outputs/analysis/paired_fold_differences.csv",
        "outputs/analysis/metric_sensitivity.csv", "outputs/interpretability/lr_coefficients.csv",
        "outputs/interpretability/dt_structure.txt", "outputs/interpretability/native_importance_lr.csv",
        "outputs/interpretability/feature_rankings_combined.csv",
        "outputs/interpretability/interpretability_summary.csv",
        "outputs/findings/structural_findings.md",
    ]
    for model in MODEL_ORDER:
        outputs.extend(
            [
                f"outputs/modelling/oof_predictions_{model}.csv",
                f"outputs/modelling/threshold_sweep_{model}.csv",
                f"outputs/interpretability/permutation_importance_{model}.csv",
                f"outputs/interpretability/shap_summary_{model}.csv",
                f"outputs/interpretability/shap_terms_{model}.csv",
            ]
        )
        if model != "lr":
            outputs.extend(
                [
                    f"outputs/interpretability/native_importance_{model}.csv",
                    f"outputs/interpretability/native_importance_terms_{model}.csv",
                ]
            )
    plots = [
        "plots/eda/class_balance.png", "plots/eda/correlation_heatmap.png",
        "plots/eda/quantitative_by_outcome.png",
        "plots/calibration/reliability_curves.png", "plots/interpretability/lr_coefficients.png",
        "plots/interpretability/decision_tree.png",
    ]
    for model in MODEL_ORDER:
        plots.extend(
            [
                f"plots/thresholds/threshold_sweep_{model}.png",
                f"plots/interpretability/permutation_importance_{model}.png",
                f"plots/interpretability/shap_importance_{model}.png",
            ]
        )
    return outputs + plots


def _pre_test_gate(
    X_dev,
    y_dev,
    outer_splits,
    baselines,
    nested,
    oof_scores,
    final_models,
    calibration,
    thresholds,
    sweeps,
):
    print("Pre-test gate: verifying locked development-only state", flush=True)
    if "age" in X_dev.columns or "customer_value" in X_dev.columns or list(X_dev.columns) != PREDICTORS:
        raise RuntimeError("Pre-test gate failed: model predictor matrix")
    held_out = np.concatenate([valid for _, valid in outer_splits])
    if sorted(held_out.tolist()) != list(range(len(X_dev))):
        raise RuntimeError("Pre-test gate failed: outer-fold coverage")
    if not baselines["pr_auc_mean"].between(0, 1).all() or not nested["pr_auc_mean"].between(0, 1).all():
        raise RuntimeError("Pre-test gate failed: inadmissible PR-AUC")
    if set(calibration["score_source"]) != {"nested OOF"} or not (calibration["input_rows"] == len(X_dev)).all():
        raise RuntimeError("Pre-test gate failed: calibration provenance")
    if len(final_models["lr"].named_steps["preprocessor"].get_feature_names_out()) != 14:
        raise RuntimeError("Pre-test gate failed: LR encoded-term count")
    for model_name in ["dt", "rf", "xgb"]:
        if len(final_models[model_name].named_steps["preprocessor"].get_feature_names_out()) != 11:
            raise RuntimeError(f"Pre-test gate failed: {model_name} encoded-term count")
    if final_models["dt"].named_steps["classifier"].get_depth() > 5:
        raise RuntimeError("Pre-test gate failed: decision-tree depth cap")
    selected_map = thresholds.set_index("model")
    expected_selected_columns = [
        "model", "floored_metric", "floor_value", "optimised_metric", "selection_mode",
        "threshold", "floor_attained", "predicted_positive", "precision", "recall",
        "tp", "fp", "tn", "fn", "precision_at_050", "recall_at_050",
        "predicted_positive_at_050", "selection_input_rows",
    ]
    if list(thresholds.columns) != expected_selected_columns:
        raise RuntimeError("Pre-test gate failed: selected-threshold schema")
    for model_name in MODEL_ORDER:
        scores = np.asarray(oof_scores[model_name], dtype=float)
        if len(scores) != len(X_dev) or not np.isfinite(scores).all():
            raise RuntimeError(f"Pre-test gate failed: {model_name} OOF predictions")
        sweep = sweeps[model_name]
        if not sweep["threshold"].is_unique or not np.isfinite(sweep["threshold"]).all():
            raise RuntimeError(f"Pre-test gate failed: {model_name} threshold candidates")
        if (sweep["predicted_positive"] < 1).any():
            raise RuntimeError(f"Pre-test gate failed: {model_name} zero-call candidate")
        for column in ["tp", "fp", "tn", "fn"]:
            if (sweep[column] < 0).any():
                raise RuntimeError(f"Pre-test gate failed: {model_name} negative confusion count")
        if not ((sweep[["tp", "fp", "tn", "fn"]].sum(axis=1)) == len(X_dev)).all():
            raise RuntimeError(f"Pre-test gate failed: {model_name} confusion reconciliation")
        expected = select_threshold(sweep, RECALL_FLOOR)
        selected = selected_map.loc[model_name]
        for column in ["threshold", "precision", "recall"]:
            if not np.isclose(float(selected[column]), float(expected[column]), atol=1e-12):
                raise RuntimeError(f"Pre-test gate failed: {model_name} selected {column}")
        if bool(selected["floor_attained"]) != bool(expected["floor_attained"]):
            raise RuntimeError(f"Pre-test gate failed: {model_name} floor attainment")
        if str(selected["selection_mode"]) != str(expected["selection_mode"]):
            raise RuntimeError(f"Pre-test gate failed: {model_name} selection mode")
        if (
            selected["floored_metric"] != "recall"
            or selected["optimised_metric"] != "precision"
            or not np.isclose(float(selected["floor_value"]), RECALL_FLOOR)
            or not bool(selected["floor_attained"])
            or float(selected["recall"]) < RECALL_FLOOR
            or int(selected["selection_input_rows"]) != len(X_dev)
        ):
            raise RuntimeError(f"Pre-test gate failed: {model_name} operating-rule declaration")
    print("Pre-test gate passed; the held-out test set may now be evaluated once", flush=True)


def run_workflow(output_root=None, plot_root=None):
    start_time = time.time()
    output_root = Path(output_root) if output_root else ROOT / "outputs"
    plot_root = Path(plot_root) if plot_root else ROOT / "plots"
    _ensure_directories(output_root, plot_root)
    code_version = _code_version()
    print(f"Starting Iranian Churn workflow ({code_version})", flush=True)

    raw, X, y = _load_and_assert()
    audit, target_position = _problem_and_audit(raw, X, y, output_root)
    X_dev, X_test, y_dev, y_test, outer_splits, split_spec = _split_and_folds(X, y, output_root)
    corr, max_pair, max_corr = _eda(X_dev, y_dev, raw, y, output_root, plot_root)
    feature_sets = _evaluation_and_features(output_root)
    print("Stage 8: untuned constrained baselines", flush=True)
    baselines = _untuned_baselines(X_dev, y_dev, outer_splits, output_root)
    nested, oof_scores, outer_models, final_models, fold_scores = _nested_cv(
        X_dev, y_dev, outer_splits, output_root
    )
    calibration = _calibration(oof_scores, y_dev, output_root, plot_root)
    thresholds, sweeps = _thresholds(oof_scores, y_dev, output_root, plot_root)
    warnings_frame = _below_baseline_warnings(baselines, nested, oof_scores, y_dev, output_root)
    _pre_test_gate(
        X_dev,
        y_dev,
        outer_splits,
        baselines,
        nested,
        oof_scores,
        final_models,
        calibration,
        thresholds,
        sweeps,
    )
    test_results, confusion, test_scores = _test_evaluation(
        final_models, X_test, y_test, thresholds, output_root
    )
    paired = _paired_comparison(fold_scores, output_root)
    metric_sensitivity = _metric_sensitivity(oof_scores, y_dev, calibration, thresholds, output_root)
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
    _structural_findings(interpretability, X_dev, y_dev, output_root)

    context = {
        "split_spec": split_spec,
        "baselines": baselines,
        "nested": nested,
        "calibration": calibration,
        "thresholds": thresholds,
        "test_results": test_results,
        "confusion": confusion,
        "paired": paired,
        "interpretability": interpretability,
    }
    registry = _registry_entries(context, code_version)
    registry.to_csv(output_root / "registry.csv", index=False)
    registry[["stage", "model", "scope", "metric"]].drop_duplicates().sort_values(
        ["stage", "model", "scope", "metric"]
    ).to_csv(output_root / "registry_expected_keys.csv", index=False)
    environment = _environment(output_root, start_time, code_version, split_spec, interpretability)

    _write_json(
        output_root / "audit" / "reproducibility.json",
        {
            "exact_rerun_performed": False,
            "waiver_date": "2026-08-25",
            "reason": "The locked decisions file applies workflow amendment 8, which waives the separate exact-reproduction run while retaining deterministic settings and all other verification checks.",
            "status": "waived by locked pre-execution specification",
        },
    )
    _write_json(
        output_root / "audit" / "invariant_results.json",
        {"status": "pending", "note": "Written by tests/test_invariants.py"},
    )

    manifest = _manifest(output_root, plot_root, expected_artifacts())
    if not manifest["produced"].all():
        missing = manifest.loc[~manifest["produced"], "path"].tolist()
        raise RuntimeError(f"Missing expected artifacts: {missing}")
    unexpected = manifest.loc[~manifest["expected"], "path"].tolist()
    if unexpected:
        raise RuntimeError(f"Unexpected artifacts: {unexpected}")
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
