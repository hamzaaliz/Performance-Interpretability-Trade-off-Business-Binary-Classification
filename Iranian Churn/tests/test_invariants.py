import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODELS = ["lr", "dt", "rf", "xgb"]
DEVELOPMENT_ROWS = 2520
TEST_ROWS = 630
RECALL_FLOOR = 0.80
LR_FEATURES = {
    "call_failure", "complains", "subscription_length", "charge_amount",
    "mean_call_duration", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status",
}
TREE_FEATURES = {
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status",
}


def require(condition, message, checks):
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def _locked_choice(sweep):
    eligible = sweep.loc[sweep["recall"] >= RECALL_FLOOR]
    if len(eligible):
        row = eligible.sort_values(
            ["precision", "recall", "threshold"], ascending=[False, False, False]
        ).iloc[0]
        return row, True, "constrained"
    row = sweep.sort_values(
        ["recall", "precision", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return row, False, "max_recall_fallback"


def _rank_check(frame, value_column):
    expected = frame.sort_values(
        [value_column, "feature"], ascending=[False, True], kind="mergesort"
    )["feature"].tolist()
    observed = frame.sort_values("rank")["feature"].tolist()
    return observed == expected and sorted(frame["rank"].astype(int)) == list(range(1, len(frame) + 1))


def main():
    checks = []
    split = json.loads((OUTPUTS / "split" / "split_spec.json").read_text(encoding="utf-8"))
    development = set(split["development_source_indices"])
    test = set(split["test_source_indices"])
    require(development.isdisjoint(test), "Development and test source indices are disjoint", checks)
    require(len(development) == DEVELOPMENT_ROWS and len(test) == TEST_ROWS, "Split sizes are 2,520/630", checks)
    require(split["development_positives"] == 396 and split["test_positives"] == 99, "Split positives are 396/99", checks)
    require(split["test_rows_with_development_feature_pattern"] == 100, "One hundred test rows share a development feature pattern", checks)
    require(split["seed"] == 42 and split["shuffle"] is True, "Split seed and shuffle rule are locked", checks)

    folds = joblib.load(OUTPUTS / "split" / "fold_indices.pkl")
    held_out = np.concatenate([valid for _, valid in folds])
    require(sorted(held_out.tolist()) == list(range(DEVELOPMENT_ROWS)), "Outer held-out folds cover development once", checks)
    require(all(set(train).isdisjoint(set(valid)) for train, valid in folds), "Outer train/validation indices are disjoint", checks)
    require(hashlib.sha256(pickle.dumps(folds)).hexdigest() == split["fold_indices_sha256"], "Outer-fold hash reproduces", checks)

    oof_by_model = {}
    fold_columns = []
    for model in MODELS:
        oof = pd.read_csv(OUTPUTS / "modelling" / f"oof_predictions_{model}.csv")
        oof_by_model[model] = oof
        require(len(oof) == DEVELOPMENT_ROWS, f"{model} OOF has development length", checks)
        require(oof["row_index"].tolist() == list(range(DEVELOPMENT_ROWS)), f"{model} OOF row order is complete", checks)
        require(np.isfinite(oof["y_score"]).all(), f"{model} OOF scores are finite", checks)
        fold_columns.append(oof["fold"].to_numpy())
    require(all(np.array_equal(fold_columns[0], values) for values in fold_columns[1:]), "Fold assignments are identical across models", checks)

    thresholds = pd.read_csv(
        OUTPUTS / "modelling" / "thresholds_selected.csv", float_precision="round_trip"
    ).set_index("model")
    expected_threshold_columns = [
        "floored_metric", "floor_value", "optimised_metric", "selection_mode", "threshold",
        "floor_attained", "predicted_positive", "precision", "recall", "tp", "fp", "tn",
        "fn", "precision_at_050", "recall_at_050", "predicted_positive_at_050",
        "selection_input_rows",
    ]
    require(list(thresholds.columns) == expected_threshold_columns, "Selected-threshold schema is exact", checks)
    for model in MODELS:
        selected = thresholds.loc[model]
        sweep = pd.read_csv(
            OUTPUTS / "modelling" / f"threshold_sweep_{model}.csv",
            float_precision="round_trip",
        )
        require(list(sweep.columns) == ["threshold", "predicted_positive", "precision", "recall", "tp", "fp", "tn", "fn"], f"{model} threshold-sweep schema is exact", checks)
        require(sweep["threshold"].is_unique and np.isfinite(sweep["threshold"]).all(), f"{model} threshold grid is distinct and finite", checks)
        require((sweep["predicted_positive"] >= 1).all(), f"{model} sweep has no empty prediction candidate", checks)
        require((sweep[["tp", "fp", "tn", "fn"]].sum(axis=1) == DEVELOPMENT_ROWS).all(), f"{model} sweep counts reconcile", checks)
        require(len(sweep.loc[np.isclose(sweep["threshold"], selected["threshold"], atol=0, rtol=0)]) == 1, f"{model} selected threshold appears once", checks)
        expected, attained, mode = _locked_choice(sweep)
        require(bool(selected["floor_attained"]) == attained and selected["selection_mode"] == mode, f"{model} threshold mode reproduces", checks)
        require(np.isclose(selected["threshold"], expected["threshold"], atol=1e-12), f"{model} threshold tie-break reproduces", checks)
        require(selected["floored_metric"] == "recall" and selected["optimised_metric"] == "precision" and np.isclose(selected["floor_value"], RECALL_FLOOR), f"{model} floor declaration is locked", checks)
        require(bool(selected["floor_attained"]) and selected["recall"] >= RECALL_FLOOR, f"{model} guaranteed recall floor is attained", checks)
        require(selected["selection_input_rows"] == DEVELOPMENT_ROWS, f"{model} threshold selection used development length", checks)
        oof = oof_by_model[model]
        pred050 = oof["y_score"].to_numpy() >= 0.50
        require(int(pred050.sum()) == selected["predicted_positive_at_050"], f"{model} conventional support reproduces", checks)
        require(np.isclose(precision_score(oof["y_true"], pred050, zero_division=0), selected["precision_at_050"]), f"{model} conventional precision reproduces", checks)
        require(np.isclose(recall_score(oof["y_true"], pred050, zero_division=0), selected["recall_at_050"]), f"{model} conventional recall reproduces", checks)

    calibration = pd.read_csv(OUTPUTS / "modelling" / "calibration.csv")
    require((calibration["input_rows"] == DEVELOPMENT_ROWS).all() and set(calibration["score_source"]) == {"nested OOF"}, "Calibration uses nested OOF development scores", checks)

    nested = pd.read_csv(OUTPUTS / "modelling" / "nested_cv.csv")
    require(nested["pr_auc_mean"].between(0, 1).all(), "Nested PR-AUC values are admissible", checks)
    baselines = pd.read_csv(OUTPUTS / "modelling" / "baseline_cv.csv").set_index("model")
    warnings = pd.read_csv(OUTPUTS / "audit" / "warnings.csv")
    below = set(nested.loc[nested["pr_auc_mean"] <= float(baselines.loc["dummy", "pr_auc_mean"]), "model"])
    warned = set(warnings["model"]) if len(warnings) else set()
    require(below.issubset(warned), "Below-baseline results carry warnings", checks)

    test_results = pd.read_csv(OUTPUTS / "modelling" / "test_results.csv")
    confusion = pd.read_csv(OUTPUTS / "modelling" / "confusion_matrices.csv")
    require((test_results["test_rows"] == TEST_ROWS).all() and (test_results["test_positives"] == 99).all(), "Holdout results use the locked test size", checks)
    require((confusion[["tp", "fp", "tn", "fn"]].sum(axis=1) == TEST_ROWS).all(), "Holdout confusion counts reconcile", checks)

    registry = pd.read_csv(OUTPUTS / "registry.csv")
    expected = pd.read_csv(OUTPUTS / "registry_expected_keys.csv")
    expected_registry_columns = ["dataset", "stage", "model", "scope", "metric", "value", "source_file", "seed_applicable", "seed", "code_version"]
    require(list(registry.columns) == expected_registry_columns, "Registry schema matches the cross-dataset protocol", checks)
    actual_keys = set(map(tuple, registry[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    expected_keys = set(map(tuple, expected[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    require(actual_keys == expected_keys, "Registry key set exactly matches its declaration", checks)
    require((registry["dataset"] == "Iranian Churn").all(), "Every registry row names Iranian Churn", checks)
    require(registry["code_version"].notna().all(), "Every registry row carries the code version", checks)
    require(registry.loc[registry["seed_applicable"], "seed"].notna().all(), "Seeded registry rows carry the seed", checks)

    interp = pd.read_csv(OUTPUTS / "interpretability" / "interpretability_summary.csv").set_index("model")
    require(float(interp.loc["dt", "tree_depth"]) <= 5, "Decision-tree realised depth does not exceed five", checks)
    require(int(interp.loc["lr", "n_features_used"]) == 14, "LR uses 14 encoded terms", checks)
    require((interp.loc[["dt", "rf", "xgb"], "n_features_used"].astype(int) == 11).all(), "Tree-family models use 11 terms", checks)
    lr_coeff = pd.read_csv(OUTPUTS / "interpretability" / "lr_coefficients.csv")
    lr_shap = pd.read_csv(OUTPUTS / "interpretability" / "shap_terms_lr.csv")
    require(set(lr_coeff["encoded_term"]) == set(lr_shap["encoded_term"]), "LR SHAP names match preprocessing output", checks)
    require(set(lr_coeff["locked_reference_metadata"]) == {"Age band 3 is the omitted reference level"}, "LR coefficient metadata records the reference band", checks)
    for model in MODELS:
        expected_features = LR_FEATURES if model == "lr" else TREE_FEATURES
        permutation = pd.read_csv(OUTPUTS / "interpretability" / f"permutation_importance_{model}.csv")
        shap_summary = pd.read_csv(OUTPUTS / "interpretability" / f"shap_summary_{model}.csv")
        require(set(permutation["feature"]) == expected_features, f"{model} permutation feature set is exact", checks)
        require(set(shap_summary["feature"]) == expected_features, f"{model} SHAP feature set is exact", checks)
        require((permutation["n_measurements"] == 150).all(), f"{model} permutation covers 5 x 30 measurements", checks)
        require(_rank_check(permutation, "importance_mean"), f"{model} permutation ranks are exact", checks)
        require(_rank_check(shap_summary, "mean_absolute_shap"), f"{model} SHAP ranks are exact", checks)
        require((shap_summary["n_rows"] == 2000).all(), f"{model} SHAP uses 2,000 rows", checks)
        require((shap_summary["background_n"] == 200).all(), f"{model} SHAP background uses 200 rows", checks)
        if model != "lr":
            native = pd.read_csv(OUTPUTS / "interpretability" / f"native_importance_{model}.csv")
            require(_rank_check(native, "importance"), f"{model} native ranks are exact", checks)

    combined = pd.read_csv(OUTPUTS / "interpretability" / "feature_rankings_combined.csv").set_index("feature")
    require(pd.isna(combined.loc["seconds_of_use", "lr_perm_rank"]) and pd.isna(combined.loc["seconds_of_use", "lr_shap_rank"]), "LR total-seconds ranks are blank", checks)
    for model in ["dt", "rf", "xgb"]:
        require(pd.isna(combined.loc["mean_call_duration", f"{model}_perm_rank"]) and pd.isna(combined.loc["mean_call_duration", f"{model}_shap_rank"]), f"{model} mean-duration ranks are blank", checks)

    applied = pd.read_csv(OUTPUTS / "audit" / "applied_rulings.csv")
    require(applied["verified"].all() and applied["ruling"].tolist() == [f"R{i}" for i in range(1, 18)], "All R1-R17 rulings are verified", checks)
    leakage = pd.read_csv(OUTPUTS / "audit" / "leakage_register.csv")
    require(leakage.empty and list(leakage.columns) == ["feature", "display_name", "availability_issue", "action"], "Leakage register is header-only as locked", checks)
    audit = pd.read_csv(OUTPUTS / "audit" / "data_audit.csv").set_index("audit_item")
    require(int(float(audit.loc["distinct_numbers_exceeds_frequency_rows", "observed_value"])) == 120, "Call-count ambiguity is recorded", checks)
    require(float(audit.loc["customer_value_max_abs_formula_residual", "observed_value"]) <= 5e-10, "Customer Value formula is verified", checks)
    require(int(float(audit.loc["dormant_conflicting_profiles", "observed_value"])) == 11 and int(float(audit.loc["rows_in_dormant_conflicting_profiles", "observed_value"])) == 56, "Dormant conflicting count correction is recorded", checks)
    feature_sets = pd.read_csv(OUTPUTS / "preprocessing" / "feature_sets_by_model.csv")
    require(set(feature_sets.loc[feature_sets["model"].eq("Logistic regression"), "expected_encoded_terms"]) == {14}, "Feature-set table records LR term count", checks)
    require(set(feature_sets.loc[~feature_sets["model"].eq("Logistic regression"), "expected_encoded_terms"]) == {11}, "Feature-set table records tree term count", checks)

    reproducibility = json.loads((OUTPUTS / "audit" / "reproducibility.json").read_text(encoding="utf-8"))
    require(reproducibility["exact_rerun_performed"] is False and reproducibility["waiver_date"] == "2026-08-25", "Exact reproduction waiver is recorded", checks)
    manifest = pd.read_csv(OUTPUTS / "manifest.csv")
    require(manifest["produced"].all() and manifest["expected"].all(), "Manifest contains all and only expected artefacts", checks)
    environment = json.loads((OUTPUTS / "audit" / "environment.json").read_text(encoding="utf-8"))
    require(environment["source_sha256"] == "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1", "Environment records the source hash", checks)
    require(environment["shap_rows_by_model"] == {model: 2000 for model in MODELS}, "Environment records SHAP coverage", checks)

    prohibited = ["statistically significant", "significantly", "p-value"]
    searched = []
    for suffix in ["*.md", "*.txt", "*.csv"]:
        searched.extend(OUTPUTS.rglob(suffix))
    corpus = "\n".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in searched)
    require(not any(term in corpus for term in prohibited), "Prohibited significance vocabulary is absent", checks)

    result = {"status": "passed", "checks": checks, "n_checks": len(checks)}
    (OUTPUTS / "audit" / "invariant_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
