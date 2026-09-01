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
DEVELOPMENT_ROWS = 36168
TEST_ROWS = 9043
PRECISION_FLOOR = 0.50


def require(condition, message, checks):
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def _locked_choice(sweep):
    eligible = sweep.loc[sweep["precision"] >= PRECISION_FLOOR]
    if len(eligible):
        row = eligible.sort_values(
            ["recall", "precision", "threshold"], ascending=[False, False, False]
        ).iloc[0]
        return row, True, "constrained"
    row = sweep.sort_values(
        ["precision", "recall", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return row, False, "max_precision_fallback"


def _rank_check(frame, value_column):
    expected = frame.sort_values(
        [value_column, "feature"], ascending=[False, True], kind="mergesort"
    )["feature"].tolist()
    observed = frame.sort_values("rank")["feature"].tolist()
    return (
        observed == expected
        and sorted(frame["rank"].astype(int).tolist()) == list(range(1, len(frame) + 1))
        and len(frame.sort_values("rank").head(10)) == 10
    )


def main():
    checks = []
    split = json.loads((OUTPUTS / "split" / "split_spec.json").read_text(encoding="utf-8"))
    development = set(split["development_source_indices"])
    test = set(split["test_source_indices"])
    require(development.isdisjoint(test), "Development and test source indices are disjoint", checks)
    require(len(development) == DEVELOPMENT_ROWS and len(test) == TEST_ROWS, "Split sizes are 36,168/9,043", checks)
    require(split["development_positives"] == 4231 and split["test_positives"] == 1058, "Split positive counts are locked", checks)
    require(split["seed"] == 42 and split["shuffle"] is True, "Split seed and shuffle rule are locked", checks)

    folds = joblib.load(OUTPUTS / "split" / "fold_indices.pkl")
    held_out = np.concatenate([valid for _, valid in folds])
    require(sorted(held_out.tolist()) == list(range(DEVELOPMENT_ROWS)), "Outer held-out folds cover each development row once", checks)
    require(all(set(train).isdisjoint(set(valid)) for train, valid in folds), "Outer train and validation indices are disjoint", checks)
    fold_hash = hashlib.sha256(pickle.dumps(folds)).hexdigest()
    require(fold_hash == split["fold_indices_sha256"], "Persisted outer folds match the recorded hash", checks)

    fold_columns = []
    oof_by_model = {}
    for model in MODELS:
        oof = pd.read_csv(OUTPUTS / "modelling" / f"oof_predictions_{model}.csv")
        oof_by_model[model] = oof
        require(len(oof) == DEVELOPMENT_ROWS, f"{model} OOF predictions have development length", checks)
        require(oof["row_index"].tolist() == list(range(DEVELOPMENT_ROWS)), f"{model} OOF row order is complete", checks)
        require(np.isfinite(oof["y_score"]).all(), f"{model} OOF scores are finite", checks)
        fold_columns.append(oof["fold"].to_numpy())
    require(all(np.array_equal(fold_columns[0], item) for item in fold_columns[1:]), "Fold assignments are identical across models", checks)

    thresholds = pd.read_csv(OUTPUTS / "modelling" / "thresholds_selected.csv").set_index("model")
    expected_threshold_columns = [
        "floored_metric", "floor_value", "optimised_metric", "selection_mode", "threshold",
        "floor_attained", "predicted_positive", "precision", "recall", "tp", "fp", "tn",
        "fn", "precision_at_050", "recall_at_050", "predicted_positive_at_050",
        "selection_input_rows",
    ]
    require(list(thresholds.columns) == expected_threshold_columns, "Selected-threshold schema is exact", checks)
    for model in MODELS:
        selected = thresholds.loc[model]
        sweep = pd.read_csv(OUTPUTS / "modelling" / f"threshold_sweep_{model}.csv")
        require(list(sweep.columns) == ["threshold", "predicted_positive", "precision", "recall", "tp", "fp", "tn", "fn"], f"{model} threshold-sweep schema is exact", checks)
        require(sweep["threshold"].is_unique and np.isfinite(sweep["threshold"]).all(), f"{model} thresholds are distinct and finite", checks)
        require((sweep["predicted_positive"] >= 1).all(), f"{model} sweep has no zero-call candidate", checks)
        require((sweep[["tp", "fp", "tn", "fn"]].sum(axis=1) == DEVELOPMENT_ROWS).all(), f"{model} sweep counts reconcile", checks)
        require((sweep["tp"] + sweep["fp"] == sweep["predicted_positive"]).all(), f"{model} sweep support reconciles", checks)
        require(len(sweep.loc[np.isclose(sweep["threshold"], selected["threshold"], atol=0, rtol=0)]) == 1, f"{model} selected threshold appears once", checks)
        expected, attained, mode = _locked_choice(sweep)
        require(bool(selected["floor_attained"]) == attained and selected["selection_mode"] == mode, f"{model} attainment and fallback mode reproduce", checks)
        require(np.isclose(selected["threshold"], expected["threshold"], atol=1e-12), f"{model} threshold tie-break reproduces", checks)
        require(selected["floored_metric"] == "precision" and selected["optimised_metric"] == "recall" and np.isclose(selected["floor_value"], PRECISION_FLOOR), f"{model} floor declaration is locked", checks)
        require(selected["selection_input_rows"] == DEVELOPMENT_ROWS, f"{model} threshold selection used development length", checks)
        oof = oof_by_model[model]
        pred050 = oof["y_score"].to_numpy() >= 0.50
        p050 = precision_score(oof["y_true"], pred050, zero_division=0)
        r050 = recall_score(oof["y_true"], pred050, zero_division=0)
        require(int(pred050.sum()) == selected["predicted_positive_at_050"] and np.isclose(p050, selected["precision_at_050"]) and np.isclose(r050, selected["recall_at_050"]), f"{model} conventional-threshold reference reproduces", checks)

    calibration = pd.read_csv(OUTPUTS / "modelling" / "calibration.csv")
    require((calibration["input_rows"] == DEVELOPMENT_ROWS).all() and set(calibration["score_source"]) == {"nested OOF"}, "Calibration used nested OOF development arrays", checks)

    nested = pd.read_csv(OUTPUTS / "modelling" / "nested_cv.csv")
    require(nested["pr_auc_mean"].between(0, 1).all(), "Nested PR-AUC values are admissible", checks)
    baselines = pd.read_csv(OUTPUTS / "modelling" / "baseline_cv.csv").set_index("model")
    warnings = pd.read_csv(OUTPUTS / "audit" / "warnings.csv")
    dummy = float(baselines.loc["dummy", "pr_auc_mean"])
    below = set(nested.loc[nested["pr_auc_mean"] <= dummy, "model"])
    warned = set(warnings["model"]) if len(warnings) else set()
    require(below.issubset(warned), "Every below-baseline nested result has a warning", checks)

    registry = pd.read_csv(OUTPUTS / "registry.csv")
    expected = pd.read_csv(OUTPUTS / "registry_expected_keys.csv")
    expected_registry_columns = ["dataset", "stage", "model", "scope", "metric", "value", "source_file", "seed_applicable", "seed", "code_version"]
    require(list(registry.columns) == expected_registry_columns, "Registry schema matches the cross-dataset protocol", checks)
    actual_keys = set(map(tuple, registry[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    expected_keys = set(map(tuple, expected[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    require(actual_keys == expected_keys, "Registry key set exactly matches its declaration", checks)
    require((registry["dataset"] == "Bank Marketing").all(), "Every registry row names the dataset", checks)
    require(registry["code_version"].notna().all(), "Every registry row carries the code version", checks)
    require(registry.loc[registry["seed_applicable"], "seed"].notna().all(), "Seeded registry rows carry the seed", checks)

    interp = pd.read_csv(OUTPUTS / "interpretability" / "interpretability_summary.csv").set_index("model")
    require(float(interp.loc["dt", "tree_depth"]) <= 5, "Decision-tree realised depth does not exceed five", checks)
    require(int(interp.loc["lr", "n_features_used"]) == 42, "LR uses 42 encoded terms", checks)
    require((interp.loc[["dt", "rf", "xgb"], "n_features_used"].astype(int) == 47).all(), "Tree-family models use 47 encoded terms", checks)
    lr_coeff = pd.read_csv(OUTPUTS / "interpretability" / "lr_coefficients.csv")
    lr_shap = pd.read_csv(OUTPUTS / "interpretability" / "shap_terms_lr.csv")
    require(set(lr_coeff["encoded_term"]) == set(lr_shap["encoded_term"]), "LR SHAP names match fitted preprocessing output", checks)
    for model in MODELS:
        permutation = pd.read_csv(OUTPUTS / "interpretability" / f"permutation_importance_{model}.csv")
        shap_summary = pd.read_csv(OUTPUTS / "interpretability" / f"shap_summary_{model}.csv")
        require((permutation["n_measurements"] == 150).all(), f"{model} permutation evidence covers 5 x 30 measurements", checks)
        require(all(f"fold_mean_{fold}" in permutation.columns for fold in range(1, 6)), f"{model} permutation fold means are preserved", checks)
        require(_rank_check(permutation, "importance_mean"), f"{model} permutation ranks and tie-break are exact", checks)
        require(_rank_check(shap_summary, "mean_absolute_shap"), f"{model} SHAP ranks and tie-break are exact", checks)
        require((shap_summary["n_rows"] == 2000).all(), f"{model} SHAP uses the locked 2,000-row cap", checks)
        require((shap_summary["background_n"] == 200).all(), f"{model} SHAP background has 200 rows", checks)
        if model != "lr":
            native = pd.read_csv(OUTPUTS / "interpretability" / f"native_importance_{model}.csv")
            require(_rank_check(native, "importance"), f"{model} native ranks and tie-break are exact", checks)

    applied = pd.read_csv(OUTPUTS / "audit" / "applied_rulings.csv")
    require(applied["verified"].all(), "Every locked ruling is verified", checks)
    leakage = pd.read_csv(OUTPUTS / "audit" / "leakage_register.csv")
    require(leakage["feature"].tolist() == ["duration"], "Duration is the sole leakage-register entry", checks)
    feature_sets = pd.read_csv(OUTPUTS / "preprocessing" / "feature_sets_by_model.csv")
    require(set(feature_sets.loc[feature_sets["model"].eq("Logistic regression"), "expected_encoded_terms"]) == {42}, "Feature-set table records LR term count", checks)
    require(set(feature_sets.loc[~feature_sets["model"].eq("Logistic regression"), "expected_encoded_terms"]) == {47}, "Feature-set table records tree term count", checks)

    manifest = pd.read_csv(OUTPUTS / "manifest.csv")
    require(manifest["produced"].all(), "Every expected artefact is present", checks)
    environment = json.loads((OUTPUTS / "audit" / "environment.json").read_text(encoding="utf-8"))
    require(environment["source_sha256"] == "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d", "Environment records the locked source hash", checks)
    require(environment["shap_rows_by_model"] == {model: 2000 for model in MODELS}, "Environment records SHAP row coverage", checks)

    prohibited = ["statistically significant", "significantly", "p-value"]
    searched = []
    for suffix in ["*.md", "*.txt", "*.csv"]:
        searched.extend(OUTPUTS.rglob(suffix))
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore").lower() for path in searched)
    require(not any(term in text for term in prohibited), "Prohibited significance vocabulary is absent", checks)

    result = {"status": "passed", "checks": checks, "n_checks": len(checks)}
    (OUTPUTS / "audit" / "invariant_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
