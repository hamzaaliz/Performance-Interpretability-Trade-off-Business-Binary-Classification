import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODELS = ["lr", "dt", "rf", "xgb"]


def require(condition, message, checks):
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main():
    checks = []
    split = json.loads((OUTPUTS / "split" / "split_spec.json").read_text(encoding="utf-8"))
    development = set(split["development_source_indices"])
    test = set(split["test_source_indices"])
    require(development.isdisjoint(test), "Development and test source indices are disjoint", checks)
    require(len(development) == 800 and len(test) == 200, "Split sizes are 800/200", checks)

    folds = joblib.load(OUTPUTS / "split" / "fold_indices.pkl")
    held_out = np.concatenate([valid for _, valid in folds])
    require(sorted(held_out.tolist()) == list(range(800)), "Outer held-out folds cover every development row once", checks)
    require(all(set(train).isdisjoint(set(valid)) for train, valid in folds), "Outer train and validation indices are disjoint", checks)

    fold_columns = []
    for model in MODELS:
        oof = pd.read_csv(OUTPUTS / "modelling" / f"oof_predictions_{model}.csv")
        require(len(oof) == 800, f"{model} OOF predictions have development length", checks)
        require(oof["row_index"].tolist() == list(range(800)), f"{model} OOF row order is complete", checks)
        require(np.isfinite(oof["y_score"]).all(), f"{model} OOF scores are finite", checks)
        fold_columns.append(oof["fold"].to_numpy())
    require(all(np.array_equal(fold_columns[0], item) for item in fold_columns[1:]), "Fold assignments are identical across models", checks)

    thresholds = pd.read_csv(OUTPUTS / "modelling" / "thresholds_selected.csv")
    require((thresholds["selection_input_rows"] == 800).all(), "Threshold selection used development-length arrays", checks)
    require((thresholds["recall"] >= 0.75).all(), "Every selected threshold attains the recall floor", checks)
    calibration = pd.read_csv(OUTPUTS / "modelling" / "calibration.csv")
    require((calibration["input_rows"] == 800).all(), "Calibration used OOF development arrays", checks)

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
    actual_keys = set(map(tuple, registry[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    expected_keys = set(map(tuple, expected[["stage", "model", "scope", "metric"]].fillna("").to_numpy()))
    require(actual_keys == expected_keys, "Registry key set exactly matches its declaration", checks)
    require(registry["code_version"].notna().all(), "Every registry row carries the code version", checks)
    require(registry.loc[registry["seed_applicable"], "seed"].notna().all(), "Seeded registry rows carry the seed", checks)

    interp = pd.read_csv(OUTPUTS / "interpretability" / "interpretability_summary.csv").set_index("model")
    require(float(interp.loc["dt", "tree_depth"]) <= 5, "Decision-tree realised depth does not exceed five", checks)
    lr_coeff = pd.read_csv(OUTPUTS / "interpretability" / "lr_coefficients.csv")
    lr_shap = pd.read_csv(OUTPUTS / "interpretability" / "shap_terms_lr.csv")
    require(set(lr_coeff["encoded_term"]) == set(lr_shap["encoded_term"]), "LR SHAP feature names match fitted preprocessing output", checks)
    for model in MODELS:
        permutation = pd.read_csv(OUTPUTS / "interpretability" / f"permutation_importance_{model}.csv")
        shap_summary = pd.read_csv(OUTPUTS / "interpretability" / f"shap_summary_{model}.csv")
        require((permutation["n_measurements"] == 150).all(), f"{model} permutation evidence covers 5 folds x 30 repeats", checks)
        require((shap_summary["n_rows"] == 800).all(), f"{model} SHAP uses development length", checks)
        require((shap_summary["background_n"] == 200).all(), f"{model} SHAP background has 200 rows", checks)

    manifest = pd.read_csv(OUTPUTS / "manifest.csv")
    require(manifest["produced"].all(), "Every expected artefact is present", checks)
    result = {"status": "passed", "checks": checks, "n_checks": len(checks)}
    (OUTPUTS / "audit" / "invariant_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
