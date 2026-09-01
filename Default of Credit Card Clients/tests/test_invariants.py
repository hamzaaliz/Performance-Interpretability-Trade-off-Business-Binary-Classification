import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SGC_REGISTRY = ROOT.parent / "South German Credit" / "outputs" / "registry.csv"
MODELS = ["lr", "dt", "rf", "xgb"]
LR_PRUNED = {
    "bill_amount_august",
    "bill_amount_july",
    "bill_amount_june",
    "bill_amount_may",
    "bill_amount_april",
}


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
    require(len(development) == 24000 and len(test) == 6000, "Split sizes are 24,000/6,000", checks)
    require(split["development_positives"] == 5309 and split["test_positives"] == 1327, "Stratified positive counts reconcile", checks)

    folds = joblib.load(OUTPUTS / "split" / "fold_indices.pkl")
    held_out = np.concatenate([valid for _, valid in folds])
    require(sorted(held_out.tolist()) == list(range(24000)), "Outer held-out folds cover every development row once", checks)
    require(all(set(train).isdisjoint(set(valid)) for train, valid in folds), "Outer train and validation indices are disjoint", checks)

    fold_columns = []
    for model in MODELS:
        oof = pd.read_csv(OUTPUTS / "modelling" / f"oof_predictions_{model}.csv")
        require(len(oof) == 24000, f"{model} OOF predictions have development length", checks)
        require(oof["row_index"].tolist() == list(range(24000)), f"{model} OOF row order is complete", checks)
        require(np.isfinite(oof["y_score"]).all(), f"{model} OOF scores are finite", checks)
        fold_columns.append(oof["fold"].to_numpy())
    require(all(np.array_equal(fold_columns[0], item) for item in fold_columns[1:]), "Fold assignments are identical across models", checks)

    thresholds = pd.read_csv(OUTPUTS / "modelling" / "thresholds_selected.csv")
    require((thresholds["selection_input_rows"] == 24000).all(), "Threshold selection used development-length arrays", checks)
    require((thresholds["recall"] >= 0.65).all(), "Every selected threshold attains the 0.65 recall floor", checks)
    calibration = pd.read_csv(OUTPUTS / "modelling" / "calibration.csv")
    require((calibration["input_rows"] == 24000).all(), "Calibration used OOF development arrays", checks)

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
    pr_auc_rows = registry[registry["metric"].str.contains("pr_auc|PR-AUC", regex=True, na=False)]
    require(np.isfinite(pr_auc_rows["value"]).all() and pr_auc_rows["value"].between(0, 1).all(), "Every registered PR-AUC is finite and admissible", checks)
    if SGC_REGISTRY.exists():
        sgc = pd.read_csv(SGC_REGISTRY)
        require(list(registry.columns) == list(sgc.columns), "Registry columns match South German Credit", checks)
        canonical = {
            "dataset": "string",
            "stage": "Int64",
            "model": "string",
            "scope": "string",
            "metric": "string",
            "value": "Float64",
            "source_file": "string",
            "seed_applicable": "boolean",
            "seed": "Int64",
            "code_version": "string",
        }
        registry_types = [str(dtype) for dtype in registry.astype(canonical).dtypes]
        sgc_types = [str(dtype) for dtype in sgc.astype(canonical).dtypes]
        require(registry_types == sgc_types, "Registry dtypes match South German Credit under the canonical nullable schema", checks)

    interp = pd.read_csv(OUTPUTS / "interpretability" / "interpretability_summary.csv").set_index("model")
    require(float(interp.loc["dt", "tree_depth"]) <= 5, "Decision-tree realised depth does not exceed five", checks)
    require((interp["n_features_used"] == 28).all(), "Every fitted design matrix contains 28 encoded terms", checks)
    lr_coeff = pd.read_csv(OUTPUTS / "interpretability" / "lr_coefficients.csv")
    lr_shap_terms = pd.read_csv(OUTPUTS / "interpretability" / "shap_terms_lr.csv")
    require(set(lr_coeff["encoded_term"]) == set(lr_shap_terms["encoded_term"]), "LR SHAP feature names match fitted preprocessing output", checks)
    combined = pd.read_csv(OUTPUTS / "interpretability" / "feature_rankings_combined.csv")
    require(len(combined) == 23, "Combined feature rankings contain exactly 23 original features", checks)
    pruned = combined[combined["feature"].isin(LR_PRUNED)]
    require(pruned["lr_shap_rank"].isna().all() and pruned["lr_direction"].isna().all(), "LR SHAP cells are blank for the five pruned bills", checks)

    for model in MODELS:
        permutation = pd.read_csv(OUTPUTS / "interpretability" / f"permutation_importance_{model}.csv")
        shap_summary = pd.read_csv(OUTPUTS / "interpretability" / f"shap_summary_{model}.csv")
        require(len(permutation) == 23, f"{model} permutation output covers 23 raw inputs", checks)
        require((permutation["n_measurements"] == 150).all(), f"{model} permutation evidence covers 5 folds x 30 repeats", checks)
        require((shap_summary["n_rows"] == 2000).all(), f"{model} SHAP uses 2,000 development rows", checks)
        require((shap_summary["background_n"] == 200).all(), f"{model} SHAP background has 200 rows", checks)
    lr_permutation = pd.read_csv(OUTPUTS / "interpretability" / "permutation_importance_lr.csv").set_index("feature")
    require(np.allclose(lr_permutation.loc[list(LR_PRUNED), "importance_mean"], 0.0), "Pruned LR bill inputs have zero pipeline permutation dependence", checks)

    manifest = pd.read_csv(OUTPUTS / "manifest.csv")
    require(manifest.loc[manifest["expected"], "produced"].all(), "Every expected artefact is present", checks)
    require(not manifest["unexpected"].any(), "No undeclared output artefacts are present", checks)
    result = {"status": "passed", "checks": checks, "n_checks": len(checks)}
    (OUTPUTS / "audit" / "invariant_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
