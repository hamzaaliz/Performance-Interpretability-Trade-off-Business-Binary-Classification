import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.tree import export_text, plot_tree

from config import (
    DISPLAY_NAMES,
    LR_USED_FEATURES,
    MODEL_DISPLAY,
    MODEL_ORDER,
    NOMINAL,
    PREDICTORS,
    PROJECT_ROOT,
    SEED,
    SHAP_BACKGROUND,
    SHAP_ROW_CAP,
)
from modules.preprocessing import encoded_feature_metadata


_shared_path = PROJECT_ROOT / "South German Credit" / "modules" / "interpretability.py"
_shared_spec = importlib.util.spec_from_file_location("_shared_interpretability", _shared_path)
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)


def cross_fitted_grouped_permutation(*args, **kwargs):
    return _shared.cross_fitted_grouped_permutation(*args, **kwargs)


def _safe_spearman(a, b):
    return _shared._safe_spearman(a, b)


def _complexity(model_name, pipeline):
    return _shared._complexity(model_name, pipeline)


def _stratified_indices(y, n, seed):
    return _shared._stratified_indices(y, n, seed)


def native_evidence(model_name, pipeline, output_dir, plot_dir):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    encoded_names, raw_names, display_terms = encoded_feature_metadata(preprocessor)
    if model_name == "lr":
        coefficients = classifier.coef_[0]
        frame = pd.DataFrame(
            {
                "encoded_term": encoded_names,
                "display_term": display_terms,
                "coefficient": coefficients,
                "sign": np.where(
                    coefficients > 0,
                    "positive",
                    np.where(coefficients < 0, "negative", "zero"),
                ),
                "odds_ratio": np.exp(coefficients),
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        frame.to_csv(output_dir / "lr_coefficients.csv", index=False)
        native = frame[["display_term", "absolute_coefficient"]].rename(
            columns={"display_term": "display_name", "absolute_coefficient": "importance"}
        )
        native.insert(0, "feature", "encoded-term-only")
        native["rank"] = np.arange(1, len(native) + 1)
        native["aggregation"] = "No variable-level native aggregate"
        native.to_csv(output_dir / "native_importance_lr.csv", index=False)

        top = frame.head(15).sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(10, 7))
        colours = np.where(top["coefficient"] >= 0, "#C44E52", "#4472C4")
        ax.barh(top["display_term"], top["coefficient"], color=colours)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Logistic-regression coefficient")
        ax.set_title("Largest encoded-term coefficients by absolute magnitude")
        fig.tight_layout()
        fig.savefig(plot_dir / "lr_coefficients.png", dpi=180)
        plt.close(fig)
        return None, len(encoded_names)

    if model_name == "dt":
        rules = export_text(classifier, feature_names=display_terms, decimals=4)
        (output_dir / "dt_structure.txt").write_text(rules, encoding="utf-8")
        fig, ax = plt.subplots(figsize=(28, 14))
        plot_tree(
            classifier,
            feature_names=display_terms,
            class_names=["No default", "Default"],
            filled=True,
            rounded=True,
            proportion=True,
            fontsize=7,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / "decision_tree.png", dpi=180)
        plt.close(fig)

    term_importance = np.asarray(classifier.feature_importances_, dtype=float)
    term_frame = pd.DataFrame(
        {
            "encoded_term": encoded_names,
            "original_feature": raw_names,
            "display_term": display_terms,
            "term_importance": term_importance,
        }
    )
    grouped = (
        term_frame.groupby("original_feature", as_index=False)["term_importance"]
        .sum()
        .rename(columns={"original_feature": "feature", "term_importance": "importance"})
    )
    grouped["display_name"] = grouped["feature"].map(DISPLAY_NAMES)
    grouped = grouped.sort_values("importance", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped.to_csv(output_dir / f"native_importance_{model_name}.csv", index=False)
    term_frame.to_csv(output_dir / f"native_importance_terms_{model_name}.csv", index=False)
    return grouped, len(encoded_names)


def shap_evidence(model_name, pipeline, X_dev, y_dev, output_dir, plot_dir):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = np.asarray(preprocessor.transform(X_dev), dtype=float)
    encoded_names, raw_names, display_terms = encoded_feature_metadata(preprocessor)
    n_explain = min(SHAP_ROW_CAP, len(X_dev))
    explain_idx = _stratified_indices(y_dev, n_explain, SEED)
    background_idx = _stratified_indices(y_dev, min(SHAP_BACKGROUND, len(X_dev)), SEED)
    explain_matrix = transformed[explain_idx]
    background = transformed[background_idx]

    if model_name == "lr":
        explainer = shap.LinearExplainer(classifier, background)
        explanation = explainer(explain_matrix)
        values = np.asarray(explanation.values, dtype=float)
        output_scale = "log-odds (linear model raw output)"
        feature_universe = LR_USED_FEATURES
    else:
        explainer = shap.TreeExplainer(
            classifier,
            data=background,
            feature_perturbation="interventional",
            model_output="raw",
        )
        explanation = explainer(explain_matrix, check_additivity=False)
        values = np.asarray(explanation.values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, 1]
        output_scale = (
            "raw class-1 probability output"
            if model_name in {"dt", "rf"}
            else "raw margin (log-odds)"
        )
        feature_universe = PREDICTORS
    if values.shape != explain_matrix.shape:
        raise RuntimeError(
            f"Unexpected SHAP shape for {model_name}: {values.shape}, expected {explain_matrix.shape}"
        )

    term_frame = pd.DataFrame(
        {
            "encoded_term": encoded_names,
            "display_term": display_terms,
            "original_feature": raw_names,
            "mean_absolute_shap": np.mean(np.abs(values), axis=0),
            "mean_signed_shap": np.mean(values, axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    term_frame["rank"] = np.arange(1, len(term_frame) + 1)
    term_frame.to_csv(output_dir / f"shap_terms_{model_name}.csv", index=False)

    rows = []
    X_explain = X_dev.iloc[explain_idx]
    for feature in feature_universe:
        positions = [index for index, raw in enumerate(raw_names) if raw == feature]
        if not positions:
            raise RuntimeError(f"No transformed SHAP term maps to model feature {feature}")
        signed = values[:, positions].sum(axis=1)
        importance = float(np.mean(np.abs(signed)))
        if feature in NOMINAL:
            rho = np.nan
            direction = "category-dependent"
        else:
            rho = _safe_spearman(X_explain[feature].to_numpy(), signed)
            if not np.isfinite(rho) or abs(rho) < 0.10:
                direction = "mixed/weak"
            else:
                direction = "positive" if rho > 0 else "negative"
        rows.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "mean_absolute_shap": importance,
                "mean_signed_shap": float(np.mean(signed)),
                "direction_rho": rho,
                "direction": direction,
                "n_rows": int(n_explain),
                "background_n": int(len(background_idx)),
                "output_scale": output_scale,
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(output_dir / f"shap_summary_{model_name}.csv", index=False)

    top = summary.head(12).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["display_name"], top["mean_absolute_shap"], color="#55A868")
    ax.set_xlabel("Mean absolute variable-level SHAP contribution")
    ax.set_title(f"{MODEL_DISPLAY[model_name]}: SHAP ranking")
    fig.tight_layout()
    fig.savefig(plot_dir / f"shap_importance_{model_name}.png", dpi=180)
    plt.close(fig)
    return summary, term_frame, len(encoded_names), output_scale


def run_interpretability(
    final_models,
    outer_models,
    outer_splits,
    X_dev,
    y_dev,
    calibration_frame,
    output_dir,
    plot_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    permutation = {}
    native = {}
    shap_summaries = {}
    encoded_counts = {}
    coverage_by_model = {}

    for model_name in MODEL_ORDER:
        native[model_name], encoded_counts[model_name] = native_evidence(
            model_name, final_models[model_name], output_dir, plot_dir
        )
        permutation[model_name], coverage_by_model[model_name] = cross_fitted_grouped_permutation(
            model_name,
            outer_models[model_name],
            outer_splits,
            X_dev,
            y_dev,
            output_dir,
            plot_dir,
        )
        shap_summaries[model_name], _, shap_count, _ = shap_evidence(
            model_name, final_models[model_name], X_dev, y_dev, output_dir, plot_dir
        )
        if shap_count != encoded_counts[model_name]:
            raise RuntimeError(f"Encoded feature count mismatch for {model_name}")

    combined = pd.DataFrame(
        {"feature": PREDICTORS, "display_name": [DISPLAY_NAMES[item] for item in PREDICTORS]}
    )
    for model_name in MODEL_ORDER:
        perm_map = permutation[model_name].set_index("feature")
        shap_map = shap_summaries[model_name].set_index("feature")
        combined[f"{model_name}_perm_rank"] = combined["feature"].map(perm_map["rank"])
        combined[f"{model_name}_shap_rank"] = combined["feature"].map(shap_map["rank"])
        combined[f"{model_name}_direction"] = combined["feature"].map(shap_map["direction"])
    combined.to_csv(output_dir / "feature_rankings_combined.csv", index=False)

    top10 = {
        model: set(permutation[model].sort_values("rank").head(10)["feature"])
        for model in MODEL_ORDER
    }
    calibration_map = calibration_frame.set_index("model")["slope"].to_dict()
    summary_rows = []
    for model_name in MODEL_ORDER:
        complexity = _complexity(model_name, final_models[model_name])
        common_features = LR_USED_FEATURES if model_name == "lr" else PREDICTORS
        perm_order = permutation[model_name].set_index("feature").loc[common_features, "rank"]
        shap_order = shap_summaries[model_name].set_index("feature").loc[common_features, "rank"]
        perm_shap_corr = _safe_spearman(perm_order, shap_order)
        if model_name == "lr":
            native_perm_corr = np.nan
            blank_reason = (
                "LR native coefficients remain at encoded-term level; LR SHAP is blank for "
                "the five pruned bill variables; permutation-versus-SHAP rho uses 18 common inputs"
            )
        else:
            native_order = native[model_name].set_index("feature").loc[PREDICTORS, "rank"]
            native_perm_corr = _safe_spearman(native_order, perm_order)
            blank_reason = ""
        row = {
            "model": model_name,
            "n_features_used": encoded_counts[model_name],
            "n_nonzero_coef": complexity["n_nonzero_coef"],
            "tree_depth": complexity["tree_depth"],
            "n_leaves": complexity["n_leaves"],
            "complexity_note": complexity["complexity_note"],
            "native_vs_perm_rank_corr": native_perm_corr,
            "perm_vs_shap_rank_corr": perm_shap_corr,
            "calibration_slope": calibration_map[model_name],
            "blank_reason": blank_reason,
        }
        for other in MODEL_ORDER:
            row[f"top10_overlap_{other}"] = len(top10[model_name] & top10[other])
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "interpretability_summary.csv", index=False)

    return {
        "permutation": permutation,
        "native": native,
        "shap": shap_summaries,
        "combined": combined,
        "summary": summary_frame,
        "coverage": coverage_by_model,
    }
