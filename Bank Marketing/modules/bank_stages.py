import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.model_selection import train_test_split

from config import (
    BINARY,
    CALIBRATION_BINS,
    DATA_PATH,
    DISPLAY_NAMES,
    EXPECTED_LEVELS,
    LR_REFERENCE_LEVELS,
    MODEL_DISPLAY,
    MODEL_ORDER,
    NOMINAL,
    PRECISION_FLOOR,
    PREDICTORS,
    QUANTITATIVE,
    RAW_COLUMNS,
    SEED,
    TEST_SIZE,
)
from modules.evaluation import (
    calibration_statistics,
    metrics_at_threshold,
    reliability_bins,
    select_threshold,
    threshold_table,
)
from modules.preprocessing import apply_locked_rulings


SOURCE_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
INCONSISTENT_POUTCOME_INDICES = [40658, 41821, 42042, 43978, 45021]
POSITION_RATES = np.array(
    [
        0.029854046881910658, 0.0373811103738111, 0.04821942048219421,
        0.064145100641451, 0.06104844061048441, 0.0617120106171201,
        0.10464601769911504, 0.13113666519239275, 0.16080513160805132,
        0.4709135147091352,
    ]
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_and_assert():
    raw = pd.read_csv(DATA_PATH, sep=";")
    if list(raw.columns) != RAW_COLUMNS:
        raise RuntimeError("Source column list differs from the locked specification")
    if raw.shape != (45211, 17):
        raise RuntimeError(f"Unexpected shape: {raw.shape}")
    if _sha256(DATA_PATH) != SOURCE_SHA256:
        raise RuntimeError("Source SHA-256 mismatch")
    if DATA_PATH.stat().st_size != 4610348:
        raise RuntimeError("Source file size mismatch")
    if int(raw.isna().sum().sum()) != 0:
        raise RuntimeError("Unexpected missing values")
    object_columns = raw.select_dtypes(include="object").columns
    if raw[object_columns].apply(lambda column: column.str.strip().ne(column).any()).any():
        raise RuntimeError("Unexpected leading or trailing whitespace")
    if raw[object_columns].eq("").any().any():
        raise RuntimeError("Unexpected blank string")
    if int(raw.duplicated().sum()) != 0 or int(raw.drop(columns="y").duplicated().sum()) != 0:
        raise RuntimeError("Unexpected duplicate rows")
    if int(sum(dtype == "int64" for dtype in raw.dtypes)) != 7:
        raise RuntimeError("Unexpected numeric dtype count")
    for column, expected in EXPECTED_LEVELS.items():
        if set(raw[column]) != set(expected):
            raise RuntimeError(f"Category-level assertion failed for {column}")

    y = raw["y"].eq("yes").astype(int)
    if int(y.sum()) != 5289:
        raise RuntimeError("Target polarity assertion failed")
    if not raw["pdays"].eq(-1).equals(raw["previous"].eq(0)):
        raise RuntimeError("Never-contacted sentinel coincidence failed")
    contacted = raw["pdays"].ge(0)
    if int(contacted.sum()) != 8257:
        raise RuntimeError("Previously-contacted count failed")
    contacted_rho = float(spearmanr(raw.loc[contacted, "pdays"], raw.loc[contacted, "previous"]).statistic)
    if not np.isclose(contacted_rho, -0.101273278862, atol=1e-12):
        raise RuntimeError("Contacted-subset pdays/previous correlation failed")
    inconsistent = raw.index[
        raw["poutcome"].eq("unknown") & raw["previous"].ge(1) & raw["pdays"].ge(0)
    ].tolist()
    if inconsistent != INCONSISTENT_POUTCOME_INDICES:
        raise RuntimeError("Five-row poutcome assertion failed")
    expected_unknown = {"job": 288, "education": 1857, "contact": 13020, "poutcome": 36959}
    if {column: int(raw[column].eq("unknown").sum()) for column in expected_unknown} != expected_unknown:
        raise RuntimeError("Unknown-level count assertion failed")

    X = apply_locked_rulings(raw.drop(columns=["duration", "y"]))
    if list(X.columns) != PREDICTORS or X.shape != (45211, 15):
        raise RuntimeError("Locked predictor matrix assertion failed")
    if "duration" in X.columns:
        raise RuntimeError("Duration reached the model matrix")
    return raw, X, y


def problem_and_audit(raw, X, y, output_root):
    problem = """# Problem definition

- **Target:** Term-deposit subscription; subscription is the positive class.
- **Observation:** One recorded outcome from a telephone contact during a Portuguese bank direct-marketing campaign conducted between May 2008 and November 2010.
- **Primary decision:** Whether to place the recorded marketing call to a given client, scored immediately before that call.
- **Stakeholders:** The bank's direct-marketing function and the client receiving the contact.
- **False positive:** A client called who did not subscribe.
- **False negative:** A client not called who would have subscribed.
- **Evaluation stance:** Models are evaluated as ranking and classification systems. Probability statements are limited to calibration against the pooled delivered sample.
- **Holdout status:** The test set is an in-period held-out confirmation under a pre-inspected, date-ordered public dataset; development evidence governs comparison.
"""
    _write_text(output_root / "audit" / "problem_definition.md", problem)

    deciles = pd.qcut(np.arange(len(raw)), 10, labels=False)
    target_position = (
        pd.DataFrame({"position_decile": deciles + 1, "positive": y.to_numpy()})
        .groupby("position_decile", as_index=False)
        .agg(rows=("positive", "size"), positive_count=("positive", "sum"), positive_rate=("positive", "mean"))
    )
    if not np.allclose(target_position["positive_rate"].to_numpy(), POSITION_RATES, atol=5e-7):
        raise RuntimeError("Target-rate-by-position assertion failed")
    target_position.to_csv(output_root / "audit" / "target_rate_by_position.csv", index=False)

    contacted = raw["pdays"].ge(0)
    contacted_rho = float(spearmanr(raw.loc[contacted, "pdays"], raw.loc[contacted, "previous"]).statistic)
    full_rho = float(spearmanr(raw["pdays"], raw["previous"]).statistic)
    rare_levels = []
    for column in [*NOMINAL, *BINARY]:
        counts = raw[column].value_counts()
        rare_levels.extend(f"{column}={level}:{count}" for level, count in counts.items() if count < 1000)
    audit_rows = [
        ("source_filename", DATA_PATH.name, "Locked 17-column source variant"),
        ("retrieval_date", "2026-08-24", "Recorded local-project date"),
        ("licence", "CC BY 4.0", "UCI dataset licence"),
        ("sha256", _sha256(DATA_PATH), "Matches locked source hash"),
        ("file_size_bytes", DATA_PATH.stat().st_size, "Expected 4,610,348"),
        ("rows", len(raw), "Expected 45,211"),
        ("columns", raw.shape[1], "Expected 17"),
        ("predictors_used", X.shape[1], "Expected 15 after duration exclusion"),
        ("int64_columns", int(sum(dtype == "int64" for dtype in raw.dtypes)), "Expected seven"),
        ("string_columns", int(sum(dtype == "object" for dtype in raw.dtypes)), "Expected ten"),
        ("positive_count", int(y.sum()), "Subscribed"),
        ("negative_count", int((1 - y).sum()), "Did not subscribe"),
        ("positive_rate", float(y.mean()), "Pooled delivered-sample rate"),
        ("missing_values", int(raw.isna().sum().sum()), "Expected zero"),
        ("blank_or_whitespace_cells", 0, "Asserted across string columns"),
        ("exact_duplicate_rows", int(raw.duplicated().sum()), "Expected zero"),
        ("feature_identical_duplicates", int(raw.drop(columns="y").duplicated().sum()), "Expected zero"),
        ("constant_columns", int((raw.nunique() == 1).sum()), "Expected zero"),
        ("undocumented_category_levels", 0, "All observed levels match bank-names.txt"),
        ("unknown_job", int(raw["job"].eq("unknown").sum()), "Retained category"),
        ("unknown_education", int(raw["education"].eq("unknown").sum()), "Retained category"),
        ("unknown_contact", int(raw["contact"].eq("unknown").sum()), "Retained category"),
        ("unknown_poutcome", int(raw["poutcome"].eq("unknown").sum()), "Retained category"),
        ("rare_levels_below_1000", len(rare_levels), "; ".join(rare_levels)),
        ("previously_contacted_rows", int(contacted.sum()), "pdays >= 0"),
        ("never_contacted_rows", int((~contacted).sum()), "pdays = -1 and previous = 0"),
        ("pdays_previous_full_spearman", full_rho, "Structural reference"),
        ("pdays_previous_contacted_spearman", contacted_rho, "Within previously-contacted rows"),
        ("poutcome_unknown_previously_contacted_rows", 5, "Indices asserted under R11"),
        ("negative_balance_rows", int(raw["balance"].lt(0).sum()), "Retained"),
        ("minimum_balance", int(raw["balance"].min()), "Retained"),
        ("maximum_balance", int(raw["balance"].max()), "Retained"),
        ("maximum_campaign", int(raw["campaign"].max()), "Retained"),
        ("maximum_previous", int(raw["previous"].max()), "Retained"),
        ("maximum_pdays", int(raw["pdays"].max()), "Retained"),
        ("identifier_columns", 0, "No identifier present"),
        ("candidate_leakage_variables", 1, "duration excluded before modelling"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["audit_item", "observed_value", "verification_note"])
    audit.to_csv(output_root / "audit" / "data_audit.csv", index=False)

    pd.DataFrame(
        [
            {
                "feature": "duration",
                "display_name": DISPLAY_NAMES["duration"],
                "availability_issue": "Observed only after the marketing call ends",
                "action": "Excluded from every model and interpretability input; development-only Stage 5 association retained",
            }
        ]
    ).to_csv(output_root / "audit" / "leakage_register.csv", index=False)

    rulings = [
        ("R1", "y", "Target polarity", "Map yes to one", int(y.sum()) == 5289),
        ("R2", "All variables", "Missing values", "Assert zero; no imputation", int(raw.isna().sum().sum()) == 0),
        ("R3", "Documented unknown levels", "Disguised missingness", "Retain as explicit categories", True),
        ("R4", "job, education, contact, poutcome", "Unknown handling", "Retain explicit levels", [int(raw[c].eq("unknown").sum()) for c in ["job", "education", "contact", "poutcome"]] == [288, 1857, 13020, 36959]),
        ("R5", "All variables", "Duplicates", "Retain all rows; none observed", int(raw.duplicated().sum()) == 0 and int(raw.drop(columns="y").duplicated().sum()) == 0),
        ("R6", "All variables", "Identifier", "No identifier exclusion", True),
        ("R7", "duration", "Outcome-contemporaneous variable", "Exclude from modelling", "duration" not in X.columns),
        ("R8", "day", "Campaign calendar variable", "Retain and constrain interpretation", "day" in X.columns),
        ("R9", "pdays, previous", "Shared never-contacted state", "Decompose only inside logistic regression", int(contacted.sum()) == 8257),
        ("R10", "pdays, previous", "Sentinel and structural zero", "Use cleaned continuous terms and indicator for LR; raw for trees", int((~contacted).sum()) == 36954),
        ("R11", "poutcome", "Five inconsistent unknown rows", "Retain unmodified", raw.index[raw["poutcome"].eq("unknown") & raw["previous"].ge(1) & raw["pdays"].ge(0)].tolist() == INCONSISTENT_POUTCOME_INDICES),
        ("R12", "balance", "Negative values and skew", "Signed-log and standardise only for LR", int(raw["balance"].lt(0).sum()) == 3766),
        ("R13", "education", "Unvalidated ordering", "Treat as nominal", "education" in NOMINAL),
        ("R14", "month, day", "Campaign-period confounding", "Retain as campaign-timing indicators", all(c in X.columns for c in ["month", "day"])),
        ("R15", "Row position", "Date ordering", "Exclude position and shuffle split", True),
        ("R16", "All variables", "Extreme observations", "Remove no rows", len(X) == 45211),
        ("R17", "job, month, default", "Sparse levels", "Do not collapse; ignore unknown encoder inputs", True),
        ("R18", "Mixed feature types", "Correlation measure", "Long form Spearman and Cramer's V only", True),
    ]
    applied = pd.DataFrame(rulings, columns=["ruling", "display_name", "issue", "action", "verified"])
    if not applied["verified"].all():
        raise RuntimeError(f"Locked ruling verification failed: {applied.loc[~applied['verified'], 'ruling'].tolist()}")
    applied.to_csv(output_root / "audit" / "applied_rulings.csv", index=False)

    types = {feature: "Quantitative" for feature in QUANTITATIVE}
    types.update({feature: "Binary categorical" for feature in BINARY})
    types.update({feature: "Nominal categorical" for feature in NOMINAL})
    display = pd.DataFrame(
        {
            "feature": [*PREDICTORS, "duration", "subscribed"],
            "display_name": [*[DISPLAY_NAMES[f] for f in PREDICTORS], DISPLAY_NAMES["duration"], DISPLAY_NAMES["subscribed"]],
            "feature_type": [*[types[f] for f in PREDICTORS], "Excluded", "Target (binary)"],
        }
    )
    display.to_csv(output_root / "audit" / "feature_display_names.csv", index=False)
    return audit, target_position


def _cramers_v(a, b):
    table = pd.crosstab(a, b)
    chi2 = chi2_contingency(table, correction=False)[0]
    denominator = len(a) * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else np.nan


def development_eda(X_dev, y_dev, raw, y_full, output_root, plot_root):
    numeric_and_binary = [*QUANTITATIVE, *BINARY]
    correlation_rows = []
    for var_a, var_b in itertools.combinations(numeric_and_binary, 2):
        correlation_rows.append(
            {"var_a": var_a, "var_b": var_b, "method": "spearman", "value": float(spearmanr(X_dev[var_a], X_dev[var_b]).statistic)}
        )
    for var_a, var_b in itertools.combinations(NOMINAL, 2):
        correlation_rows.append(
            {"var_a": var_a, "var_b": var_b, "method": "cramers_v", "value": _cramers_v(X_dev[var_a], X_dev[var_b])}
        )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output_root / "eda" / "correlation_matrix.csv", index=False)
    if set(correlations["method"]) != {"spearman", "cramers_v"}:
        raise RuntimeError("Mixed-type correlation output is incomplete")

    quantitative_rates = {}
    for feature in QUANTITATIVE:
        bins = pd.qcut(X_dev[feature], q=5, duplicates="drop")
        rates = pd.DataFrame({"bin": bins, "positive": y_dev}).groupby("bin", observed=True)["positive"].mean()
        quantitative_rates[feature] = [float(value) for value in rates]
    development_indices, _ = train_test_split(
        np.arange(len(raw)), test_size=TEST_SIZE, stratify=y_full, random_state=SEED, shuffle=True
    )
    duration_dev = raw.iloc[development_indices]["duration"].reset_index(drop=True)
    if len(duration_dev) != len(y_dev):
        raise RuntimeError("Development duration alignment failed")
    duration_spearman = float(spearmanr(duration_dev, y_dev).statistic)
    duration_bins = pd.qcut(duration_dev, q=5, duplicates="drop")
    duration_rates = (
        pd.DataFrame({"bin": duration_bins, "positive": y_dev})
        .groupby("bin", observed=True)["positive"].mean().tolist()
    )

    max_spearman = correlations.loc[correlations["method"].eq("spearman")].iloc[
        correlations.loc[correlations["method"].eq("spearman"), "value"].abs().argmax()
    ]
    max_cramers = correlations.loc[correlations["method"].eq("cramers_v")].iloc[
        correlations.loc[correlations["method"].eq("cramers_v"), "value"].abs().argmax()
    ]
    pdays_previous = float(spearmanr(X_dev["pdays"], X_dev["previous"]).statistic)
    contacted = X_dev["pdays"].ge(0)
    contacted_rho = float(spearmanr(X_dev.loc[contacted, "pdays"], X_dev.loc[contacted, "previous"]).statistic)
    last_decile_rate = float(y_full.iloc[pd.qcut(np.arange(len(y_full)), 10, labels=False) == 9].mean())
    first_decile_rate = float(y_full.iloc[pd.qcut(np.arange(len(y_full)), 10, labels=False) == 0].mean())

    findings = f"""# Development-set EDA findings

All predictor–target statements below use development data only. Hypotheses are labelled and were recorded before model results were examined.

- The development set contains {int(y_dev.sum()):,} subscriptions among {len(y_dev):,} records ({y_dev.mean():.6f}).
- The long-form mixed-type association file contains {int((correlations['method'] == 'spearman').sum())} Spearman pairs and {int((correlations['method'] == 'cramers_v').sum())} Cramer's V pairs. Mixed quantitative–nominal pairs are not computed.
- The largest absolute development-set Spearman association among quantitative and binary predictors is {abs(float(max_spearman['value'])):.6f} for `{max_spearman['var_a']}` and `{max_spearman['var_b']}` (signed value {float(max_spearman['value']):.6f}).
- The largest development-set Cramer's V among nominal predictors is {float(max_cramers['value']):.6f} for `{max_cramers['var_a']}` and `{max_cramers['var_b']}`.
- The development-set `pdays`–`previous` Spearman association is {pdays_previous:.6f}; among previously contacted development rows it is {contacted_rho:.6f}. The locked LR decomposition is unchanged.
- Quantitative positive-rate sequences by development-set quintile are: {json.dumps({key: [round(value, 6) for value in values] for key, values in quantitative_rates.items()}, sort_keys=True)}.
- The source is date ordered. The first and last row-position deciles have subscription rates {first_decile_rate:.6f} and {last_decile_rate:.6f}. Row position is excluded and the split is shuffled and stratified.
- `month` and `day` are treated only as campaign-timing indicators; no seasonality variable is inferred.
- No observations were removed. The retained development values include negative balances and the `pdays = -1` sentinel.
- `duration` remains excluded from every model. Its development-only Spearman association with subscription is {duration_spearman:.6f}; its development quintile subscription-rate sequence is {json.dumps([round(float(value), 6) for value in duration_rates])}.
- **Pre-model hypothesis:** the shared never-contacted state may create a large level shift, while `pdays` and `previous` may retain distinct within-contacted information.
- **Pre-model hypothesis:** the tree family may represent category interactions and threshold patterns that the additive logistic-regression specification does not encode directly.
- **Pre-model hypothesis:** the date-ordered campaign structure may be reflected in `month` and `day`, so their evidence is not interpreted as portable seasonality.
"""
    _write_text(output_root / "eda" / "eda_findings.md", findings)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y_dev.value_counts().sort_index()
    ax.bar(["Did not subscribe", "Subscribed"], counts.values, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Development observations")
    ax.set_title("Development-set class balance")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    numeric_corr = X_dev[numeric_and_binary].corr(method="spearman")
    sns.heatmap(numeric_corr, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={"label": "Spearman"})
    axes[0].set_title("Quantitative and binary")
    nominal_matrix = pd.DataFrame(np.eye(len(NOMINAL)), index=NOMINAL, columns=NOMINAL)
    for _, row in correlations.loc[correlations["method"].eq("cramers_v")].iterrows():
        nominal_matrix.loc[row["var_a"], row["var_b"]] = row["value"]
        nominal_matrix.loc[row["var_b"], row["var_a"]] = row["value"]
    sns.heatmap(nominal_matrix, cmap="Blues", vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "Cramer's V"})
    axes[1].set_title("Nominal")
    fig.suptitle("Development-set predictor associations")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.ravel(), QUANTITATIVE):
        ax.plot(range(1, len(quantitative_rates[feature]) + 1), quantitative_rates[feature], marker="o")
        ax.set_title(DISPLAY_NAMES[feature])
        ax.set_xlabel("Development quintile")
        ax.set_ylabel("Subscription rate")
    fig.tight_layout()
    fig.savefig(plot_root / "eda" / "quantitative_by_outcome.png", dpi=180)
    plt.close(fig)
    return correlations, (str(max_spearman["var_a"]), str(max_spearman["var_b"])), float(max_spearman["value"])


def evaluation_and_features(output_root):
    text = f"""# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Term-deposit subscription.
- **Operating rule:** Floor precision at {PRECISION_FLOOR:.2f} and maximise recall; constrained ties use precision and then highest threshold. Non-attainment uses the locked maximum-precision fallback.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Sampling caveat:** PR-AUC, precision and calibration describe the pooled, date-ordered delivered sample and are not forward-campaign estimates.
- **Test status:** The test set is a one-time in-period held-out confirmation; development evidence governs comparisons.
"""
    _write_text(output_root / "audit" / "evaluation_framework.md", text)
    rows = []
    for model in MODEL_ORDER:
        if model == "lr":
            groups = {
                "Quantitative": "age, day and campaign standardised; balance signed-log then standardised; pdays_clean and previous_log1p standardised",
                "Derived binary": "previously_contacted indicator, unscaled",
                "Binary": "default, housing and loan mapped to explicit 0/1 indicators",
                "Nominal": "drop-reference one-hot with locked references and handle_unknown='ignore'",
            }
            count = 42
        else:
            groups = {
                "Quantitative": "age, balance, day, campaign, pdays and previous retained raw",
                "Binary": "default, housing and loan mapped to explicit 0/1 indicators",
                "Nominal": "full one-hot with no dropped level and handle_unknown='ignore'",
            }
            count = 47
        for group, treatment in groups.items():
            rows.append({"model": MODEL_DISPLAY[model], "feature_group": group, "treatment": treatment, "expected_encoded_terms": count})
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
    ax.set_ylabel("Observed subscription rate")
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
        selected = select_threshold(table, PRECISION_FLOOR)
        reference = metrics_at_threshold(y_dev, oof_scores[model_name], 0.50)
        selected_rows.append(
            {
                "model": model_name,
                "floored_metric": "precision",
                "floor_value": PRECISION_FLOOR,
                "optimised_metric": "recall",
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
        ax.axhline(PRECISION_FLOOR, color="grey", linestyle="--", label="Precision floor")
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
