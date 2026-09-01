# Performance Interpretability Trade-off Evaluation in Financial and Business Binary Classification Tasks

## Does model complexity actually buy anything?

**MSc Data Science Dissertation at Durham University** 

Machine learning models used in lending, collections, marketing and retention are increasingly complex, and complexity is usually assumed to cost transparency. This project tests whether the exchange is worth making, and finds no single answer: across the four datasets examined, complexity bought almost nothing in one case and a great deal in another.

Four model families spanning a range of flexibility: **logistic regression, a depth-constrained decision tree, random forest and XGBoost**, were applied to **four binary classification datasets** under a single pre-registered workflow, with performance, calibration, decision-threshold behaviour and four separate properties of interpretability measured identically for every model on every dataset.

---

## Headline findings

**1. The predictive gain from complexity ranged from negligible to very large.**

| Dataset | Rows | Positive Class Rate | Best Interpretable | Best Complex | Gain | Consistent across folds? |
|---|---:|---:|---:|---:|---:|---|
| South German Credit | 1,000 | 30.0% | 0.633 (logistic) | 0.640 (XGBoost) | **+0.007** | No — 3 of 5, reverses on holdout |
| Credit Card Default | 30,000 | 22.1% | 0.545 (logistic) | 0.560 (forest) | **+0.014** | Yes — 5 of 5 |
| Bank Marketing | 45,211 | 11.7% | 0.406 (logistic) | 0.452 (XGBoost) | **+0.046** | Yes — 5 of 5 |
| Iranian Churn | 3,150 | 15.7% | 0.746 (logistic) | 0.939 (forest) | **+0.193** | Yes — 5 of 5 |

*PR-AUC from nested cross-validation. A win-count would report "complex models won 4–0" — which would be true and almost entirely uninformative.*

**2. The complex models never found different variables.** On every dataset, all four models ranked the *same* predictor most important. On the dataset with the largest performance gap, they shared **nine or ten of their top ten variables**. The extra performance came from representing familiar information more flexibly, not from discovering new information.

**3. Where flexibility bought nothing, the flexible model chose to be simple.** On South German Credit, XGBoost selected a maximum depth of 2 and fitted an average of 3.7 leaves per tree, close to additive. On the other three datasets, the same algorithm fitted depth 5–6 and 17–20 leaves. The model declined the flexibility exactly where the data had no use for it.

**4. Interpretability is not one property.** Only *Structural Readability*, which can read the fitted model directly, ordered the four models consistently. On three of the four datasets, the most readable model failed on some other property: the decision tree produced the least trustworthy probabilities in the study on one dataset (calibration slope 0.44, falling to 0.20 on held-out data), the weakest internal explanation agreement on another, and probabilities that were not a well-defined quantity on a third.

**5. Two measurement artefacts would have changed the conclusions.** A hyperparameter search score exceeded its nested estimate by 0.021 on one model, enough to reverse that model's ranking against logistic regression. And hyperparameter tuning left the decision tree *worse* than its untuned baseline on three of four datasets.

---

## What was fixed before the results were seen

The project was built to be checkable rather than merely reported.

**Decisions were pre-registered.** Every dataset-specific choice: encodings, exclusions, transformations, decision thresholds, was written into a `decisions.md` file and locked *before* the pipeline ran. Nothing was chosen after seeing a result. Where a general rule needed revision, it was recorded as a numbered amendment applying to all four datasets rather than as a one-off exception.

**Selection was separated from evaluation.** Hyperparameters were chosen in an inner cross-validation loop and scored in an outer loop, so no model was credited with a score produced by the same data that selected it. Both quantities were recorded, which is how finding 5 above became visible.

**The held-out data was touched once.** A 20% partition was set aside before any exploratory analysis and evaluated exactly once, after preprocessing, hyperparameters and thresholds were all frozen.

**Comparisons are paired.** All four models share the same fold indices, so differences are computed within folds and reported with the number of folds agreeing in direction, not as a single aggregate that can hide instability.

**Runs are deterministic and audited.** Fixed seed throughout, single-threaded, with automated invariant checks verifying train/test separation, fold sharing, development-only calibration and threshold selection, and interpretability data coverage.

---

## Method in brief

| | |
|---|---|
| **Validation** | Nested cross-validation, 5 outer folds × 3 inner folds, stratified, shuffled, seed 42 |
| **Tuning** | Randomised search, 40 candidates per model, identical budget across families |
| **Primary metric** | PR-AUC (average precision) — chosen because the positive class is the minority in all four cases |
| **Decision threshold** | Constrained optimisation: one metric floored at a level declared before results, the other maximised |
| **Calibration** | Slope and intercept by logistic recalibration on out-of-fold predictions; assessed, not corrected |
| **Interpretability** | Coefficients, fitted tree structure, permutation importance (30 repeats/fold), SHAP — all four models, identical configuration |
| **Uncertainty** | Stratified percentile bootstrap, 5,000 resamples, shared indices across models |
| **Stack** | Python 3.9 · scikit-learn · XGBoost · SHAP · pandas · NumPy |

---

## What this repository contains

```
workflow/             the locked protocol applied identically to all four datasets
datasets/<name>/
  decisions.md        pre-registered rulings for that dataset
  data/               source file as obtained
  src/                pipeline code
  outputs/            every number the dissertation reports, as written by the pipeline
```

Start with `workflow/` for the protocol, or any `datasets/*/outputs/modelling/nested_cv.csv` for the headline results.

**To run one:**
```bash
pip install -r requirements.txt
cd datasets/south-german-credit && python -m src.run_pipeline
```

---

## Honest limitations

Four datasets are comparative cases, not a sample; nothing here establishes general properties of these algorithms. Their characteristics vary simultaneously, so no observed difference can be attributed to any single one of them. The comparison is between complete pipelines rather than estimators on identical inputs, since three datasets required restructuring the logistic regression design matrix for collinearity. The decision tree's deficit is partly a consequence of a deliberate depth cap imposed to keep it readable. Bank Marketing's file is date-ordered with a subscription rate rising from 3% to 47% across the collection period, and the shuffled split estimates within-period performance rather than forward performance.

---

## Data

All four datasets come from the UCI Machine Learning Repository under CC BY 4.0 and are redistributed here under that licence.

[South German Credit](https://doi.org/10.24432/C5QG88) · [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) · [Bank Marketing](https://doi.org/10.24432/C5K306) · [Iranian Churn](https://doi.org/10.24432/C5JW3Z)

Code is released under the MIT Licence.
