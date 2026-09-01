# South German Credit — Decisions File

This file serves as an input to `Per_Dataset_Analytical_Workflow.md`. No analytical run may alter these rulings retrospectively.

**Dataset:** South German Credit, DOI `10.24432/C5QG88`, CC BY 4.0.

**Documentation used:** the UCI dataset page and the accompanying `codetable.txt` for all variable types, level labels and coding.

**Final lock amendments:** nominal inputs are one-hot encoded for all three tree-family models; below-baseline PR-AUC is a warning and investigated result rather than a halt condition; binary mappings and logistic-regression reference levels are explicit; grant/refuse is the primary decision context; threshold ties follow the rule in Section 12; and no cost-ratio sensitivity analysis is performed.

---

## 1. Problem definition [Stage 1]

**Target.** `kredit` → `credit_risk`.

**Positive class: bad credit.**

`codetable.txt` codes `kredit` as `0: bad`, `1: good`. The modelling target is therefore constructed as:

```
y = 1 - kredit          # y = 1 → bad credit (positive), y = 0 → good credit (negative)
```

**Verification at Stage 3:** `y.sum() == 300`. If it returns 700, the inversion has been applied twice or not at all — halt.

**What one observation represents.** One credit contract granted by a bank in southern Germany between 1973 and 1975, with the outcome recorded after the fact.

**Decision supported.** Whether to grant or refuse a credit application. Pricing and monitoring are outside the primary decision context.

**Stakeholder.** The credit-granting institution, and the applicant as the party bearing the cost of a false positive.

**False positive.** A credit that would have been repaid, refused. Cost: forgone interest to the bank, denied access to the applicant.

**False negative.** A credit that was not complied with, granted. Cost: principal and recovery to the bank.

**Evaluation stance.** The models are evaluated as classification and ranking systems. Probability-level statements are made only where Stage 11 supports them, subject to Section 1.1.

### 1.1 Sampling caveat

The UCI page describes the data as a stratified sample from actual credits with bad credits heavily oversampled: 300 bad against 700 good, a 30.0% sample positive rate.

Binding consequences:

1. Predicted probabilities are calibrated to a 30% base rate. A model reported as well calibrated at Stage 11 is calibrated **to this sample**, not to a population.
2. PR-AUC and precision are base-rate dependent. Only within-dataset model ordering transfers to the cross-dataset stage.

Stage 11 and Stage 13 outputs carry this caveat.

---

## 2. Source file and assertions [Stage 2]

| Item | Value |
|---|---|
| Filename | `SouthGermanCredit.asc` |
| File size | 47,940 bytes |
| Delimiter | Whitespace, single header row, CRLF line endings |
| Expected rows | **1000** |
| Expected columns | **21** (20 predictors + target) |
| Expected positive count | **300** |
| Expected positive rate | **30.0%** |
| Missing values | **0** |

Assert rows, columns and positive count. Halt on mismatch.

**Reader note.** Use `pandas.read_csv(path, sep=r'\s+')`. Not `sep=' '`. All 21 columns parse as `int64`.

The `.asc` file is never modified. All rulings are applied inside the pipeline.

---

## 3. Rulings [Stage 3]

| # | Variable | Issue | Ruling | Verification |
|---|---|---|---|---|
| R1 | `kredit` | Target polarity | `y = 1 - kredit`; positive = bad | `y.sum() == 300` |
| R2 | — | Missing values | None present. Assert zero and halt otherwise | `df.isna().sum().sum() == 0` |
| R3 | — | Disguised missing codes | None. No `?`, blank or sentinel appears in any column | All observed codes matched to `codetable.txt` |
| R4 | — | Duplicate rows | 0 exact and 0 feature-identical duplicates. No de-duplication step | Both counts asserted as 0 |
| R5 | — | Identifier columns | None present. No column dropped as an identifier | Column list asserted equal to the 21 expected names |
| R6 | `verw` (purpose) | Code 7 (education) has **zero observations** | Documented but unobserved. Absent from every design matrix; **not** created as an all-zero dummy | Assert `7 not in df.verw.unique()` |
| R7 | `verw` (purpose) | Rare levels: 4 (n=12), 5 (n=22), 8 (n=9), 10 (n=12) | **No collapsing.** All one-hot encoders use `handle_unknown='ignore'` so a level absent from a training fold does not fail at prediction. LR coefficients for these levels are recorded but not interpreted individually at Stage 16 | Encoder configurations asserted; rare-level flag written to the audit |
| R8 | `bishkred` (number_credits) | Sparse top levels: 3 (n=28), 4 (n=6) | **Collapse levels 3 and 4** into a single top level. Applied for all four models | Resulting level counts: 633 / 333 / 34 |
| R9 | `pers`, `gastarb` | Coded opposite to the Statlog German Credit data | The South German file already carries the corrected coding. **No further recoding.** `gastarb`: 1 = yes, 2 = no. `pers`: 1 = 3 or more dependants, 2 = 0 to 2 | `gastarb == 1` count is 37, not 963. Halt if inverted |
| R10 | `hoehe` (amount) | Right-skewed (skew 1.95) | **`log1p` transform in the logistic regression pipeline only.** Tree family receives the raw value. Skew after transform is 0.130 | Feature-set table records the difference |
| R11 | — | Row order | The file is ordered on the target (Section 5.1). Row position is never a feature; the split is shuffled; no positional slicing anywhere in the pipeline | `shuffle=True` asserted in the split |
| R12 | All | Outliers | **None removed.** `amount` reaches 18,424 DM and `age` reaches 75; both are plausible values | No outlier filter in any pipeline |

### 3.1 Leakage register [Stage 2 output]

**No variables are registered.** All 20 predictors are applicant and application attributes available at the point of the credit decision. `credit_history` is included in that assessment and is treated as an ordinary predictor throughout.

`leakage_register.csv` is still produced with a header row and no entries, so the absence is a recorded result rather than a missing file.

### 3.2 Feature types and display names [Stage 3 output]

Raw German column codes are never shown in a table or figure.

| Column | Display name | Feature type | Treatment |
|---|---|---|---|
| `laufzeit` | Credit duration (months) | Quantitative | Numeric; standardised for logistic regression |
| `hoehe` | Credit amount (DM) | Quantitative | Numeric; `log1p` transformed and standardised for logistic regression |
| `alter` | Age (years) | Quantitative | Numeric; standardised for logistic regression |
| `laufkont` | Checking account status | Ordinal | Retained as an ordered score |
| `moral` | Credit history | Ordinal | Retained as an ordered score |
| `sparkont` | Savings | Ordinal | Retained as an ordered score |
| `beszeit` | Employment duration | Ordinal | Retained as an ordered score |
| `rate` | Instalment rate (% of income) | Ordinal | Retained as an ordered score |
| `wohnzeit` | Time at present residence | Ordinal | Retained as an ordered score |
| `verm` | Most valuable property | Ordinal | Retained as an ordered score |
| `bishkred` | Number of credits at this bank | Ordinal | Retained as an ordered score; sparse upper categories combined (R8) |
| `beruf` | Job quality | Ordinal | Retained as an ordered score |
| `pers` | People financially dependent | Binary categorical | `three_or_more_dependants = 1` when raw code is 1; otherwise 0 |
| `telef` | Telephone registered | Binary categorical | `telephone_registered = 1` when raw code is 2; otherwise 0 |
| `gastarb` | Foreign worker | Binary categorical | `foreign_worker = 1` when raw code is 1; otherwise 0 |
| `verw` | Purpose | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `famges` | Personal status and sex | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `buerge` | Other debtors or guarantor | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `weitkred` | Other instalment plans | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `wohn` | Housing | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `kredit` | Credit risk (bad = positive) | Target (binary) | Positive class = bad (R1) |

**Taxonomy: 3 quantitative, 9 ordinal, 3 binary, 5 nominal — 20 predictors.**

Level labels are taken verbatim from `codetable.txt`.

**Assignments that depart from the UCI page.** The UCI variable descriptions label `laufkont`, `moral` and `sparkont` as categorical. They are assigned as ordinal here. The assumption this makes is explicit: each of the three consists of an ordered scale plus one level denoting absence — no checking account, no credits taken, no savings account — and the file's coding places that absence level at the low end of the scale. Treating the code as an ordered score asserts that placement. This is an analytical judgement, recorded as such and logged in the deviations section, not a claim about what the documentation says.

---

## 4. Split and fold design [Stage 4]

| Item | Value |
|---|---|
| Split | Stratified 80/20 on `y` |
| Development set | 800 rows, 240 positive |
| Test set | 200 rows, 60 positive |
| Random seed | **42** (split, folds, search, bootstrap) |
| Outer folds | 5, stratified |
| Inner folds | 3, stratified |
| Shuffle | **True** — mandatory, see Section 5.1 |

**Small-sample note for the write-up.** The test set contains 60 positives, so a single confusion-matrix cell moves precision and recall visibly and the bootstrap interval on test PR-AUC will be wide. The nested cross-validation estimate over 800 development rows is the more stable quantity; the test evaluation is a confirmation, not the headline.

---

## 5. Audit expectations [Stages 2 and 5]

Computed from the source file before the run, so the pipeline's own output can be checked against an independent count.

### 5.1 Row order — the file is sorted on the target

**Bad credits are concentrated in the final quarter of the file.** Rows 750–999 are almost entirely bad; the first three quarters are predominantly good.

The workflow requires this check at Stage 2 and it is safety-critical here:

- The split **must** be shuffled and stratified. An unshuffled split would produce a test set of almost entirely bad credits.
- No operation anywhere may use row position — no `head`, no `iloc` slicing for subsets, no positional subsampling at Stage 16.
- Row order is an artefact of file construction, not information available at decision time.

Stage 5 reports this as an observed characteristic. It does not alter the split, which is already fixed.

### 5.2 Category codes

All observed codes match `codetable.txt`. **No undocumented codes in any variable.** The only anomaly is the empty level `verw == 7` (R6).

### 5.3 Correlated blocks [determines Stage 7 handling]

Maximum absolute Spearman correlation between any pair of predictors is **0.625** (`duration`–`amount`). Next: 0.506 (`number_credits`–`credit_history`), 0.397 (`telephone`–`job`).

**Ruling: no collapsing or pruning of correlated blocks for logistic regression.** Nothing approaches the level that destabilises coefficients. This contrasts with the Default of Credit Card Clients dataset, where a block at mean |r| ≈ 0.89 forced a collapse — a contrast that is itself material at the cross-dataset stage, since SGC coefficients can be read directly where CCD's could not.

### 5.4 Sparse levels

`gastarb` (foreign worker) has 37 observations in one level against 963 in the other — roughly 30 in the development set. A variable this unbalanced is exactly the configuration in which native tree importance and permutation importance can diverge; Stage 16 should be expected to surface it.

---

## 6. Evaluation framework [Stage 6]

| Purpose | Metric |
|---|---|
| Primary model selection | PR-AUC (average precision) |
| Discrimination | ROC-AUC |
| Probability quality | Calibration slope and intercept, Brier score, log loss |
| Positive-class behaviour | Precision and recall at the declared operating point |
| Error profile | Confusion matrix, FP and FN counts |

**Positive class:** bad credit. **Base rate:** 30.0%. A prevalence/dummy baseline is recorded alongside model PR-AUC. If a model's PR-AUC is at or below the matched baseline, the pipeline raises a prominent warning, runs target-polarity and prediction-score diagnostics, and continues. The result is investigated and documented rather than suppressed. The pipeline halts only for an invalid metric, failed polarity assertion or other data/prediction integrity failure.

**Domain description of errors**, used verbatim in written outputs:

- **False positive** — a credit that was repaid, refused.
- **False negative** — a credit that defaulted, granted.

**Class imbalance is handled through metric choice and operating-point selection only.** No class weighting, no resampling, no loss modification. At a 30% positive rate the imbalance is mild in any case.

---

## 7. Model-specific preprocessing [Stage 7]

The four models do not receive identical feature matrices. This table is authoritative; `feature_sets_by_model.csv` must reproduce it.

### Logistic regression

| Feature group | Treatment |
|---|---|
| `duration`, `age` | Numeric, standardised |
| `amount` | `log1p` transformed, then standardised (R10) |
| 9 ordinal variables | Ordered scores, standardised — one coefficient per variable |
| 3 binary variables | Explicit 0/1 indicators using the mappings in Section 3.2 |
| 5 nominal variables | One-hot, declared reference level dropped, `handle_unknown='ignore'` |
| Correlated blocks | No collapse (Section 5.3) |
| Regularisation | L2, strength `C` tuned in the inner loop |

**Parameter count: approximately 33** (3 quantitative + 9 ordinal + 3 binary + 18 nominal dummies).

**Locked nominal reference levels for logistic regression:** `purpose = 3` (furniture/equipment); `personal_status_sex = 3` (male: married/widowed); `other_debtors = 1` (none); `other_installment_plans = 3` (none); and `housing = 2` (rent). These are the most frequent observed levels and must be recorded in the coefficient-table metadata. Changing a reference level is a specification change, not a tuning choice.

Events per parameter: 7.3 across the development set, **5.8 in an outer-fold training set, and roughly 3.9 in an inner-fold training set** where the search actually runs. Tuned L2 is what makes the inner-fold figure workable, and individual coefficients — particularly on the rare `purpose` levels (R7) — should be read with that in mind at Stage 16.

**Ordinal variables enter as ordered scores.** One coefficient per variable keeps the coefficient table readable as a description of the credit decision, which is the reason logistic regression is in this study. The ordering asserted for `laufkont`, `moral` and `sparkont` is the analytical judgement set out in Section 3.2.

**This diverges from the workflow text**, which instructs "bin ordinal codes into interpretable buckets" for logistic regression. For this dataset the codes already are the buckets, and one-hot encoding all nine would raise the parameter count well beyond what the event count supports. Logged in the deviations section.

### Decision tree, random forest, XGBoost

| Feature group | Treatment |
|---|---|
| `duration`, `amount`, `age` | Raw values, no scaling, no transformation |
| 9 ordinal variables | Raw integer codes |
| 3 binary variables | Explicit 0/1 indicators using the mappings in Section 3.2 |
| 5 nominal variables | Full one-hot encoding with no dropped reference, `handle_unknown='ignore'` |

The tree-family design matrix contains 38 terms: 15 quantitative, ordinal or binary terms and 23 full nominal indicators. No split may operate directly on an arbitrary nominal category code. A nominal split is therefore read as category present versus absent. Feature-importance and explanation outputs are reduced from encoded-term level to the original 20-variable level using the workflow's declared aggregation rule.

---

## 9. Hyperparameter search [Stage 9]

Randomised search, `n_iter = 40`, seed 42, identical budget across all four models, scored on PR-AUC in the inner loop.

| Model | Search space |
|---|---|
| Logistic regression | `C`: log-uniform 1e-3 to 1e2; `penalty`: l2; `solver`: lbfgs; `max_iter`: 2000 |
| Decision tree | `max_depth`: 2–**5** (capped); `min_samples_leaf`: 5–50; `min_samples_split`: 10–100; `criterion`: gini, entropy; `ccp_alpha`: 0.0–0.02 |
| Random forest | `n_estimators`: 300–800; `max_depth`: 3–12 or None; `min_samples_leaf`: 1–20; `max_features`: sqrt, log2, 0.5 |
| XGBoost | `n_estimators`: 200–600; `max_depth`: 2–6; `learning_rate`: log-uniform 0.01–0.3; `subsample`: 0.6–1.0; `colsample_bytree`: 0.6–1.0; `min_child_weight`: 1–10; `reg_lambda`: log-uniform 0.1–10 |

**Decision tree maximum depth is capped at 5.** The search may not exceed it; realised depth is asserted `<= 5` in the invariant checks. The cap is a specification decision, not a tuning outcome: a tree deeper than five levels on 800 rows is no longer a structure a stakeholder can read, which would defeat its inclusion in the interpretability spectrum. Record that the ensembles were not similarly constrained.

**Both the inner-loop search score and the nested outer estimate are recorded for every model**, so any divergence between them is visible. `search_score` is defined in workflow Stage 9.

---

## 12. Threshold selection [Stage 12]

**Rule: floor recall at 0.75, maximise precision subject to that floor.**

Applied identically to all four models, on out-of-fold development predictions, fixed before the sweep is inspected.

**Deterministic tie-break:** among eligible thresholds, choose maximum precision; if tied, choose maximum recall; if still tied, choose the highest threshold.

**Why recall is floored.** The positive class is bad credit and the false negative is the costlier error, so the operating point guarantees coverage of bad credits and lets precision follow. Flooring recall is also mechanically safe: recall is monotone in the threshold, so every model can attain any recall floor and the Stage 12 halt condition will not fire spuriously.

**Why 0.75.** A declared analytical anchor, fixing coverage at three in four bad credits before precision is considered. It is **not derived from a verified economic cost** and must be described that way.

**Also recorded:**

- Precision, recall, FP and FN at the conventional `0.50` threshold, **as a reference convention only**.
- **No cost-ratio sensitivity analysis.** The previously mentioned 5:1 FN:FP convention is excluded from this protocol and must not appear in calculations, tables, figures or written outputs.

**One consequence to record rather than argue.** At a 30% base rate, a recall floor of 0.75 pushes all four models towards the region where precision converges on the base rate, so precision differences at the operating point may be compressed relative to the full PR curve. If the Stage 15 ranking under precision-at-threshold disagrees with the ranking under PR-AUC, check the threshold sweep files before offering any other reading.

---

## 16. Interpretability evidence [Stage 16]

The governing rules — development-set data source, signed-summation aggregation to variable level, and no variable-level aggregate for logistic regression coefficients — are set out in workflow Stage 16 and apply to every dataset. Two dataset-specific points follow from them here.

**Scale.** Logistic regression is fitted on approximately 33 encoded terms and the tree family on 38 encoded terms. Reduction to the original variable level therefore applies to the five nominal variables for every model family. `purpose`, contributing nine LR terms and ten tree-family terms, is the variable most exposed to a faulty reduction rule.

**Why the data source matters on this dataset.** The test set holds 200 rows and 60 positives — too few to support stable feature rankings. This is the smallest test set of the four datasets, so if the development-set rule is ever questioned, this is the dataset that demonstrates why it exists.

---

## Recorded limitations

1. **Oversampled positives.** 30% sample base rate from a stratified sample with bad credits heavily oversampled. Probabilities, precision and PR-AUC levels are conditional on the sampling design (Section 1.1).
2. **Age of the data.** 1973–1975. Economic conditions, credit products and the legal treatment of variables such as personal status and sex, and foreign worker, have all changed. Using these variables is a fact about a historical dataset, not an endorsement.
3. **Transformed amount.** The UCI page records the credit amount as the result of an unknown monotonic transformation. Rank-based and tree-based use is unaffected; the logistic regression coefficient on `log1p(amount)` cannot be read as an elasticity in DM.
4. **Small test set.** 60 positives (Section 4).
5. **Sex is not recoverable.** The UCI page records that `personal_status_sex` code 2 combines male singles with female non-singles, so sex cannot be derived from the variable. No sex variable is constructed.
6. **Thin events per parameter for logistic regression.** Roughly 3.9 in an inner-fold training set (Section 7).
7. **Ordinality assumed for three variables.** `laufkont`, `moral` and `sparkont` are treated as ordered scores against the UCI page's categorical label (Section 3.2). Each places a level denoting absence at the low end of an otherwise ordered scale. State the assumption in the methodology chapter rather than leaving it implicit in the coefficient table.

---

## Deviations to log

**From the earlier South German Credit work.** The earlier analysis used repeated stratified 5×5 cross-validation with a single hold-out as confirmation, and paired significance testing. This workflow uses a stratified 80/20 split with nested 5-outer / 3-inner cross-validation and **no significance testing** — comparison is by effect size, sign agreement across folds, and bootstrap interval.

Consequences:

1. Figures from the earlier run **are not comparable** to figures produced under this workflow and must not be quoted alongside them. Any earlier number reused is re-derived under the current protocol or dropped.
2. The words *significant*, *significantly*, *statistically significant* and *p-value* must not appear anywhere in the outputs in relation to differences between models or metrics.

**From the workflow text.** Stage 7 instructs "bin ordinal codes into interpretable buckets" for logistic regression. This file instead enters the nine ordinal variables as ordered scores (Section 7). Recorded as a deliberate, dataset-specific divergence.

**From the UCI variable descriptions.** `laufkont`, `moral` and `sparkont` are labelled categorical by UCI and assigned as ordinal here, on the reasoning in Section 3.2. Recorded as an analytical judgement.

---

## Verification log

Computed directly from `SouthGermanCredit.asc` before this file was written:

- Shape 1000 × 21; all columns `int64`; zero missing values
- `kredit`: 300 zeros, 700 ones, consistent with `codetable.txt` (`0: bad`, `1: good`)
- Zero exact duplicate rows; zero feature-identical duplicate rows
- All category codes matched against `codetable.txt`; no undocumented codes; `verw == 7` empty
- Concentration of the positive class at the end of the file (Section 5.1)
- Level counts for all categorical variables (Sections 3, 5.4)
- Spearman correlation matrix across all 20 predictors (Section 5.3)
- Skew of `amount` (1.950) and of `log1p(amount)` (0.130)
- SHA-256 of the source file
