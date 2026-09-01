# Bank Marketing — Decisions File

This file serves as an input to `Per_Dataset_Analytical_Workflow.md`. No analytical run may alter these rulings retrospectively.

**Dataset:** Bank Marketing, DOI `10.24432/C5K306`, CC BY 4.0.

**Documentation used:** the actual `bank-full.csv` header and `bank-names.txt` govern the schema, variable types, levels and coding. 

**File variant.** This file governs `bank-full.csv`, the 17-column version: 16 documented inputs plus the target. 

**Final lock amendments:** one of the sixteen documented inputs is excluded (`duration`; Section 3.1); `day` is retained because it is available at the locked scoring time and is not selected using full-data outcome evidence; nominal inputs are one-hot encoded for all three tree-family models; `education` is treated as nominal, not ordinal; the shared never-contacted state in `pdays` and `previous` is decomposed for logistic regression without discarding either variable; the correlation-matrix output carries a `method` column because the predictors are of mixed type (R18); the threshold rule floors precision rather than recall and uses the amended non-attainment procedure (Section 12); the decision-tree depth cap is retained at 5; and leaf-size and child-weight search ranges are scaled to the sample (Section 9).

---

## 1. Problem definition [Stage 1]

**Target.** `y` → `subscribed`.

**Positive class: subscription.**

`bank-names.txt` codes `y` as `"yes"` / `"no"`. The modelling target is constructed as:

```
y = (y == "yes").astype(int)      # y = 1 → subscribed (positive), y = 0 → did not subscribe
```

**Verification at Stage 3:** `y.sum() == 5289`. If it returns 39,922 the mapping has been inverted — halt.

**What one observation represents.** One recorded campaign-contact outcome for a client contacted by telephone during a direct marketing campaign run by a Portuguese banking institution between May 2008 and November 2010. The subscription outcome is recorded after the call. The file contains no client identifier, so uniqueness of clients across rows cannot be established.

**Decision supported.** Whether to place the recorded marketing call to a given client. Product design, pricing and channel strategy are outside the primary decision context.

**Scoring moment and feature availability.** Scoring occurs immediately before the recorded call. At that moment the client attributes, loan/default indicators, planned contact channel, calendar month and day, ordinal number of the planned call within the current campaign, and previous-campaign history are treated as available. `duration` is unavailable because it is only observed after the call ends. If implementation discovers that any retained field is not available at this scoring moment, the pipeline halts at Stage 3 rather than silently changing the leakage register.

**Stakeholder.** The bank's direct marketing function, operating with finite calling resources, and the client as the party receiving an unsolicited contact. No numerical call-capacity constraint is documented.

**False positive.** A client predicted to subscribe who does not. Cost: agent time consumed on an unproductive call, and an unwanted contact for the client.

**False negative.** A client predicted not to subscribe who would have. Cost: a forgone term deposit.

**Evaluation stance.** The models are evaluated as classification and ranking systems. Probability-level statements are made only where Stage 11 supports them, subject to Section 1.1.

### 1.1 Data collection caveat

The file records the outcome of a sequence of campaign waves rather than a random sample of the bank's client base, and `bank-names.txt` states that the rows are ordered by date across the collection period.

Binding consequences:

1. Rows are not exchangeable by construction, and the positive rate is not constant across the file (Section 5.1). The split is stratified and shuffled, so the development and hold-out sets are treated as exchangeable draws from a single pooled population. That treatment is an assumption, not a property of the data, and is recorded in the limitations.
2. This dataset contains no period or macro-economic covariate, so a design that separated the collection period could not distinguish period effects from predictor effects. That is the reason the stratified split is retained.
3. Predicted probabilities are calibrated to the 11.70% pooled base rate. A model reported as well calibrated at Stage 11 is calibrated **to this pooled sample**, not to any future campaign.
4. PR-AUC and precision are base-rate dependent. Only within-dataset model ordering transfers to the cross-dataset stage.

Stage 11 and Stage 13 outputs carry this caveat.

---

## 2. Source file and assertions [Stage 2]

| Item | Value |
|---|---|
| Filename | `bank-full.csv` |
| Delimiter | Semicolon, single header row, quoted string fields, LF line endings |
| Expected rows | **45,211** |
| Expected columns | **17** (16 inputs + target) |
| Expected positive count | **5,289** |
| Expected positive rate | **11.70%** |
| Missing values | **0** |

Assert rows, columns and positive count. Halt on mismatch.

**Reader note.** Use `pandas.read_csv(path, sep=';')`. Not the default comma. Quoted fields are handled by the default parser; do not set `quoting`. Seven columns parse as `int64`, ten as string.

The `.csv` file is never modified. All rulings are applied inside the pipeline.

---

## 3. Rulings [Stage 3]

| # | Variable | Issue | Ruling | Verification |
|---|---|---|---|---|
| R1 | `y` | Target polarity | `y = 1` where `y == "yes"`; positive = subscription | `y.sum() == 5289` |
| R2 | — | Missing values | None present. Assert zero and halt otherwise | `df.isna().sum().sum() == 0` |
| R3 | — | Disguised missing codes | `"unknown"` appears in `job`, `education`, `contact` and `poutcome`. It is **documented as a category**, not missingness, and is retained as an explicit level throughout (R4) | Observed level sets matched to `bank-names.txt` |
| R4 | `job`, `education`, `contact`, `poutcome` | `"unknown"` handling | **Retained as an explicit level.** No imputation, no dropping, no separate missingness indicator. For `poutcome`, 36,954 rows combine `unknown` with no prior contact and five rows combine it with recorded prior contacts; it is therefore described as an unknown or unavailable previous-campaign outcome, not universally as proof of no prior contact | Level counts asserted: 288 / 1,857 / 13,020 / 36,959; the five exceptions in R11 remain unaltered |
| R5 | — | Duplicate rows | 0 exact and 0 feature-identical duplicates. No de-duplication step | Both counts asserted as 0 |
| R6 | — | Identifier columns | None present. No column dropped as an identifier | Column list asserted equal to the 17 expected names |
| R7 | `duration` | Outcome-contemporaneous variable | **Excluded from all modelling** (Section 3.1). Retained for development-only Stage 5 reporting | Asserted absent from every design matrix |
| R8 | `day` | Current-campaign calendar variable | **Retained in all models.** It is available at the locked scoring moment. Its earlier proposed exclusion relied partly on a full-file target association and is void; lack of a monotone marginal association is not a defensible feature-selection rule for models that can use nonlinear structure. Like `month`, it is interpreted only as a campaign-timing indicator | Asserted present in every design matrix; no full-file target statistic determines its treatment |
| R9 | `pdays`, `previous` | Their shared never-contacted state produces full-file Spearman ρ = 0.986, but among the 8,257 previously contacted rows their Spearman ρ is −0.101; they are not redundant within that group | **Logistic regression only:** retain information from both variables by using `previously_contacted`, `pdays_clean` and `previous_log1p = log1p(previous)`. Tree family receives both raw columns unchanged (Section 7) | The coincidence and contacted-subset correlation are asserted; all three LR terms and both tree inputs are recorded in the feature-set table |
| R10 | `pdays`, `previous` | Sentinel/structural zeros for never contacted | **Tree family:** retain both raw variables, including `pdays = -1`. **LR:** `pdays_clean = pdays where pdays >= 0, else 0`; `previous_log1p = log1p(previous)`; `previously_contacted = 1` where `pdays >= 0`. The indicator represents the level shift, while the two continuous terms retain within-contacted information | Both cleaned terms have minimum 0; `previously_contacted.sum() == 8257`; no original information column is dropped before transformation |
| R11 | `poutcome` | 5 rows carry `"unknown"` with `previous >= 1` and `pdays >= 0`, contradicting the pattern in the other 45,206 rows | **Retained unmodified.** Recoding to `failure` or `other` would be an unsourced inference | 0-based file indices 40658, 41821, 42042, 43978, 45021 asserted present and unaltered |
| R12 | `balance` | Negative in 3,766 rows (8.33%), minimum −8,019; right skew 8.36 | Negative values are legitimate overdrawn balances and are **retained**. `log1p` is therefore unavailable. **LR only: `sign(x) · log1p(|x|)`, then standardised.** Tree family receives the raw value | Skew after transform −1.583; feature-set table records the difference |
| R13 | `education` | The labels have a broad educational hierarchy, but `unknown` has no defensible position and the documentation supplies categories rather than a validated ordered scale | **Treated as nominal, not ordinal.** No ordered score is constructed and no full-file target rate is used to choose the encoding | Encoder configuration asserted; divergence from SGC logged |
| R14 | `month`, `day` | Retained predictors confounded with campaign period (Section 5.1) | Retained. **Interpretive constraint:** coefficients, importances and SHAP values for both are read as campaign-timing indicators, never as seasonality or portable calendar effects | Constraint carried in Stage 16 output metadata |
| R15 | — | Row order | The file is ordered by date (Section 5.1). Row position is never a feature; the split is shuffled; no positional slicing anywhere in the pipeline | `shuffle=True` asserted in the split |
| R16 | All | Outliers | **None removed.** `campaign` reaches 63, `previous` reaches 275, `balance` reaches 102,127 and `pdays` reaches 871; all are plausible values | No outlier filter in any pipeline |
| R17 | `job`, `month`, `default` | Sparse levels (Section 5.4) | **No collapsing.** All one-hot encoders use `handle_unknown='ignore'` so a level absent from a training fold does not fail at prediction | Encoder configurations asserted; rare-level flag written to the audit |
| R18 | — | Mixed variable types in `outputs/eda/correlation_matrix.csv` | Six predictors are nominal strings, so a single coefficient does not cover the matrix. **Write one long-form file with a `method` column**: `spearman` for every pair drawn from the 6 quantitative and 3 binary predictors, `cramers_v` for every nominal–nominal pair. Mixed quantitative–nominal pairs are **not** computed and their absence is recorded in `eda_findings.md`. Schema stays `var_a`, `var_b`, `method`, `value` | File contains both method values; no pair appears under two methods |

### 3.1 Leakage register [Stage 2 output]

**One variable is registered.**

| Variable | Basis | Ruling |
|---|---|---|
| `duration` | The UCI variable table states that the value is known only once the call has ended, at which point the outcome is already determined. This documented availability rule is sufficient; no full-file target association is used to decide exclusion | Excluded from every design matrix, hyperparameter search and interpretability output. Its development-set association with the target is reported once at Stage 5 as an EDA finding |

`day` is **not** a leakage entry. It is known at the locked scoring moment and is retained under R8. Its empirical association is estimated on development data only and cannot alter that ruling.

`leakage_register.csv` therefore carries one entry. 

### 3.2 Feature types and display names [Stage 3 output]

| Column | Display name | Feature type | Treatment |
|---|---|---|---|
| `age` | Age (years) | Quantitative | Numeric; standardised for logistic regression |
| `balance` | Average yearly balance (EUR) | Quantitative | Numeric; signed-log transformed and standardised for logistic regression (R12) |
| `day` | Day of month of planned contact | Quantitative | Numeric; standardised for logistic regression; interpreted only as campaign timing (R8, R14) |
| `campaign` | Contacts this campaign | Quantitative | Numeric; standardised for logistic regression |
| `pdays` | Days since previous contact | Quantitative | LR: `pdays_clean`, standardised (R10); tree family: raw with `-1` sentinel |
| `previous` | Contacts before this campaign | Quantitative | LR: `previous_log1p = log1p(previous)`, standardised; tree family: raw (R9–R10) |
| — | Previously contacted | Derived binary | LR only: `1` when `pdays >= 0`, else `0` (R9) |
| `default` | Credit in default | Binary categorical | `default = 1` when raw value is `"yes"`; otherwise 0 |
| `housing` | Housing loan | Binary categorical | `housing = 1` when raw value is `"yes"`; otherwise 0 |
| `loan` | Personal loan | Binary categorical | `loan = 1` when raw value is `"yes"`; otherwise 0 |
| `job` | Occupation | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `marital` | Marital status | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `education` | Education level | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot (R13) |
| `contact` | Contact channel | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `month` | Month of planned contact | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot (R14) |
| `poutcome` | Previous campaign outcome | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `duration` | Last contact duration (s) | **Excluded** | Leakage register (Section 3.1) |
| `y` | Term deposit subscribed (positive) | Target (binary) | Positive class = subscription (R1) |

**Taxonomy: 6 quantitative, 0 ordinal, 3 binary, 6 nominal — 15 predictors** from 16 documented inputs, after excluding `duration`.

Level labels are taken verbatim from `bank-names.txt`.

---

## 4. Split and fold design [Stage 4]

| Item | Value |
|---|---|
| Split | Stratified 80/20 on `y` |
| Development set | 36,168 rows, 4,231 positive (11.698%) |
| Test set | 9,043 rows, 1,058 positive (11.700%) |
| Random seed | **42** for every stochastic component: split, folds, model estimators, randomised search, bootstrap, permutation importance, SHAP row selection and SHAP background selection |
| Outer folds | 5, stratified |
| Inner folds | 3, stratified |
| Shuffle | **True** — mandatory, see Section 5.1 |

---

## 5. Audit expectations [Stages 2 and 5]

Full-file **structural** assertions in this section may be checked before the split: dimensions, codes, sentinels, duplicates, predictor-only relationships and the workflow-mandated row-position audit. Predictor–target relationships other than the target distribution and mandated position audit are not locked here and do not determine preprocessing. Stage 5 computes every predictor–target EDA quantity on development data only. Development-only outputs are not required to reproduce any historical full-file outcome statistic.

### 5.1 Row order — the file is ordered by date

`bank-names.txt` records that the rows are ordered by date across the collection period. Workflow Stage 2 requires target rate by row-position decile to be computed and recorded whichever way it falls, and Stage 5 requires any variation found to be reported as an observed characteristic. The positive rate is **not constant across the file**:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Positive rate (%) | 2.99 | 3.74 | 4.82 | 6.41 | 6.10 | 6.17 | 10.46 | 13.12 | 16.08 | 47.09 |

These are the expected contents of `outputs/audit/target_rate_by_position.csv`; the pipeline's own values are checked against them.

The workflow requires this check at Stage 2 and it is binding here:

- The split **must** be shuffled and stratified. An unshuffled split would produce a hold-out drawn entirely from the final campaign wave.
- No operation anywhere may use row position — no `head`, no `iloc` slicing for subsets, no positional subsampling at Stage 16.
- Row order is an artefact of file construction, not information available at decision time.
- No year, period or time-index variable is derived from row order or from `month` for any purpose, modelling or reporting. The source file contains no such column and none is inferred.

Stage 5 reports the ordering and the first-to-last decile movement in prose — two sentences, no figure. The decile table is produced as a required Stage 2 output and lives in the audit directory; it is not promoted into the dataset chapter. Stage 5 does not alter the split, which is already fixed.

**Consequence for `month` and `day`.** Because calendar values are unevenly distributed across the collection period, their development-set associations may partly reflect when in the campaign sequence the calls occurred. This is the basis of the interpretive constraint in R14.

### 5.2 Category codes

All observed levels match `bank-names.txt`. **No undocumented levels in any variable.** The only structural anomaly is the five-row `poutcome` inconsistency (R11).

### 5.3 Correlated blocks [verifies the locked Stage 7 handling]

**Full-file structural reference.** Maximum absolute Spearman correlation is **0.986** (`pdays`–`previous`), driven by the shared never-contacted state. Among the 8,257 previously contacted rows, their Spearman correlation is **−0.101**, so the two measurements are not treated as redundant. Excluding the full-file pair, the largest absolute quantitative correlation is **0.112** (`campaign`–`pdays`).

**Nominal predictors.** Maximum Cramér's V is **0.512** (`contact`–`month`), then 0.504 (`month`–`housing`) and 0.458 (`job`–`education`).

**Ruling: the shared sentinel state is represented explicitly for logistic regression by R9–R10, while information from both measurements is retained; no collapsing or pruning is performed.** Stage 5 recomputes the correlation matrix on development data. The values above are full-file structural references and are not expected values for `outputs/eda/correlation_matrix.csv`.

At the cross-dataset stage this is described as a sentinel-driven representation issue, not as evidence that two distinct prior-contact measurements were interchangeable.

### 5.4 Sparse levels

| Variable | Sparse level | n | Share |
|---|---|---|---|
| `job` | `unknown` | 288 | 0.64% |
| `month` | `dec` | 214 | 0.47% |
| `month` | `mar` | 477 | 1.06% |
| `month` | `sep` | 579 | 1.28% |
| `month` | `oct` | 738 | 1.63% |
| `default` | `yes` | 815 | 1.80% |

No level is expected to fall below roughly 170 observations in an inner-fold training set, so all are retained. `default = yes` remains structurally sparse at roughly 1:54. Its target association and any divergence between importance measures are measured on development evidence and reported whichever way they fall; neither direction is predeclared here.

---

## 6. Evaluation framework [Stage 6]

| Purpose | Metric |
|---|---|
| Primary model selection | PR-AUC (average precision) |
| Discrimination | ROC-AUC |
| Probability quality | Calibration slope and intercept, Brier score, log loss |
| Positive-class behaviour | Precision and recall at the declared operating point |
| Error profile | Confusion matrix, FP and FN counts |

**Positive class:** subscription. **Base rate:** 11.70%. A prevalence/dummy baseline is recorded alongside model PR-AUC. If a model's PR-AUC is at or below the matched baseline, the pipeline raises a prominent warning, runs target-polarity and prediction-score diagnostics, and continues. The result is investigated and documented rather than suppressed. The pipeline halts only for an invalid metric, failed polarity assertion or other data/prediction integrity failure.

**Domain description of errors**, used verbatim in written outputs:

- **False positive** — a client called who did not subscribe.
- **False negative** — a client not called who would have subscribed.

**Class imbalance is handled through metric choice and operating-point selection only.** No class weighting, no resampling, no loss modification. Workflow Stage 6 permits no modification to the models, the loss or the training data on account of imbalance, and no exception is taken here. At an 11.70% positive rate, the imbalance is a moderate problem.

---

## 7. Model-specific preprocessing [Stage 7]

The four models do not receive identical feature matrices. This table is authoritative; `feature_sets_by_model.csv` must reproduce it.

### Logistic regression

| Feature group | Treatment |
|---|---|
| `age`, `day`, `campaign` | Numeric, standardised |
| `balance` | Signed-log transformed, then standardised (R12) |
| `pdays` | Replaced by `pdays_clean`, standardised (R10) |
| `previous` | Replaced by `previous_log1p = log1p(previous)`, standardised (R9–R10) |
| `previously_contacted` | Derived 0/1 indicator, unscaled |
| 3 binary variables | Explicit 0/1 indicators using the mappings in Section 3.2 |
| 6 nominal variables | One-hot, declared reference level dropped, `handle_unknown='ignore'` |
| Shared sentinel block | Decomposed into contact-status and within-contacted measurements (R9–R10); no predictor information discarded |
| Regularisation | L2, strength `C` tuned in the inner loop |

**Parameter count: 42** (6 quantitative + 1 derived binary + 3 binary + 32 nominal dummies), plus intercept.

**Locked nominal reference levels for logistic regression:** `job = blue-collar` (n = 9,732); `marital = married` (n = 27,214); `education = secondary` (n = 23,202); `contact = cellular` (n = 29,285); `month = may` (n = 13,766); `poutcome = unknown` (n = 36,959). These are the most frequent observed levels and must be recorded in the coefficient-table metadata. Changing a reference level is a specification change, not a tuning choice.

Events per parameter are approximately 100.7 across the development set, **80.6 in an outer-fold training set and 53.7 in an inner-fold training set** where the search runs. These counts provide more information per parameter than South German Credit, but coefficient stability remains an empirical result assessed from the fitted model rather than guaranteed in advance.

### Decision tree, random forest, XGBoost

| Feature group | Treatment |
|---|---|
| `age`, `balance`, `day`, `campaign`, `pdays`, `previous` | Raw values, no scaling, no transformation; `pdays` retains its `-1` sentinel |
| 3 binary variables | Explicit 0/1 indicators using the mappings in Section 3.2 |
| 6 nominal variables | Full one-hot encoding with no dropped reference, `handle_unknown='ignore'` |

The tree-family design matrix contains **47 terms**: 9 quantitative and binary terms and 38 full nominal indicators. No split may operate directly on an arbitrary nominal category code. A nominal split is therefore read as category present versus absent. Feature-importance and explanation outputs are reduced from encoded-term level to the original 15-variable level using the workflow's declared aggregation rule.

---

## 9. Hyperparameter search [Stage 9]

Randomised search, `n_iter = 40`, seed 42, identical budget across all four models, scored on PR-AUC in the inner loop.

| Model | Search space |
|---|---|
| Logistic regression | `C`: log-uniform 1e-3 to 1e2; `penalty`: l2; `solver`: lbfgs; `max_iter`: 2000 |
| Decision tree | `max_depth`: 2–**5** (capped, see below); `min_samples_leaf`: 50–500; `min_samples_split`: 100–1000; `criterion`: gini, entropy; `ccp_alpha`: 0.0–0.02 |
| Random forest | `n_estimators`: 300–800; `max_depth`: 3–12 or None; `min_samples_leaf`: 5–100; `max_features`: sqrt, log2, 0.5 |
| XGBoost | `n_estimators`: 200–600; `max_depth`: 2–6; `learning_rate`: log-uniform 0.01–0.3; `subsample`: 0.6–1.0; `colsample_bytree`: 0.6–1.0; `min_child_weight`: 1–50; `reg_lambda`: log-uniform 0.1–10 |

**Leaf-size and child-weight ranges are scaled to the sample.** The development set is 45× larger than South German Credit's, so SGC's `min_samples_leaf` of 5–50 would permit leaves holding under 0.02% of the data. The ranges above hold the equivalent leaf fraction roughly constant. `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree` and `C` are carried across unchanged so the tuning budget stays comparable. Locked at the scaled ranges.

**Decision tree maximum depth is capped at 5**, retained from South German Credit so the readability standard is uniform across datasets. The cap is a specification decision, not a tuning outcome: a tree deeper than five levels is no longer a structure a stakeholder can read, which would defeat its inclusion in the interpretability spectrum. Record that the ensembles were not similarly constrained.

**Stage 8 baseline.** Per workflow Stage 8, dataset-level constraints that define a model remain active at the untuned baseline. The decision-tree baseline therefore uses `max_depth = 5` with all other estimator hyperparameters untuned.

**Both the inner-loop search score and the nested outer estimate are recorded for every model**, so any divergence between them is visible. `search_score` is defined in workflow Stage 9.

---

## 12. Threshold selection [Stage 12]

**Rule: floor precision at 0.50, maximise recall subject to that floor.**

Applied identically to all four models, on out-of-fold development predictions, fixed before the sweep is inspected.

**Candidate grid and prediction rule.** For each model, evaluate every distinct finite OOF score as a threshold using `predicted_positive = (score >= threshold)`. Every candidate therefore predicts at least one positive case. A point above the maximum score, or any synthetic precision-recall endpoint with zero predicted positives, is excluded from attainment and fallback selection. The conventional probability threshold `0.50` is calculated separately as a reference and uses precision `0` if it predicts no positive cases.

**Deterministic constrained tie-break:** among thresholds with precision at least 0.50, choose maximum recall; if tied, choose maximum precision; if still tied, choose the highest threshold.

**Why precision is floored.** Calling resources are finite and a false positive consumes agent time. The rule is a **minimum hit-rate constraint**, not a numerical call-capacity model: no source supplies a fixed number of calls. Where the floor is attained, at least half of the model-selected calls in OOF development predictions reach a client who subscribes; recall is then maximised subject to that requirement.

**Why 0.50.** A declared analytical anchor, fixing a one-in-two hit rate against a 11.70% base rate — a lift of roughly 4.3× over undirected calling. It is **not derived from a verified economic cost** and must be described that way.

Workflow Stage 12 as originally written required a halt in that case, which would have fired on a legitimate result rather than a defect. **Workflow amendment seven** resolves this generally: where the floored metric is not monotone in the threshold, non-attainment is recorded as a result rather than a halt. The behaviour for this dataset follows from that amendment:

> If no non-empty development-set threshold achieves precision ≥ 0.50 for a model, set that model's operating point at the threshold maximising precision; among equal-precision thresholds maximise recall; if still tied choose the highest threshold. Record `floor_attained = FALSE` and `selection_mode = "max_precision_fallback"` in `thresholds_selected.csv`, and continue. **The floor is not relaxed and is not re-tuned per model.**

Non-attainment for any model is itself a reportable finding: it states that the model cannot reach a one-in-two hit rate at any operating point, which is a substantive fact about the marketing problem rather than a pipeline failure. Where it occurs, the four models are no longer at a common operating point and Stage 15's threshold-metric rankings say so.

**Also recorded:**

- Precision, recall, FP and FN at the conventional `0.50` probability threshold, **as a reference convention only**. Note that the probability threshold 0.50 and the precision floor 0.50 are different quantities that happen to share a number; the output labels must distinguish them.
- The selected-row metadata: `floored_metric = precision`, `floor_value = 0.50`, `optimised_metric = recall`, `selection_mode`, `predicted_positive`, `predicted_positive_at_050`, and `selection_input_rows = 36,168`.
- **No cost-ratio sensitivity analysis.** No FN:FP cost convention exists in the source documentation for this dataset and none is invented.

---

## 16. Interpretability evidence [Stage 16]

The governing rules — development-set data source, signed-summation aggregation to variable level, and no variable-level aggregate for logistic regression coefficients — are set out in workflow Stage 16 and apply to every dataset. Three dataset-specific points follow from them here.

**Scale.** Logistic regression is fitted on 42 encoded terms and the tree family on 47. Reduction to the original variable level therefore applies to the six nominal variables for every model family. `job` and `month`, contributing 11 LR terms and 12 tree-family terms each, are the variables most exposed to a faulty reduction rule.

**Why the data source matters on this dataset.** The hold-out contains 9,043 rows and 1,058 positives, but its size does not authorise its use for feature ranking. The development-set rule protects the confirmatory role of the hold-out and is a protocol commitment rather than a sample-size workaround.

**SHAP coverage differs sharply from SGC, and the difference is a consequence of the fixed cap.** Workflow Stage 16 caps SHAP explanation rows at 2,000 by stratified subsampling and fixes the background sample at `min(200, n_development)`. SGC's development set holds 800 rows, so it falls under the cap and every development row is explained. Bank Marketing's holds 36,168, so the cap binds and SHAP describes roughly **5.5%** of development rows against SGC's 100%, with a 200-row background in both cases. The cap is not altered — it is what keeps the configuration identical across datasets — but the coverage difference is recorded in the Stage 16 metadata and must be stated wherever SHAP evidence from the two datasets is placed side by side.

---

## Recorded limitations

1. **Reduced feature set.** One of the sixteen documented inputs, `duration`, is excluded on decision-time leakage grounds (Section 3.1). Reported performance is therefore not comparable to published results that retain it.
2. **Non-exchangeable rows treated as exchangeable.** The file is ordered by date and the positive rate is not constant across it (Section 5.1). The stratified split pools the collection period, so hold-out performance is an in-period estimate rather than a forward-looking one. A period-separated design was not adopted because this file contains no period or macro-economic covariate, so period effects could not be separated from predictor effects.
3. **Calendar variables are not seasonality.** `month` and `day` are unevenly distributed across the collection period, so their model evidence may partly reflect campaign sequence. All interpretive text observes R14.
4. **`education` ordinality not asserted.** Treated as nominal (R13), so this dataset contributes no ordered-score coefficients to the cross-dataset comparison with SGC.
5. **`unknown` has limited semantic resolution.** It is retained as a documented category in four variables and is the modal level of `poutcome` at 81.75%. For `poutcome`, it usually accompanies no prior contact but also occurs in five previously contacted records, so it is interpreted as an unknown or unavailable prior outcome rather than assigned one universal meaning.
6. **`default` is structurally sparse.** The `yes` level contains 815 rows. Its target association and model evidence are reported from development outputs without an expected direction.
7. **Campaign-outcome data only.** Every row is a client the bank chose to call. Nothing in the file describes clients who were never contacted, so no statement about the untargeted population is supportable.
8. **Client grouping cannot be verified.** The source description permits repeated contacts, but the file contains no client identifier. Client uniqueness and dependence across rows cannot be checked, and a client-grouped split is impossible.
9. **Aggregate full-file outcomes were inspected before the final lock.** An earlier draft calculated several full-file predictor–target summaries. The final feature rulings do not use those summaries: `day` is retained, `duration` is excluded from external availability documentation, and `education` is nominal on semantic grounds. All reported predictor–target EDA is recomputed on development data. The aggregate pre-lock exposure is nevertheless disclosed as a limitation on complete analyst blinding.

---

## Deviations to log

**From the workflow text.** None taken. Two points where a deviation might have been expected:

1. Workflow Stage 7 instructs "bin ordinal codes into interpretable buckets" for logistic regression. SGC logged a deliberate divergence from this. **No divergence arises here**, because this dataset has no ordinal variables (R13, Section 3.2). The instruction is vacuous rather than overridden.
2. Workflow Stage 12's attainment halt was incompatible with a precision floor. Rather than overriding it in this file, **workflow amendment seven** was made, generalising the rule to any non-monotone floored metric (Section 12). This file therefore takes no deviation; it depends on the amended workflow and must not be run against an unamended copy.

**Amendment seven schema consequence.** `thresholds_selected.csv` gains a `floor_attained` column for every dataset. The completed South German Credit run predates the amendment and its file lacks the column; SGC floored recall, so attainment held by construction and the value would have been `TRUE` for all four models. Recorded here rather than corrected by re-running SGC, which Stage 13's prohibition forbids without written authorisation.

The nominal one-hot encoding for the tree family follows the SGC final-lock amendment and is applied here for uniformity.

**From the documented feature set.** One input is excluded: `duration` (R7, leakage register), using documented decision-time availability as the basis. `day` is retained under R8. Development-only EDA may describe either variable's observed association but cannot revise these rulings.

**From the South German Credit decisions file.**

1. No variable is treated as ordinal here; `education` is nominal (R13). SGC entered nine ordinal variables as ordered scores.
2. The threshold rule floors precision rather than recall, and therefore carries an attainability fallback that SGC does not need (Section 12).
3. Inner-fold events per parameter are roughly 14× higher; whether that produces more stable coefficients is assessed empirically rather than assumed (Section 7).
4. The leakage register is non-empty (Section 3.1).

Each is a dataset-driven difference rather than a protocol change, and each is material at the cross-dataset stage.

**On terminology.** The words *significant*, *significantly*, *statistically significant* and *p-value* must not appear anywhere in the outputs in relation to differences between models or metrics. Comparison is by effect size, sign agreement across folds, and bootstrap interval.

---

## Verification log

Computed directly from `bank-full.csv` before this file was written. Structural assertions and the workflow-mandated row-position audit may be checked against the full file. Historical predictor–target scores are disclosed separately and are not pipeline expectations or feature-selection evidence.

- Shape 45,211 × 17; 7 `int64` columns, 10 string columns; zero missing values
- `y`: 5,289 `"yes"`, 39,922 `"no"`; positive rate 11.6985%
- Zero exact duplicate rows; zero feature-identical duplicate rows
- All observed levels matched against `bank-names.txt`; no undocumented levels
- `pdays == -1` and `previous == 0` coincide exactly, 36,954 rows; Spearman ρ = 0.986
- Within the 8,257 previously contacted rows, `pdays`–`previous` Spearman ρ = −0.101
- 5 rows with `poutcome == "unknown"` and `previous >= 1`, indices as listed in R11
- Positive rate by row-position decile: 2.99, 3.74, 4.82, 6.41, 6.10, 6.17, 10.46, 13.12, 16.08, 47.09 (%)
- Spearman correlation matrix across quantitative predictors; Cramér's V matrix across nominal predictors (Section 5.3)
- Skew of `balance` (8.360) and of `sign(x)·log1p(|x|)` (−1.583)
- Level counts for all categorical variables (Sections 3, 5.4)
- Stratified 80/20 split at seed 42 reproduces 36,168 / 9,043 with 4,231 / 1,058 positives
- SHA-256 of the source file

**Historical pre-lock outcome inspection, disclosed but not used by the pipeline:** full-file single-feature `duration` ROC-AUC 0.8076 and PR-AUC 0.3861; full-file single-feature `day` ROC-AUC 0.4735. Stage 5 recomputes any reported single-feature evidence on development data and the final rulings do not depend on these values.
