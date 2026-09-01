# Default of Credit Card Clients — Decisions File

The file serves as an input to `Per_Dataset_Analytical_Workflow.md`. No analytical run may alter these rulings retrospectively.

**Dataset:** Default of Credit Card Clients, DOI `10.24432/C55S3H`, CC BY 4.0. Creator: I-Cheng Yeh.

**Documentation used:** the UCI dataset page, including its Additional Variable Information block, and the introductory paper — Yeh, I-C. and Lien, C-H. (2009), *Expert Systems with Applications*.

**Lock notes:** class weighting is excluded entirely; `EDUCATION` is nominal; nominal inputs are one-hot encoded for all four models; the decision-tree depth cap is 5; the operating point floors recall at 0.65; no cost-ratio sensitivity analysis is performed; the zero-recorded-amount segment and bill-trajectory structure are recorded as pre-model hypotheses, not as engineered features. No CCD-only model-fitting probe is added beyond the fixed workflow.

**Cross-dataset comparability.** No [FIXED] workflow stage, output schema, evaluation rule or interpretability procedure is changed for CCD. The four objects compared are the four locked end-to-end model pipelines defined by workflow Stage 7. Because those pipelines use model-specific representations, performance differences are not attributed to estimator class alone.

These rulings are locked before execution; no run-time choice may replace them. Any data issue not ruled on below halts the pipeline at Stage 3.

---

## 1. Problem definition [Stage 1]

**Target.** `default payment next month` → `default`.

**Positive class: default.**

The source codes the target directly as `1 = default`, `0 = no default`. No inversion is applied:

```
y = df['default payment next month']     # y = 1 → default (positive)
```

**Verification at Stage 3:** `y.sum() == 6636`. If it returns 23364, an inversion has been applied in error — halt.

**What one observation represents.** One credit card account held with a Taiwanese bank, observed over six consecutive months from April to September 2005, with the October 2005 repayment outcome recorded after the fact.

**Decision supported.** Whether to route an existing account for pre-delinquency intervention — collections outreach, credit-line review, or a payment-plan offer. This is **not** an origination decision. The card is already issued and the exposure exists whichever way the decision falls.

**Stakeholder.** The card-issuing bank, and the account holder as the party contacted on a false positive.

**False positive.** An account that would have paid, routed for intervention. Cost: wasted collections capacity, unnecessary customer contact, some attrition risk. Cheap and reversible.

**False negative.** An account that defaulted, not routed. Cost: the outstanding balance goes bad and only recovery remains.

**Evaluation stance.** The models are evaluated as classification and ranking systems. Probability-level statements are made only where Stage 11 supports them.

### 1.1 Sampling and context note

The sampling design of the delivered UCI file is **undocumented**. No sampling or prevalence correction is applied. The 22.12% positive rate is the observed rate in the delivered file, and calibration assessed at Stage 11 is calibration to that delivered-sample rate; it is not asserted to be population calibration. This contrasts with SGC, whose designed 30% sampling rate is documented, but it does not justify calling CCD an unstratified population sample.

Three context limits carry into the write-up:

1. **Single institution, single window.** One Taiwanese issuer, April–September 2005, with the outcome in October 2005. No macroeconomic covariates are present and no temporal generalisation is claimed.
2. **Fixed cross-section, not a time series.** Every account is observed over the same six months. The month indices are feature positions, not a panel dimension, and no chronological split is used or justified.
3. **Currency.** All monetary amounts are New Taiwan dollars.

### 1.2 Pre-split audit and holdout status

The public source file was inspected before the Stage 4 split so that all dataset-specific rulings could be frozen, matching the protocol used for SGC. Full-file quantities in this decisions file are **pre-split design-audit evidence only**. They may verify a locked ruling or motivate a pre-model hypothesis, but they cannot be recomputed after the split to revise preprocessing, thresholds, model spaces or output selection.

After the split is created, the 6,000-row test set is operationally locked and is not accessed until Stage 13. All predictor–target EDA reported by the workflow, model fitting, tuning, calibration assessment and threshold selection use development data only. Written outputs describe the test result as a **one-time held-out confirmation under a pre-inspected public dataset**, not as blind external validation. Development evidence governs comparative claims, identically to SGC.

---

## 2. Source file and assertions [Stage 2]

| Item | Value |
|---|---|
| Filename | `credit_card_default.csv` |
| File size | 2,867,208 bytes |
| Delimiter | Comma, single header row, **LF line endings** (no CRLF) |
| Expected rows | **30000** |
| Expected columns | **25** (`ID` + 23 predictors + target) |
| Expected positive count | **6636** |
| Expected positive rate | **22.12%** |
| Missing values | **0** |

Assert rows, columns and positive count. Halt on mismatch. Assert the SHA-256 before parsing.

**Provenance note.** UCI distributes this dataset as `default of credit card clients.xls`. The file above is a CSV conversion of that source. The hash recorded here is of the converted CSV, not of the UCI `.xls`, and it is the conversion that must be reproduced. If the hash assertion fails, the file has been re-converted or re-saved and the run halts — it is not to be re-hashed to match.

**Paper–repository discrepancy.** Yeh and Lien (2009) reports 25,000 observations and 5,529 defaults, whereas the delivered UCI file contains 30,000 observations and 6,636 defaults. Both ratios round to 22.12%, but the relationship between the paper sample and the current repository file is undocumented. The paper supports variable context; it does not establish the sampling design of this delivered file.

**Reader note.** Use `pandas.read_csv(path)`. All 25 columns parse as `int64`. The target column name contains spaces and is renamed to `default` once, inside the shared preprocessing module.

The CSV is never modified. All rulings are applied inside the pipeline.

---

## 3. Rulings [Stage 3]

| # | Variable | Issue | Ruling | Verification |
|---|---|---|---|---|
| R1 | `default payment next month` | Target polarity | No inversion. Positive = default. Renamed to `default` | `y.sum() == 6636` |
| R2 | — | Missing values | None present. Assert zero and halt otherwise | `df.isna().sum().sum() == 0` |
| R3 | — | Conventional missing tokens | No `?`, blank, null or non-numeric sentinel appears. The undocumented categorical codes governed by R6–R8 are not asserted to be known non-missing states; their handling is an explicit analytical ruling | All columns parse as `int64`; conventional missing-cell count is zero |
| R4 | `ID` | Identifier column | **Dropped.** Never a feature. Values 1–30000, all unique | `df.ID.nunique() == 30000`; `ID` absent from every design matrix |
| R5 | — | Repeated profiles after identifier removal | **Retained. No de-duplication step.** There are zero fully duplicated rows because `ID` is unique. After excluding `ID`, there are 35 redundant feature-and-target copies (70 rows in repeated feature-and-target groups) and 56 redundant feature-only copies, across 52 feature groups covering 108 rows; **21 feature groups carry conflicting target labels** | Each quantity is asserted with its scope and written to the audit |
| R6 | `EDUCATION` | Undocumented codes 0 (n=14), 5 (n=280), 6 (n=51) | **Fold 0, 5 and 6 into category 4 (`other_or_undocumented`).** Applied for all four models; no claim is made that the undocumented codes share the documented meaning of 4 | Resulting level counts: 10585 / 14030 / 4917 / **468**; assert `set(EDUCATION) == {1,2,3,4}` after folding |
| R7 | `MARRIAGE` | Undocumented code 0 (n=54) | **Fold 0 into category 3 (`other_or_undocumented`).** Applied for all four models; no claim is made that raw 0 shares the documented meaning of 3 | Resulting level counts: 13659 / 15964 / **377**; assert `set(MARRIAGE) == {1,2,3}` after folding |
| R8 | `PAY_0`, `PAY_2`–`PAY_6` | Codes **−2** and **0** appear in the data but are documented nowhere — not on the UCI page, not in Yeh and Lien (2009), and not in the three papers listed as citing the dataset. Code **9** is documented but unobserved | **Keep raw integer codes. No shift, no recode.** Evidence-based labels are attached as documentation only (Section 3.3). The tree family consumes the raw codes; logistic regression bins them per Section 7 | Assert observed code set is `{-2..8}` for `PAY_0`–`PAY_4` and `{-2,-1,0,2..8}` for `PAY_5`, `PAY_6`; assert `9 not in` any column |
| R9 | `BILL_AMT1`–`BILL_AMT6` | Negative values, 590–688 per column | **Retained as plausible credit balances; no source evidence establishes them as errors.** No clipping, no flooring at zero | No clipping step in any pipeline; negative counts asserted per column |
| R10 | `BILL_AMT1`–`BILL_AMT6` | Severe collinearity: mean off-diagonal \|r\| = **0.8865** (range 0.8027–0.9515); VIF 9.71–16.02; PC1 carries 90.56% of variance | **Logistic regression only:** retain `BILL_AMT1`, the latest bill known before the outcome, and prune `BILL_AMT2`–`BILL_AMT6`. The representative is selected by chronology, not target association. **Tree family retains all six raw columns.** No derived bill-to-limit feature is engineered | Feature-set table records the difference; assert `BILL_AMT1` present and `BILL_AMT2`–`BILL_AMT6` absent from the LR transformed matrix; assert all six present in the tree matrix |
| R11 | `PAY_AMT1`–`PAY_AMT6` | Severe right skew (skew 10.64–30.45); all values ≥ 0 | **`log1p` transform in the logistic regression pipeline only.** Tree family receives raw values. Marginal association strengthens from r = −0.0729 raw to −0.1703 transformed for `PAY_AMT1` | Feature-set table records the difference; assert no negative values before transform |
| R12 | `PAY_AMT1`–`PAY_AMT6` | Mutually near-independent: mean off-diagonal \|r\| = 0.189, max 0.286 | **No collapse.** All six retained in every model | Six columns asserted present in both matrices |
| R13 | — | Row order | No ordering detected. Correlation of `ID` with target −0.0140, with `LIMIT_BAL` +0.0262, with `AGE` +0.0187, with `PAY_0` −0.0306. Target rate 0.2214 in the first 5,000 rows against 0.2116 in the last 5,000 | Row position is never a feature; the split is shuffled and stratified; no positional slicing anywhere |
| R14 | All | Outliers | **None removed.** `BILL_AMT3` reaches 1,664,089 on a 500,000 limit; `PAY_AMT2` reaches 1,684,259; bills exceed the stated limit in 3,931 rows (13.10%), maximum ratio 10.69. These observations show that bill amount cannot be cleaned by enforcing `BILL_AMT <= LIMIT_BAL`; they do not establish the contractual meaning of the limit | No outlier filter in any pipeline |
| R15 | `LIMIT_BAL` | Off-grid values: **16,000** appears twice (IDs 5,477 and 27,004) and **327,680** appears once (ID 12,526); these are the three rows not divisible by 10,000. No verified explanation | **Retained unchanged.** Recorded as unexplained anomalies, not corrected | Both values and all three IDs asserted present; noted in the audit |
| R16 | `SEX` | Sensitive demographic attribute | **Retained** as a binary predictor in all four models. Removing it would alter the dataset's published feature set. No fairness metrics or legal-compliance claims are made; scope is confined to what the models reveal about the attribute (Section 16) | Present in both design matrices |

### 3.1 Leakage register [Stage 2 output]

**No variables are registered.** All 23 predictors are account attributes observed between April and September 2005; the target is the October 2005 outcome. No feature post-dates the target. There is no CCD analogue to the Bank Marketing `duration` problem.

`leakage_register.csv` is still produced with a header row and no entries, so the absence is a recorded result rather than a missing file.

### 3.2 Feature types and display names [Stage 3 output]

Raw column codes are never shown in a table or figure.

| Column | Display name | Feature type | Treatment |
|---|---|---|---|
| `LIMIT_BAL` | Credit limit (NT$) | Quantitative | Numeric; robust-scaled for logistic regression |
| `AGE` | Age (years) | Quantitative | Numeric; robust-scaled for logistic regression |
| `SEX` | Sex | Binary categorical | `sex_female = 1` when raw code is 2; otherwise 0 |
| `EDUCATION` | Education level | **Nominal categorical** | LR: drop-reference one-hot; tree family: full one-hot |
| `MARRIAGE` | Marital status | Nominal categorical | LR: drop-reference one-hot; tree family: full one-hot |
| `PAY_0` | Repayment status, September | Ordinal | LR: four behavioural buckets; tree family: raw integer code |
| `PAY_2` | Repayment status, August | Ordinal | LR: three behavioural buckets; tree family: raw integer code |
| `PAY_3` | Repayment status, July | Ordinal | LR: three behavioural buckets; tree family: raw integer code |
| `PAY_4` | Repayment status, June | Ordinal | LR: three behavioural buckets; tree family: raw integer code |
| `PAY_5` | Repayment status, May | Ordinal | LR: three behavioural buckets; tree family: raw integer code |
| `PAY_6` | Repayment status, April | Ordinal | LR: three behavioural buckets; tree family: raw integer code |
| `BILL_AMT1` | Bill amount, September (NT$) | Quantitative | Retained in every model; robust-scaled for LR (R10) |
| `BILL_AMT2` | Bill amount, August (NT$) | Quantitative | Tree family only; pruned for LR (R10) |
| `BILL_AMT3` | Bill amount, July (NT$) | Quantitative | Tree family only; pruned for LR (R10) |
| `BILL_AMT4` | Bill amount, June (NT$) | Quantitative | Tree family only; pruned for LR (R10) |
| `BILL_AMT5` | Bill amount, May (NT$) | Quantitative | Tree family only; pruned for LR (R10) |
| `BILL_AMT6` | Bill amount, April (NT$) | Quantitative | Tree family only; pruned for LR (R10) |
| `PAY_AMT1` | Payment made, September (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `PAY_AMT2` | Payment made, August (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `PAY_AMT3` | Payment made, July (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `PAY_AMT4` | Payment made, June (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `PAY_AMT5` | Payment made, May (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `PAY_AMT6` | Payment made, April (NT$) | Quantitative | `log1p` transformed and robust-scaled for LR (R11) |
| `default payment next month` | Default next month | Target (binary) | Positive class = default (R1) |

**Taxonomy: 14 quantitative, 6 ordinal, 1 binary, 2 nominal — 23 predictors.** No derived model feature is added.

**Derived level labels.** `EDUCATION`: 1 graduate school, 2 university, 3 high school, 4 other or undocumented (folded, R6). `MARRIAGE`: 1 married, 2 single, 3 other or undocumented (folded, R7).

**`EDUCATION` is assigned nominal, against an apparent ordering.** Levels 1, 2 and 3 are monotone in default rate (0.1923, 0.2373, 0.2516), which invites an ordered score. Two pieces of evidence rule against it. First, the log-odds spacing is uneven — steps of +0.2676 then +0.0770, a ratio of roughly 3.5 to 1 — so an equal-interval score misfits the very structure it would be adopted to capture. Second, **the monotonicity does not survive conditioning.** Within credit-limit tertiles the default rates run 0.2712 / 0.3095 / 0.3148 in the low tertile, 0.2040 / 0.2156 / 0.1986 in the mid, and 0.1472 / 0.1475 / 0.1653 in the high — monotone in the low tertile only. High-school clients hold systematically smaller limits (mean 126,550 against 212,956 for graduate school), and that carries most of the marginal gradient. The folded category 4 sits at log-odds −2.5788, a step of −1.4886 from level 3, and has no position on an education scale at all.

This is the point of contrast with SGC, where `credit_history` was assigned ordinal on monotonic evidence covering **every** level. It is also unforced here: logistic regression has roughly 101 events per parameter in an inner-fold training set (Section 7), so there is no parsimony pressure of the kind that drove SGC's ordinal treatment.

### 3.3 Repayment-status code labels [documentation only]

Attached for interpretation. **Not a data transformation.**

| Code | Label | Basis |
|---|---|---|
| −1 | Paid duly | Documented in Yeh and Lien (2009) |
| −2 | No balance or no consumption that month | **Inferred.** For `PAY_2`–`PAY_6`, between 55.7% and 70.8% of code −2 rows carry a contemporaneous bill ≤ 0, with a median bill of exactly 0 |
| 0 | Revolving credit — account current with an outstanding balance | **Inferred.** Median contemporaneous bill 31,032–49,605 NT$, an order of magnitude above codes −2 and −1, while the default rate (0.1281 for `PAY_0`) is the lowest of any code. A carried balance serviced on time is not delinquency |
| 1–8 | Payment delay of that many months | Documented |
| 9 | Delay of nine months or more | Documented; **unobserved** |

**Evidence for −2 is weaker in `PAY_0`.** Against `BILL_AMT1` the median is 1,179 and only 32.7% of rows carry a bill ≤ 0, unlike the lagged columns. State this asymmetry rather than smoothing it.

**The +1 shift hypothesis is rejected.** Adding 1 so that "0 = duly" is (a) a mechanical no-op for every model used here — tree splits are threshold-based and invariant to a constant shift, and the LR encoding is bucketed rather than linear; (b) it overwrites the single documented anchor at −1; and (c) it relabels code 0, the modal and lowest-risk category, as one month delinquent, which the default-rate evidence contradicts directly.

**The repayment-status cliff.** Every code at or below 0 sits at or below the base rate; `PAY_0` code 1 jumps to 0.3395 and code 2 to 0.6914. This discontinuity is the dominant marginal signal in the dataset and is the primary structure any model must recover.

---

## 4. Split and fold design [Stage 4]

| Item | Value |
|---|---|
| Split | Stratified 80/20 on `default` |
| Development set | **24,000 rows, 5,309 positive** |
| Test set | **6,000 rows, 1,327 positive** |
| Random seed | **42** (split, folds, search, permutation, SHAP, bootstrap) |
| Outer folds | 5, stratified — 4,800 validation rows each, 1,061–1,062 positives |
| Inner folds | 3, stratified — roughly 12,800 training rows, 2,832 positives |
| Shuffle | **True** |

**Sample-size note for the write-up.** This is the largest of the four datasets. The test set holds 1,327 positives against SGC's 60, so the bootstrap interval on test PR-AUC will be comparatively narrow and the confirmation evidence correspondingly firmer. The nested cross-validation estimate remains the headline quantity; the test evaluation remains confirmation.

---

## 5. Audit expectations [Stages 2 and 5]

Computed from the source file before the run, so the pipeline's own output can be checked against an independent count.

### 5.1 Row order — no ordering detected

Correlations of `ID` with the target and with major predictors are all below 0.031 in absolute value (R13), and the target rate is 0.2214 in the first 5,000 rows against 0.2116 in the last 5,000. Unlike SGC, the file is not sorted on the target. The split is shuffled and stratified regardless; row position is never a feature.

### 5.2 Category codes

Undocumented codes appear in three places: `EDUCATION` (0, 5, 6), `MARRIAGE` (0), and the repayment-status columns (−2 and 0). The first two are folded under R6 and R7. **The repayment codes are not resolvable from any available documentation** — the UCI page, the introductory paper and all three citing papers listed on the UCI page are silent on them. This is a recorded limitation, not a deferred task.

### 5.3 Correlated blocks [determines Stage 7 handling]

Three blocks, handled differently.

**Bill amounts — pruned for logistic regression.** Mean off-diagonal Pearson \|r\| = **0.8865**, range 0.8027–0.9515, with a clean autoregressive band structure: mean correlation 0.9380 at one month's separation, decaying monotonically to 0.8027 at five. Spearman gives mean \|rho\| = 0.8409, so this is not an artefact of skew. VIF runs 9.71–16.02 across the six columns; the correlation matrix has condition number 134.7; PC1 carries **90.56%** of the standardised variance with near-uniform loadings.

**What the pruning discards.** PC2 carries 5.10% of the variance, with loadings running monotonically from −0.536 on the September bill to +0.529 on the April bill — a recent-versus-older contrast, that is, a **trend** component. It out-associates PC1 with the target: correlation with default is +0.0123 for PC1 against **+0.0208** for PC2. Retaining only the latest bill deliberately gives logistic regression a single contemporaneous level measure and removes the historical trajectory. This is recorded honestly as a cost, and is a pre-model hypothesis under Section 5.6.

**Repayment status — no collapse needed.** Raw codes correlate at mean \|r\| = 0.6564 (max 0.820, adjacent months), but the Section 7 bucketing absorbs most of that dependence. The realised LR design block of 13 dummies has mean \|r\| = 0.2104, maximum pairwise r = 0.6317, and **maximum VIF 2.36**. Coefficients on the repayment block will be readable.

**Payment amounts — no collapse.** Mean \|r\| = 0.189, max 0.286 (R12).

### 5.4 Sparse levels

Smallest categories after folding, and their expected survival through the fold structure:

| Category | Full | Development | Smallest inner training fold | Test |
|---|---|---|---|---|
| `EDUCATION` = 4 (other or undocumented) | 468 | 374 | ~199 | ~93 |
| `MARRIAGE` = 3 (other or undocumented) | 377 | 301 | ~160 | ~75 |
| `PAY_0` delay 3m+ | 463 | 370 | ~197 | ~92 |
| `PAY_6` delay 3m+ | 313 | 250 | ~133 | ~62 |

No degenerate or constant columns arise. All one-hot encoders use `handle_unknown='ignore'` as insurance rather than necessity.

**Two folded categories warrant advance warning.** `EDUCATION` = 4 is internally heterogeneous — its components run 0/14, 0.0569, 0.0643 and 0.1569 — and lands at an aggregate 0.0705, markedly **below** every documented level. It is not a low-information residual: median credit limit 170,000 against 140,000 for the rest, and 86.97% of its rows sit at `PAY_0 ≤ 0` against 77.12%. No verified explanation exists for why an undocumented education category should be the lowest-risk group. `MARRIAGE` = 3 sits at log-odds −1.1743, effectively identical to `MARRIAGE` = 1 at −1.1819, on n = 377; a near-zero coefficient with a wide interval is the predictable outcome and is not a finding.

### 5.5 Zero recorded bill-and-payment segment — pre-model hypothesis

**795 rows (2.65%) carry all six bill amounts and all six payment amounts at exactly zero.** Their default rate is **0.3761** against a base rate of 0.2212.

Their repayment codes are perfectly structured. `PAY_2` through `PAY_6` are −2 in **all 795 rows without exception**. `PAY_0` is either −2 (n = 300, default rate 0.3433) or **1 (n = 495, default rate 0.3960)**.

Those 495 rows are apparently inconsistent with a literal reading of the documented scale: `PAY_0 = 1` appears alongside zero recorded bills and payments in all six months. Reporting conventions, omitted account activity or a code-semantics problem could each explain the pattern; the data do not distinguish them. It supports retaining the asymmetric bucketing in Section 7 without asserting that code 1 has been definitively mistranslated.

**Pre-model hypothesis.** The tree family may recover this segment from the six bill-amount thresholds unaided. Logistic regression sees only a zero latest bill, alongside low-balance rows with recorded activity. **No segment indicator is engineered.** The hypothesis is revisited descriptively at Stage 17 as candidate evidence that the ensembles used structure the linear model could not represent; model-score differences alone cannot establish that explanation.

**Robustness.** The repayment cliff is not an artefact of this segment. Excluding the 795 rows, `PAY_0` = 1 still sits at 0.3307 against 0.1281 for code 0.

### 5.6 Bill-to-limit-ratio non-monotonicity — pre-model hypothesis

The audit-only quantity `average_bill_to_limit_ratio = mean(BILL_AMT1..6) / LIMIT_BAL` is U-shaped in default. Decile rates: 0.2600, 0.1937, 0.1433, 0.1420, 0.1587, 0.1943, 0.2397, 0.2560, 0.2990, 0.3253. Point-biserial correlation with the target is only 0.1155, which understates the association because the two arms partly cancel. It is EDA evidence, not an engineered model feature.

It is not purely a repayment-status confound. Within `PAY_0 ≤ 0` the quintile rates run 0.1833, 0.1109, 0.0951, 0.1296, 0.1727.

**Roughly half the low arm is the zero-recorded-amount segment.** The lowest decile decomposes into 795 such rows at 0.3761 and 2,209 other rows at 0.2187. Excluding the segment and re-deciling, the low arm flattens from 0.2600 to 0.2232.

**Pre-model hypothesis.** The tree family can represent interactions between the credit limit and the six bills, whereas logistic regression receives additive terms for the limit and latest bill only. Bill trajectory (Section 5.3), the zero-recorded-amount segment (Section 5.5) and this curvature are the three named structures available to the tree family and not explicitly represented for logistic regression. All three are revisited at Stage 17. **No ratio, interaction, binned, spline or quadratic term is added.**

### 5.7 Marginal associations recorded before the run

- `LIMIT_BAL` decile default rates decrease monotonically: 0.3585 at the lowest decile to 0.1187 at the highest. A linear term is adequate.
- `AGE` band rates are mildly U-shaped — 0.2666 at 21–25, minimum 0.1943 at 31–35, 0.2684 at 61–79 — a spread of roughly seven percentage points. Retained as a linear term; the curvature is recorded, not modelled.
- `AGE` spans 21–79 but takes only 56 distinct values: **76, 77 and 78 are absent.** Consistent with sparsity in the upper tail (272 rows above 60).
- `MARRIAGE` is substantially an age proxy: correlation with `AGE` is −0.412, mean age 40.02 married, 31.45 single, 42.08 others. Neither coefficient should be read in isolation at Stage 16.
- Dividing by `LIMIT_BAL` reverses the sign of the association with limit: mean bill correlates +0.3020 with `LIMIT_BAL`, while the audit-only `average_bill_to_limit_ratio` correlates −0.3834. The ratio is not entered into a model.

---

## 6. Evaluation framework [Stage 6]

| Purpose | Metric |
|---|---|
| Primary model selection | PR-AUC (average precision) |
| Discrimination | ROC-AUC |
| Probability quality | Calibration slope and intercept, Brier score, log loss |
| Positive-class behaviour | Precision and recall at the declared operating point |
| Error profile | Confusion matrix, FP and FN counts |

**Positive class:** default. **Base rate:** 22.12%. A prevalence/dummy baseline is recorded alongside model PR-AUC. If a model's PR-AUC is at or below the matched baseline, the pipeline raises a prominent warning, runs target-polarity and prediction-score diagnostics, and continues. The result is investigated and documented rather than suppressed.

**Domain description of errors**, used verbatim in written outputs:

- **False positive** — an account that paid, routed for intervention.
- **False negative** — an account that defaulted, not routed.

**Class imbalance is handled through metric choice and operating-point selection only.** No class weighting, no resampling, no loss modification. `class_weight` remains at its default and `scale_pos_weight` remains at 1.0 for XGBoost. No weighted variant is fitted at any stage, and no weighting dimension is added to any output schema.

---

## 7. Model-specific preprocessing [Stage 7]

The four models do not receive identical feature matrices. This table is authoritative; `feature_sets_by_model.csv` must reproduce it.

### Shared steps, applied at load

Drop `ID`; rename the target to `default`; apply the `EDUCATION` and `MARRIAGE` folds (R6, R7). These are fixed lookups estimating nothing from data, so they carry no leakage risk. Everything below is fitted within folds.

### Logistic regression

| Feature group | Treatment |
|---|---|
| `LIMIT_BAL`, `AGE` | Numeric, robust-scaled |
| `BILL_AMT1` | Retained as the chronology-selected representative of the bill block; robust-scaled (R10) |
| `BILL_AMT2`–`BILL_AMT6` | Pruned (R10) |
| `PAY_AMT1`–`PAY_AMT6` | `log1p` transformed, then robust-scaled (R11) |
| `SEX` | Explicit 0/1 indicator per Section 3.2 |
| `EDUCATION`, `MARRIAGE` | One-hot, declared reference level dropped, `handle_unknown='ignore'` |
| 6 repayment columns | Behavioural buckets, one-hot, reference dropped |
| Regularisation | L2, strength `C` tuned in the inner loop |

**Robust rather than standard scaling**, given residual skew and extreme upper tails after transformation.

**Repayment bucketing — asymmetric, and deliberately so.**

- **`PAY_0` — four buckets:** `non_positive_status (≤0)`, `delay_1m`, `delay_2m`, `delay_3m+`. Counts 23,182 / 3,688 / 2,667 / 463; default rates 0.1383 / 0.3395 / 0.6914 / 0.7192.
- **`PAY_2`–`PAY_6` — three buckets:** `non_positive_status (≤0)`, `delay_1_2m`, `delay_3m+`.

A uniform four-bucket scheme is **not viable**. Observed counts in a one-month-delay bucket are 3,688 for `PAY_0` but 28, 4, 2, 0 and 0 for `PAY_2` through `PAY_6` — two constant-zero dummies and three fitted on 28 rows or fewer. In `PAY_0`, by contrast, the one- and two-month levels are both well populated and sharply separated, so merging them would discard real signal. The asymmetry is empirically driven, declared before fitting, and corroborated independently by Section 5.5.

Merged `delay_1_2m` counts for `PAY_2`–`PAY_6`: 3,955 / 3,823 / 3,161 / 2,626 / 2,766, default rates 0.5535 / 0.5153 / 0.5233 / 0.5419 / 0.5065. `delay_3m+` counts 483 / 390 / 349 / 342 / 313, rates 0.5942 / 0.5949 / 0.6447 / 0.6608 / 0.6709.

**Locked reference levels for logistic regression:** `EDUCATION = 2` (university, n = 14,030); `MARRIAGE = 2` (single, n = 15,964); and `non_positive_status` for all six repayment columns. These are the most frequent observed levels and must be recorded in the coefficient-table metadata. Changing a reference level is a specification change, not a tuning choice.

**Parameter count: 28** — 9 scaled numeric (`LIMIT_BAL`, `AGE`, `BILL_AMT1`, six transformed payment amounts), 1 binary, 3 `EDUCATION` dummies, 2 `MARRIAGE` dummies, 3 `PAY_0` dummies, and 2 dummies for each of `PAY_2`–`PAY_6`.

Events per parameter: **189.6 across the development set, 151.7 in an outer-fold training set, and 101.1 in an inner-fold training set** where the search runs. This is roughly twenty-six times SGC's inner-fold figure of 3.9. Logistic regression is not parameter-starved on this dataset, and if it fails to match the ensembles, sample size cannot be the explanation.

### Decision tree, random forest, XGBoost

| Feature group | Treatment |
|---|---|
| `LIMIT_BAL`, `AGE` | Raw values, no scaling, no transformation |
| `BILL_AMT1`–`BILL_AMT6` | All six retained raw, negatives intact (R9, R10) |
| `PAY_AMT1`–`PAY_AMT6` | Raw values, no transformation |
| 6 repayment columns | Raw integer codes (R8) |
| `SEX` | Explicit 0/1 indicator per Section 3.2 |
| `EDUCATION`, `MARRIAGE` | **Full one-hot encoding with no dropped reference**, `handle_unknown='ignore'` |

The tree-family design matrix contains **28 terms**: 14 quantitative, 6 ordinal codes, 1 binary, 4 full `EDUCATION` indicators and 3 full `MARRIAGE` indicators. Coincidentally the same width as the logistic regression matrix — **the two matrices are not the same columns**, and no output may imply otherwise.

No split may operate directly on an arbitrary nominal category code; a nominal split is read as category present versus absent. XGBoost receives the pre-encoded matrix, so `enable_categorical` is not used. Feature-importance and explanation outputs are reduced from encoded-term level to the original 23-variable level using the workflow's declared aggregation rule.

**Retaining the bill block for the tree family is deliberate.** Under collinearity of this severity, impurity-based importance spreads across the six correlated columns while permutation importance does not. That divergence is evidence for Stage 16, not a defect to be preprocessed away.

---

## 9. Hyperparameter search [Stage 9]

Randomised search, `n_iter = 40`, seed 42, identical budget across all four models, scored on PR-AUC in the inner loop.

| Model | Search space |
|---|---|
| Logistic regression | `C`: log-uniform 1e-3 to 1e2; `penalty`: l2; `solver`: lbfgs; `max_iter`: 2000 |
| Decision tree | `max_depth`: 2–**5** (capped); `min_samples_leaf`: 20–500; `min_samples_split`: 50–1000; `criterion`: gini, entropy; `ccp_alpha`: 0.0–0.02 |
| Random forest | `n_estimators`: 300–800; `max_depth`: 3–16 or None; `min_samples_leaf`: 1–50; `max_features`: sqrt, log2, 0.5 |
| XGBoost | `n_estimators`: 200–600; `max_depth`: 2–8; `learning_rate`: log-uniform 0.01–0.3; `subsample`: 0.6–1.0; `colsample_bytree`: 0.6–1.0; `min_child_weight`: 1–20; `reg_lambda`: log-uniform 0.1–10 |

Leaf and split minima are scaled up from SGC's values to reflect a development set thirty times larger; depth ranges for the ensembles are widened for the same reason.

**Decision tree maximum depth is capped at 5.** The search may not exceed it; realised depth is asserted `<= 5` in the invariant checks. Depth is selected by PR-AUC **within** the capped range by the inner loop — the cap bounds the range, it does not fix the value.

The cap is a specification decision, not a tuning outcome. Depth 5 permits up to 32 leaves; depth 8 permits 256. Neither figure depends on having 24,000 rows, so the larger sample is not a reason to relax it. The binding reason is cross-dataset: SGC caps the tree at 5, and a different cap here would place the decision tree at a different point on the interpretability spectrum in each dataset, degrading the cross-dataset interpretability comparison. Record that the ensembles were not similarly constrained.

**Both the inner-loop search score and the nested outer estimate are recorded for every model**, so any divergence between them is visible. `search_score` is defined in workflow Stage 9.

---

## 12. Threshold selection [Stage 12]

**Rule: floor recall at 0.65, maximise precision subject to that floor.**

Applied identically to all four models, on out-of-fold development predictions, fixed before the sweep is inspected.

**Deterministic tie-break:** among eligible thresholds, choose maximum precision; if tied, choose maximum recall; if still tied, choose the highest threshold.

**Why recall is floored.** The positive class is default and the false negative is the costlier error. Flooring recall is also mechanically safe: recall is monotone in the threshold, so every model can attain any recall floor and the Stage 12 halt condition will not fire spuriously.

**Why 0.65 — a declared analytical anchor, with naive-rule context.** Four rules available without a fitted model were computed during the pre-split design audit:

| Rule | Flagged | Recall | Precision |
|---|---|---|---|
| `PAY_0 ≥ 2` | 3,130 (10.43%) | 0.3281 | 0.6955 |
| `PAY_0 ≥ 1` | 6,818 (22.73%) | 0.5167 | 0.5029 |
| any `PAY_* ≥ 2` | 8,380 (27.93%) | 0.5847 | 0.4630 |
| **any `PAY_* ≥ 1`** | **10,069 (33.56%)** | **0.6483** | **0.4273** |

A floor of 0.65 places the operating point in the same recall region as the best naive rule. It is not claimed to exceed that rule in every split. Under the locked development split, `any PAY_* ≥ 1` is independently expected to produce recall 0.6553 and precision 0.4293; Stage 5 verifies those development-only values without changing the floor. Model precision and recall are compared with the naive rule descriptively, with their realised recall difference shown rather than treated as an exact matched-recall contest.

The decision context supports the same region. This is intervention routing, not origination — the exposure exists either way — so the false negative is costlier but less asymmetrically than in a grant decision, where the credit need not have been extended at all. No unsupported claim is made about the operational usefulness of any higher-recall region before model outputs exist.

Like SGC's 0.75, **0.65 is a declared analytical anchor and is not derived from a verified economic cost.** It must be described that way in every written output.

**Also recorded:**

- Precision, recall, FP and FN at the conventional `0.50` threshold, **as a reference convention only**.
- The development-only naive-rule benchmarks, as contextual evidence in `outputs/eda/eda_findings.md` and traceable registry rows. They do not alter Stage 12 selection or the fixed `thresholds_selected.csv` schema.
- **No cost-ratio sensitivity analysis.** No cost ratio is documented for this dataset — Yeh and Lien (2009) supplies none and there is no carried convention analogous to SGC's Statlog 5:1. Nothing of the kind may appear in calculations, tables, figures or written outputs.

---

## 16. Interpretability evidence [Stage 16]

The governing rules — development-set data source, signed-summation aggregation to variable level, and no variable-level aggregate for logistic regression coefficients — are set out in workflow Stage 16 and apply to every dataset. Dataset-specific points follow.

**Scale.** Both design matrices carry 28 encoded terms. Reduction to the original variable level applies to `EDUCATION`, `MARRIAGE` and the six repayment columns for logistic regression, and to `EDUCATION` and `MARRIAGE` for the tree family. No variable here approaches the ten-term exposure that made SGC's `purpose` the most reduction-sensitive case.

**Raw-variable permutation universe.** Cross-fitted permutation importance follows the fixed workflow and permutes each of the same 23 raw predictors before model-specific preprocessing. For logistic regression, the five pruned bill variables have exactly zero pipeline dependence and therefore zero permutation importance; `BILL_AMT1` is the retained bill variable. All four models have 23-variable permutation rankings, and every permutation-based top-ten overlap remains populated exactly as in SGC.

**Logistic-regression SHAP universe.** Linear SHAP operates on the transformed LR matrix and maps directly back to the 18 original predictors that the LR pipeline uses. `feature_rankings_combined.csv` keeps exactly one row for each of the 23 original predictors. The LR SHAP rank and direction are blank, with reason recorded, for the five bill variables pruned from that model; no extra derived-feature row is introduced. The LR `perm_vs_shap_rank_corr` is computed over the 18 original predictors present in both LR rankings, and that comparison universe is recorded. Tree-model values and every permutation-based top-ten overlap remain defined. This preserves the fixed schema and avoids an invented attribution.

**Expect native and permutation importance to diverge on the bill block.** With mean \|r\| = 0.8865 across six columns, impurity and gain importance will spread across the block while permutation importance, which moves correlated columns independently, will not. This is the CCD counterpart to SGC's `foreign_worker` case, and the mechanism differs: SGC's divergence arose on a near-null predictor in a highly unbalanced binary variable, CCD's arises from collinearity within a block. **Two mechanisms, one conclusion about post-hoc importance, across two datasets** — this is CRQ2 material if Stage 16 supports it.

**`SEX` is treated as a sensitive demographic attribute.** Its native-versus-permutation behaviour is inspected explicitly and recorded. Scope is confined to what the models reveal about the attribute; no fairness metric, legal-compliance conclusion or fairness claim is made.

**Data source.** The test set holds 1,327 positives, so the development-set rule is less load-bearing here than on SGC. It is applied identically regardless, for cross-dataset uniformity.

---

## 17. Declared probes [Stage 17]

**No CCD-only model-fitting probe is added.** Stage 17 follows the fixed workflow and revisits the pre-model structural hypotheses using the standard EDA and interpretability outputs. The decision-tree depth cap remains 5 and is assessed through the same realised depth, leaf count, decision rules and nested PR-AUC evidence recorded for SGC. A depth sweep would add a model-fitting experiment and artifact set not produced for SGC; it is therefore excluded to preserve cross-dataset comparability. Any future depth sensitivity analysis must be separately authorised and applied consistently to all datasets.

---

## Recorded limitations

1. **Undocumented repayment codes.** Codes −2 and 0 account for a majority of observations in every repayment column and are documented nowhere in the available literature. The labels in Section 3.3 are inferences from bill-amount evidence, and the evidence is weaker for `PAY_0` than for the lagged columns.
2. **Apparently inconsistent code assignments.** 495 accounts with zero recorded bills and payments across all six months carry `PAY_0 = 1` (Section 5.5). Reporting conventions, omitted activity and code semantics cannot be distinguished from the delivered data.
3. **Label-conflicting duplicates.** 21 feature-identical groups carry differing outcomes (R5). These are in-sample evidence of irreducible error: no model of any complexity can separate them, which places a bound on achievable performance.
4. **Discarded bill trajectory.** Pruning to the latest bill removes a modestly predictive historical trend component from logistic regression (Section 5.3). Its `BILL_AMT1` coefficient describes the latest bill only.
5. **Bill-to-limit curvature unmodelled.** The U-shaped association (Section 5.6) is not represented by a single linear term, by declared choice.
6. **`EDUCATION` folding is heterogeneous.** Category 4 mixes one documented and three undocumented codes with default rates from 0.0000 to 0.1569 (Section 5.4), and its low aggregate rate has no verified explanation.
7. **Confounded demographics.** `MARRIAGE` correlates −0.412 with `AGE`; `EDUCATION` is confounded with `LIMIT_BAL` (Section 3.2). Neither coefficient supports a standalone reading.
8. **Single institution, single six-month window, 2005 Taiwan.** No macroeconomic covariates; no temporal generalisation claimed (Section 1.1).
9. **Unexplained anomalies retained.** `LIMIT_BAL` = 16,000 in two rows and 327,680 in one row (R15); bills exceed the stated limit in 13.10% of rows with a maximum ratio of 10.69 (R14). These observations are retained without inferring the contractual meaning of the limit.
10. **Asymmetric feature matrices.** The logistic regression receives a curated matrix — collinearity resolved, skew transformed, repayment status discretised — while the tree family receives raw data. Each family gets the representation a competent practitioner would build for it, but the asymmetry must be stated in the methodology so no result is read as reflecting the algorithm alone.

---

## Deviations to log

**From the earlier Default of Credit Card Clients work.** The earlier exploratory analysis used a single non-nested `StratifiedKFold(5)` with tuning search scores reported directly. This workflow uses nested 5-outer / 3-inner cross-validation per workflow Stage 4, which is **[FIXED]** for every dataset without exception.

Consequences:

1. Figures from the earlier run **are not comparable** to figures produced under this workflow and must not be quoted alongside them. Any pre-existing CCD rows in a results registry are discarded in full, not appended to.
2. The justification is empirical: the SGC data showed a random-forest search score of 0.653 against a nested estimate of 0.614, reversing the LR-versus-RF ranking.
3. The words *significant*, *significantly*, *statistically significant* and *p-value* must not appear anywhere in the outputs in relation to differences between models or metrics.

**From an earlier CCD plan.** A weighted-versus-unweighted contrast at baseline was previously contemplated. It is **excluded entirely** (Section 6), consistent with workflow Stage 6 and with the SGC treatment.

**From the workflow text.** None. Stage 7's instruction to "bin ordinal codes into interpretable buckets" for logistic regression is followed here — unlike SGC, which entered ordinal codes as ordered scores and logged that as a divergence.

**From the UCI variable descriptions.** `EDUCATION` is assigned nominal despite a marginal monotone gradient across its three documented substantive levels, on the conditional evidence in Section 3.2. Recorded as an analytical judgement.

---

## Verification log

Computed directly from `credit_card_default.csv` before this file was written:

- Shape 30,000 × 25; all columns `int64`; zero missing values; LF line endings; SHA-256 as recorded in Section 2
- Target: 6,636 positives, 23,364 negatives, base rate 0.2212, negative-to-positive ratio 3.5208
- `ID` 1–30,000, all unique; no row ordering detected (Section 5.1)
- Off-grid credit limits verified: 16,000 at IDs 5,477 and 27,004; 327,680 at ID 12,526 (R15)
- Repeated profiles after excluding `ID`: 35 redundant feature-and-target copies (70 rows in repeated feature-and-target groups), 56 redundant feature-only copies, 52 feature groups / 108 rows, 21 groups with conflicting labels (R5); zero fully duplicated rows
- Full level counts and default rates for `SEX`, `EDUCATION`, `MARRIAGE` and all six repayment columns, before and after folding
- Bill-amount Pearson and Spearman correlation matrices, VIF, condition number and PCA eigenvalues (Section 5.3)
- Payment-amount correlation matrix and skew (R11, R12)
- Realised logistic-regression repayment-block correlation and VIF after bucketing (Section 5.3)
- `average_bill_to_limit_ratio` distribution and decile default rates, with and without the zero-recorded-amount segment (Section 5.6)
- Zero recorded bill-and-payment segment: 795 rows, complete repayment-code profile (Section 5.5)
- Marginal association checks for `LIMIT_BAL`, `AGE`, `MARRIAGE` and `EDUCATION`, including conditioning on credit-limit tertiles and on `PAY_0` (Sections 3.2, 5.7)
- Naive-rule recall and precision benchmarks (Section 12)
- Split arithmetic and per-fold positive counts (Section 4); events per parameter (Section 7)
