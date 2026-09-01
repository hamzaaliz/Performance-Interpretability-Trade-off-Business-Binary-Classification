# Iranian Churn — Decisions File

This file serves as an input to `Per_Dataset_Analytical_Workflow.md`. No analytical run may alter these rulings retrospectively.

**Dataset:** Iranian Churn, DOI `10.24432/C5JW3Z`, CC BY 4.0.

**Documentation used:** the UCI dataset page — the Additional Information block, the Variables Table, and the Additional Variable Information block. There is no accompanying code table; the source limitations and inconsistencies relevant to execution are recorded at R5, R6, R16 and R17.

**Post-draft amendments folded in before lock:** R11 extended to cover all six columns carrying a zero mass, since `Call Failure` (702) and `Frequency of SMS` (603) were previously unruled and would have halted Stage 3 (Section 3.0); R14 added for the zero-denominator case in mean call duration (Section 3.4); R15 added to fix which recodes are fold-safe and which are learned; Stage 18 reproducibility re-run formally waived; `Age Group` encoder categories declared explicitly.

**Pre-execution amendments:** R16 added for the 120 rows where `Distinct Called Numbers` exceeds `Frequency of use`; R17 added after establishing the exact empirical construction of `Customer Value`; `Customer Value` is dropped from all four model families; the dormant conflicting-pattern count is corrected from 13 to 11 patterns (56 of 64 rows); feature-count arithmetic and the binary-direction description are corrected; and the execution workflow is merged from amendment 7 plus amendment 8. These changes precede all splitting, fitting and test-set evaluation and must be reported in the Facts Document as specification amendments.

---

## 1. Problem definition [Stage 1]

**Target.** `Churn` → `churn`.

**Positive class: churn.**

The UCI page codes `Churn` as `1: churn`, `0: non-churn`. The coding already places the minority event at 1, so:

```
y = Churn               # no inversion
```

**Verification at Stage 3:** `y.sum() == 495`. If it returns 2655, an inversion has been applied that this file does not authorise — halt.

**What one observation represents.** One customer of an Iranian telecommunications operator, described by usage and account attributes aggregated over the first nine months of a twelve-month window, with churn status recorded at the end of month twelve. The intervening three months are the planning gap declared on the UCI page.

**Decision supported.** Whether to direct a retention intervention at a customer at the end of the nine-month observation window. Pricing, tariff redesign and network remediation are outside the primary decision context.

**Stakeholder.** The operator's retention function. The customer is the party receiving or not receiving the intervention.

**False positive.** A customer who would have stayed, flagged as at risk and given a retention offer. Cost: the offer itself and any margin conceded, plus contact fatigue.

**False negative.** A customer who churned, not flagged. Cost: the remaining lifetime value of that customer and the cost of reacquisition.

**Evaluation stance.** The models are evaluated as classification and ranking systems. Probability-level statements are made only where Stage 11 supports them.

### 1.1 Observation-window caveat

The nine-month aggregation followed by a three-month planning gap is a genuine prospective design and is the reason no leakage registration is required (Section 3.1). Two consequences bind:

1. Every predictor is a **nine-month aggregate**, not a rate or a trend. `Seconds of Use` is total seconds over nine months, not seconds per month. Coefficients and importances are read on that scale, and no output may describe a predictor as a monthly or per-period quantity.
2. `Subscription Length` is total months of subscription and is **not bounded by the nine-month window** — it reaches 47. It therefore measures tenure accrued before the window opened, which the other predictors do not. Section 5.5 records the consequence.

The sample base rate is **15.71%** (495 of 3,150). The UCI page describes the data as randomly collected from the operator's database and states no stratification, so — unlike South German Credit — the base rate is not known to be distorted by a sampling design. That is a statement about the absence of documented stratification, not a verified claim that the sample is representative of the operator's book. PR-AUC and precision levels remain base-rate dependent and only within-dataset model ordering transfers to the cross-dataset stage.

---

## 2. Source file and assertions [Stage 2]

| Item | Value |
|---|---|
| Filename | `data/Customer Churn.csv` |
| SHA-256 | `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1` |
| File size | 131,728 bytes |
| Delimiter | Comma, single header row, CRLF line endings |
| Expected rows | **3150** |
| Expected columns | **14** (13 predictors + target) |
| Expected positive count | **495** |
| Expected positive rate | **15.714%** |
| Missing values | **0** |

Assert rows, columns and positive count. Halt on mismatch.

**Reader note — header whitespace is irregular and must be normalised, not hand-typed.** Three headers contain a double space:

```
'Call  Failure', 'Subscription  Length', 'Charge  Amount'
```

The remaining eleven are single-spaced: `'Complains'`, `'Seconds of Use'`, `'Frequency of use'`, `'Frequency of SMS'`, `'Distinct Called Numbers'`, `'Age Group'`, `'Tariff Plan'`, `'Status'`, `'Age'`, `'Customer Value'`, `'Churn'`. Note also the lower-case *u* in `'Frequency of use'` against the upper-case *S* in `'Frequency of SMS'`.

Immediately after load, apply `df.columns = [' '.join(c.split()) for c in df.columns]` and then assert the resulting fourteen names against the canonical list in Section 3.2. Do not select columns by a hand-typed literal before this step.

Thirteen columns parse as `int64`; `Customer Value` parses as `float64`.

The `.csv` file is never modified. All rulings are applied inside the pipeline.

---

## 3. Rulings [Stage 3]

| # | Variable | Issue | Ruling | Verification |
|---|---|---|---|---|
| R1 | `Churn` | Target polarity | `y = Churn`; positive = churn. No inversion | `y.sum() == 495` |
| R2 | — | Missing values | None present. Assert zero and halt otherwise | `df.isna().sum().sum() == 0` |
| R3 | — | Disguised missing codes | None. No `?`, blank or sentinel appears. Zeros in the usage columns are **substantive**, not sentinels — see R11 | All observed values are numeric and non-negative; the documented-range discrepancy is handled separately at R6 |
| R4 | `Age` | **Exact functional duplicate of `Age Group`** | **Drop `Age` from all four design matrices.** See Section 3.3 | `Age`/`Age Group` crosstab is diagonal; `Age` absent from every design matrix |
| R5 | — | Identifier columns | **None present**, despite the UCI description block naming an "Anonymous Customer ID". No column dropped as an identifier | Column list asserted equal to the 14 expected names |
| R6 | `Charge Amount` | UCI documents an ordinal scale `0`–`9`; the file contains **code 10** (n=7), which is undocumented | Resolved by R7, which collapses it. **No separate ruling is applied to code 10 in isolation** | Assert `df['Charge Amount'].max() == 10` before R7 and `== 5` after |
| R7 | `Charge Amount` | Sparse upper levels: 5 (n=30), 6 (n=11), 7 (n=14), 8 (n=19), 9 (n=14), 10 (n=7) | **Collapse levels 5–10 into a single top level coded 5.** Applied to all four models | Resulting level counts: 1768 / 617 / 395 / 199 / 76 / 95 |
| R8 | — | Duplicate rows | 300 exact duplicates; 314 feature-identical duplicates; 476 rows sit in a repeated feature pattern. **Retain all. No de-duplication step.** See Section 5.6 | Both counts asserted as 300 and 314; row count after rulings still 3150 |
| R9 | — | Feature-identical rows with **conflicting** targets | 14 feature patterns, 64 rows. **Retain.** These are an irreducible-error floor, not a defect (Section 5.6) | Conflicting-pattern count asserted as 14 |
| R10 | All | Outliers | **None removed.** `Seconds of Use` reaches 17,090 and `Frequency of SMS` reaches 522; both are plausible nine-month aggregates | No outlier filter in any pipeline |
| R11 | **All six columns carrying an exact-zero mass** — see Section 3.0 | Large exact-zero mass | **Retain as zeros. Not imputed, not treated as missing, not flagged with an indicator variable.** A zero is the substantive value in every case | Zero counts asserted per Section 3.0 |
| R12 | — | Row order | No target concentration by position (Section 5.1). The split is shuffled and stratified regardless; row position is never a feature | `shuffle=True` asserted in the split |
| R13 | — | Negative values | None in any column. Unlike Default of Credit Card Clients, no signed-value ruling is required | `(df < 0).sum().sum() == 0` |
| R14 | *derived* | Mean call duration is undefined where `Frequency of use == 0` | **Set to `0.0` on those 154 rows.** Justified at Section 3.4 | 154 rows assert to exactly `0.0`; no `NaN` or `inf` in the column |
| R15 | — | Timing of each recode relative to the split | R4, R7, the binary remappings in Section 3.2 and the R14 construction are **deterministic row-wise operations** and are fold-safe. Standardisation and one-hot fitting are **learned** and sit inside the fold pipeline | No scaler or encoder is fitted outside a fold; assert per workflow Stage 7 |
| R16 | `Distinct Called Numbers`, `Frequency of use` | `Distinct Called Numbers > Frequency of use` in 120 rows, despite the source describing the latter as total calls | **Retain both fields unchanged.** Treat this as unresolved source-semantic ambiguity, not a correctable row error. Do not derive a ratio between these two fields | Assert 120 rows and record the limitation; no row removal or value correction |
| R17 | `Customer Value` | Exact deterministic composite of age band and three usage variables; source does not document the formula | **Drop from all four design matrices.** Retain only in raw audit outputs. See Sections 5.3 and 7 | Assert the empirical formula on all 3,150 rows to absolute tolerance `5e-10`, and assert absence from every design matrix |

### 3.0 Zero masses — the complete list [R11]

Six columns carry an exact-zero mass. **All six are ruled on here**, so that none of them reaches Stage 3 without a ruling.

| Column | Zeros | % | Part of the dormant block? |
|---|---|---|---|
| `Call Failure` | 702 | 22.29 | **No** |
| `Frequency of SMS` | 603 | 19.14 | **No** |
| `Seconds of Use` | 154 | 4.89 | Yes |
| `Frequency of use` | 154 | 4.89 | Yes |
| `Distinct Called Numbers` | 154 | 4.89 | Yes |
| `Customer Value` | 132 | 4.19 | Yes (the SMS-silent subset) |

Two distinct phenomena, and the file must not conflate them:

1. **Dormancy.** The three call columns zero out on exactly the same 154 rows, and `Customer Value = 0` on exactly the 132 of those that are also SMS-silent. This is the block described at Section 5.4, at a churn rate of 0.5260.
2. **Ordinary non-use of a service.** `Call Failure = 0` (702 rows) means a customer experienced no dropped calls, and `Frequency of SMS = 0` (603 rows) means a customer does not text. Neither implies dormancy: most of the 603 SMS-silent customers are active callers. These are common, unremarkable values.

All six retain their zeros untransformed under R11. No indicator variable is created for any of them, for the reason given at Section 5.4.

### 3.1 Leakage register [Stage 2 output]

**No variables are registered.** The UCI page states that every attribute other than `Churn` is aggregated over the first nine months and that the label is the customer's state at month twelve, with a three-month planning gap between them. All thirteen predictors are therefore available at the point the retention decision would be taken.

Two predictors were reviewed explicitly because their association with the target is strong enough to invite the question, and both are **cleared**:

- **`Status` (1 active, 2 non-active).** Churn rate is 47.31% among non-active customers against 5.28% among active ones. It is nonetheless an aggregate over months 1–9, and it is far from deterministic: 412 of the 782 non-active customers did not churn. It is an antecedent indicator of the same disengagement process the label records, not a measurement of the label.
- **`Complains` (0 none, 1 complaint registered).** Churn rate is 82.99% among the 241 complainants against 10.14% otherwise. Again a months 1–9 aggregate; 41 complainants did not churn.

**Binding interpretive constraint on both.** Stage 16 and Stage 17 outputs describe these two as *early-stage indicators of a disengagement process that the label subsequently records*, never as causes of churn and never as levers. The associational language required by workflow Stage 17 applies with particular force here.

`leakage_register.csv` is still produced with a header row and no entries, so the absence is a recorded result rather than a missing file. This matches South German Credit and preserves the schema.

### 3.2 Feature types and display names [Stage 3 output]

Raw column headers are never shown in a table or figure.

| Column (normalised) | Display name | Feature type | Treatment |
|---|---|---|---|
| `Call Failure` | Call failures | Quantitative | Numeric; standardised for logistic regression |
| `Complains` | Complaint registered | Binary categorical | Already 0/1; `complaint_registered = 1` when code is 1 |
| `Subscription Length` | Subscription length (months) | Quantitative | Numeric; standardised for logistic regression. See Section 5.5 |
| `Charge Amount` | Charge amount band | Ordinal | Collapsed to 0–5 (R7); retained as an ordered score |
| `Seconds of Use` | Total seconds of use | Quantitative | Tree family only (Section 7) |
| `Frequency of use` | Number of calls | Quantitative | Numeric; standardised for logistic regression |
| `Frequency of SMS` | Number of text messages | Quantitative | Numeric; standardised for logistic regression |
| `Distinct Called Numbers` | Distinct numbers called | Quantitative | Numeric; standardised for logistic regression |
| `Age Group` | Age band | Categorical, 5 levels | LR: drop-reference one-hot (Section 7); tree family: raw integer code |
| `Tariff Plan` | Contractual plan | Binary categorical | `contractual_plan = 1` when raw code is 2 (contractual); 0 when 1 (pay as you go) |
| `Status` | Account inactive | Binary categorical | `account_inactive = 1` when raw code is 2 (non-active); 0 when 1 (active) |
| `Age` | — | **Dropped (R4)** | Absent from every design matrix; no display name issued |
| `Customer Value` | Customer value score | Quantitative, exactly derived | **Dropped from all four model families (R17); retained in raw audit outputs only** |
| `Churn` | Churn (churn = positive) | Target (binary) | Positive class = churn (R1) |
| *derived* | Mean call duration (seconds) | Quantitative, derived | **Logistic regression only** (Section 7) |

**Taxonomy after R4 and R17: 6 quantitative, 1 ordinal, 3 binary and 1 five-level categorical — 11 model predictors** (13 source predictors less `Age` and `Customer Value`).

**Binary coding convention.** Each binary indicator uses the displayed category as the value `1`: complaint registered, contractual plan and account inactive. This is a semantic naming convention, not a common risk-direction convention. In particular, `contractual_plan = 1` is the lower-churn tariff category in this sample, while `account_inactive = 1` and `complaint_registered = 1` are the higher-churn categories. Coefficient and SHAP directions are interpreted against the displayed category.

### 3.3 Why `Age` is dropped [R4]

`Age` and `Age Group` are the same variable twice. The crosstab is exactly diagonal:

| `Age` | 15 | 25 | 30 | 45 | 55 |
|---|---|---|---|---|---|
| `Age Group` | 1 | 2 | 3 | 4 | 5 |
| n | 123 | 1037 | 1425 | 395 | 170 |

Spearman correlation between them is **1.000**. `Age` holds five distinct values and is a relabelling of the five bands with representative ages, not a measured age in years. Three further points support the drop:

1. The UCI Additional Information block lists the dataset's attributes and names *age group* but not *age*. `Age` appears only in the Variables Table.
2. Retaining both would place a perfectly collinear pair in the logistic regression design matrix.
3. Retaining both would split importance arbitrarily between two identical columns in every tree model, corrupting Stage 16 rankings for reasons that have nothing to do with the models.

`Age Group` is kept rather than `Age` because it is the documented variable and because its integer codes carry no false implication of a measurement in years.

**Consequence to record.** The Pearson correlation between the two is 0.9608 while the Spearman correlation is 1.000. Any collinearity screen run on Pearson alone would have rated this pair as merely high rather than as exact redundancy. That is a methodological observation worth one line in the write-up.

---

### 3.4 Why mean call duration is set to zero at a zero denominator [R14]

Mean call duration is `Seconds of Use / Frequency of use`, undefined on the 154 dormant rows. Setting it to `0.0` places dormant customers at the bottom of the scale, which conflates *made no calls* with *made very short calls*. The conflation was checked and is benign:

| Group | n | Churn rate |
|---|---|---|
| `Frequency of use == 0` (assigned 0.0) | 154 | **0.5260** |
| `Frequency of use > 0` and mean duration < 10 s | **6** | **0.5000** |

Only six customers occupy the genuine low end, and they sit at essentially the same risk as the dormant block, so the two groups are adjacent on the scale rather than contradictory. The minimum observed non-dormant value is 2.0 seconds and the median is 60.1, so the assignment extends an existing low tail rather than inventing a new region.

**Alternatives considered and rejected.** Fold-median imputation would make the feature a learned transform, forcing it inside the fold pipeline for no measurable gain over an assignment justified by the table above. A separate dormancy indicator is rejected for the reason given at Section 5.4. Both are logged.

---

## 4. Split and fold design [Stage 4]

| Item | Value |
|---|---|
| Split | Stratified 80/20 on `y` |
| Development set | 2,520 rows, 396 positive |
| Test set | 630 rows, 99 positive |
| Random seed | **42** (split, folds, search, bootstrap) |
| Outer folds | 5, stratified |
| Inner folds | 3, stratified |
| Shuffle | **True** |

These counts are verified: `train_test_split(test_size=0.2, stratify=y, random_state=42, shuffle=True)` returns exactly 2,520 / 630 with 396 / 99 positives. Assert them.

**Scale note for the write-up.** With 99 test positives this is the second-smallest test set of the four datasets, larger than South German Credit's 60 but far smaller than Bank Marketing's. The bootstrap interval on test PR-AUC will be visibly wide and the nested cross-validation estimate over 2,520 development rows remains the more stable quantity.

---

## 5. Audit expectations [Stages 2 and 5]

Computed from the source file before the run, so the pipeline's own output can be checked against an independent count.

### 5.1 Row order — no target structure

Churn rate by decile of row position ranges from **0.1429 to 0.1714**, and in blocks of one hundred rows from 0.14 to 0.18. Spearman correlation between row position and the target is **-0.014**.

This is the opposite of South German Credit, where the file is sorted on the target. The split is shuffled and stratified regardless, as the workflow requires. Stage 5 reports the flatness as an observed characteristic.

**One point to record rather than argue.** The churn rate is unusually stable across position — thirty-two consecutive blocks of one hundred rows all fall between 0.14 and 0.18. This is consistent with a randomly collected sample, as the UCI page states, and no further inference is drawn from it.

### 5.2 Category codes

`Complains`, `Tariff Plan` and `Status` contain only their two documented codes. `Age Group` contains only 1–5. The single documentation discrepancy is `Charge Amount` code 10 (R6), resolved by the collapse at R7.

### 5.3 Correlated blocks [determines Stage 7 handling]

After dropping `Age`, three pairs exceed |r| = 0.7 on Pearson:

| Pair | Pearson | Spearman |
|---|---|---|
| `Seconds of Use` ~ `Frequency of use` | **0.9465** | 0.9370 |
| `Frequency of SMS` ~ `Customer Value` | **0.9249** | 0.7796 |
| `Frequency of use` ~ `Distinct Called Numbers` | 0.7361 | 0.8242 |

Variance inflation factors on the full twelve-predictor matrix, standardised:

| Predictor | VIF |
|---|---|
| Customer Value | 52.69 |
| Frequency of SMS | 41.68 |
| Seconds of Use | 22.26 |
| Frequency of use | 18.81 |
| Charge Amount | 3.19 |
| all remaining | < 3.0 |

Condition number of the standardised matrix: **20.53**.

**Ruling: `Customer Value` is dropped from all four model families, and the logistic regression call-usage block is additionally restructured.** The tree family otherwise retains the remaining raw predictors. The mechanism is set out in Section 7. This is the same shape of problem as Default of Credit Card Clients, where a block at mean |r| ≈ 0.89 forced a collapse, and the opposite of South German Credit, where the maximum was 0.625 and nothing was done. Iranian Churn therefore sits with CCD on this axis, and the contrast across the three is itself material at the cross-dataset stage.

`Customer Value` is **exactly derived in the supplied CSV**, although the UCI page describes it only as "the calculated value of customer" and does not document the construction. For every row,

```
Customer Value = age_group_factor × (Seconds of Use + Frequency of use + 100 × Frequency of SMS)
```

where the factors for age groups 1–5 are respectively `0.055`, `0.045`, `0.040`, `0.025` and `0.015`. The maximum absolute floating-point residual is `4.55e-13` across all 3,150 rows, below the locked `5e-10` verification tolerance. It therefore supplies a pre-engineered age-by-usage interaction but no new underlying source information. R17 drops it from every model so the tree family is not uniquely handed that engineered interaction. The source-undocumented but empirically exact status is recorded without claiming that the recovered formula is the operator's documented business definition.

### 5.4 The dormant-account block

154 rows have `Seconds of Use = 0`. The three call-usage columns zero out on **exactly the same 154 rows** — the index sets are identical, not merely coincident in count. Of these, 132 also have `Frequency of SMS = 0`, and that subset is **exactly** the set with `Customer Value = 0`.

| Block | n | Churn rate |
|---|---|---|
| Zero call usage | 154 | **0.5260** |
| Zero call usage and zero SMS (`Customer Value = 0`) | 132 | **0.5227** |
| Whole dataset | 3150 | 0.1571 |

This block is 4.9% of the data at more than three times the base rate. R11 retains the zeros untransformed and creates no indicator. Stage 5 records the block as an observed structure with a pre-model hypothesis: the tree family can isolate it with a single split at or near zero on any of the three coincident columns, whereas logistic regression must approximate it through the lower tail of several standardised linear terms. Stage 17 revisits this descriptively.

**No dormancy indicator is engineered.** Constructing one would hand the linear model a feature discovered from the outcome and would remove precisely the contrast the pre-model hypothesis is there to test. The alternative was considered and rejected on that ground; the rejection is logged.

### 5.5 Non-monotone predictor–target relationships

Two predictors have relationships that a single linear term in the log-odds cannot represent. Both are recorded here **before** modelling, as Stage 5 pre-model hypotheses.

**`Subscription Length` — U-shaped.** Churn rate by tenure, in months:

| Months | 3–9 | 10–17 | 18–23 | 24–29 | 30–35 | 36–41 | 42–47 |
|---|---|---|---|---|---|---|---|
| Approx. churn rate | ~0.45 | ~0.10 | **0.00** | ~0.10 | ~0.22 | ~0.14 | ~0.09 |

178 rows in the 18–23 band contain **zero** churners. Overall Spearman with the target is only **-0.035**, which understates the relationship rather than describing its absence.

**`Age Group` — inverted-U.** Churn rate by band: 1 → **0.0000** (n=123), 2 → 0.1774, 3 → 0.1614, 4 → 0.2000, 5 → **0.0118** (n=170). Overall Spearman with the target is **-0.005**.

Both shapes are the reason for the encoding rule in Section 7. `Age Group` is one-hot encoded for logistic regression, which lets the linear model represent it; `Subscription Length` enters as a single linear term, which does not. The asymmetry is deliberate and is justified in Section 7.

**Quasi-separation to expect.** Several regions contain zero positives: `Age Group = 1` (123 rows), `Subscription Length` 18–23 (178 rows), `Charge Amount ≥ 5` after collapse (95 rows), `Seconds of Use > 7023` (626 rows), `Frequency of use > 109` (625 rows), `Frequency of SMS > 240` (628 rows). Tuned L2 controls the resulting coefficient magnitudes; the `Age Group = 1` dummy is the one to inspect at Stage 16, since some inner folds will contain roughly twenty such rows and no positives among them. Record its coefficient with that caveat attached, and do not read its odds ratio as an estimate.

### 5.6 Duplicate structure

| Quantity | Value |
|---|---|
| Distinct feature patterns | 2,836 |
| Exact duplicate rows (features and target) | 300 |
| Feature-identical duplicate rows | 314 |
| Rows sitting in a repeated feature pattern | **476 (15.1%)** |
| Feature patterns with conflicting targets | **14** |
| Rows in conflicting patterns | 64 |
| Largest pattern multiplicity | 11 |

**R8 and R9 retain all of them**, consistent with Default of Credit Card Clients, which retained 35 feature-identical duplicates.

The rows involved are not arbitrary. 77 of the 476 are dormant accounts, and duplicated rows are lower on every usage variable than unique rows — mean `Seconds of Use` 3,805 against 4,591. Customers with little activity are described by few distinguishing numbers and collide. Of the 14 conflicting patterns, **11** lie in the dormant block, covering **56 of the 64 rows**: identical observed predictor profiles, opposite outcomes.

**Two consequences, both recorded rather than engineered away.**

1. **The conflicting patterns place a ceiling on attainable performance.** No model can separate rows it cannot distinguish. Stage 17 should treat any observation that all four models plateau on the dormant block against this fact rather than against a model property.
2. **Duplicated patterns straddle the split.** Under the locked split, **100 of 630 test rows (15.87%)** have their exact feature pattern also present in the development set. Test metrics are therefore mildly optimistic in a way the bootstrap interval does not capture, because the bootstrap resamples the same test set.

A group-aware split would remove the second effect. It is **rejected**: the stratified 80/20 scheme is FIXED across all four datasets and changing it here would break the comparability the cross-dataset chapter depends on. The exposure is quantified above, carried into the recorded limitations, and stated in the results chapter. The alternative and the reason for rejecting it are logged.

---

## 6. Evaluation framework [Stage 6]

| Purpose | Metric |
|---|---|
| Primary model selection | PR-AUC (average precision) |
| Discrimination | ROC-AUC |
| Probability quality | Calibration slope and intercept, Brier score, log loss |
| Positive-class behaviour | Precision and recall at the declared operating point |
| Error profile | Confusion matrix, FP and FN counts |

**Positive class:** churn. **Base rate:** 15.714%. A prevalence/dummy baseline is recorded alongside model PR-AUC. If a model's PR-AUC is at or below the matched baseline, the pipeline raises a prominent warning, runs target-polarity and prediction-score diagnostics, and continues. The result is investigated and documented rather than suppressed. The pipeline halts only for an invalid metric, failed polarity assertion or other data/prediction integrity failure.

**Domain description of errors**, used verbatim in written outputs:

- **False positive** — a customer who stayed, targeted with a retention offer.
- **False negative** — a customer who churned, not targeted.

**Class imbalance is handled through metric choice and operating-point selection only.** No class weighting, no resampling, no loss modification.

---

## 7. Model-specific preprocessing [Stage 7]

The four models do not receive identical feature matrices. This table is authoritative; `feature_sets_by_model.csv` must reproduce it.

### 7.1 The encoding rule for this dataset

Stated once, because two decisions below follow from it and both would otherwise look arbitrary:

> A variable enters logistic regression as a single ordered score only where an ordering is documented **and** the observed relationship with the target is monotone. An ordered score encodes an assumption of monotonicity; where the audit contradicts that assumption, the assumption is not imposed. Categorical variables without a documented ordering are one-hot encoded. Quantitative measurements enter as single linear terms and are not binned, splined or otherwise given extra flexibility.

Under this rule:

- **`Charge Amount`** — documented ordering, and observed churn falls monotonically across the collapsed bands (0.2381, 0.0697, 0.0582, 0.0352, 0.0132, 0.0000). **Ordered score.**
- **`Age Group`** — documented as ordinal, but the observed relationship is decisively non-monotone (Section 5.5). The monotonicity assumption is not imposed. **One-hot, drop-reference.**
- **`Subscription Length`** — a quantitative measurement in months, not a band. It stays a **single linear term** despite being U-shaped.

**The `Age Group` / `Subscription Length` asymmetry is deliberate and must be defended in the methodology chapter in these terms.** A five-level demographic band is one-hot encoded in any practical scorecard, and forcing it into a linear score would manufacture a failure attributable to our encoding rather than to the model family. A continuous tenure variable is not binned, because choosing cut-points after seeing the churn curve would hand the linear model an outcome-derived nonlinearity. The line is between *how a categorical variable is represented*, which is a routine specification choice, and *adding flexibility to a continuous term*, which is not. `Subscription Length` is consequently the cleanest place in this dissertation to observe a linear model failing on structure the tree family can represent, and Stage 17 should look there first.

**This diverges from South German Credit**, which entered all nine ordinal variables as ordered scores. The reason given there was parameter economy — 240 development positives could not support one-hot encoding nine variables. That constraint does not bind here: 396 development positives, and one-hot encoding `Age Group` costs four parameters. Logged in the deviations section.

### 7.2 Logistic regression

| Feature group | Treatment |
|---|---|
| `Call Failure`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`, `Subscription Length` | Numeric, standardised |
| **Mean call duration** (derived) | `Seconds of Use / Frequency of use`, set to `0.0` where `Frequency of use == 0`; standardised |
| `Charge Amount` | Ordered score 0–5 after R7, standardised |
| `Age Group` | One-hot with **`categories` declared explicitly as `[1, 2, 3, 4, 5]`**, reference level dropped |
| `Complains`, `Tariff Plan`, `Status` | Explicit 0/1 indicators using the mappings in Section 3.2 |
| `Seconds of Use` | **Excluded** — collapsed into mean call duration |
| `Customer Value` | **Excluded** — see below |
| Regularisation | L2, strength `C` tuned in the inner loop |

**The correlated-block collapse.** `Seconds of Use` and `Frequency of use` (r = 0.9465) are replaced by `Frequency of use` and mean call duration, separating call **volume** from call **intensity**. `Customer Value` (VIF 52.69 and r = 0.9249 with `Frequency of SMS`) is dropped under R17 because it is an exact deterministic composite of age group and usage variables already present.

Verified effect on the eleven-term numeric block: **every VIF falls below 3.0** (highest: `Frequency of use` at 2.94), and the condition number falls from **20.53 to 4.10**. Mean call duration is close to orthogonal to `Frequency of use` (r = 0.085) and carries its own association with the target (Spearman -0.123), so the collapse is not a relabelling of an existing column.

**Alternative considered and rejected.** Simply dropping `Seconds of Use` without constructing mean call duration achieves a comparable condition number (4.00). It is rejected because it discards the volume/intensity distinction entirely, leaving the logistic regression unable to express a pattern the tree family retains in full — an avoidable handicap, unlike the `Subscription Length` case, where the handicap is the object of study. Logged.

**Parameter count: 14** (6 quantitative including the derived mean-duration term, 1 ordinal score, 3 binary indicators and 4 `Age Group` dummies), plus intercept.

**Locked reference level for `Age Group`: band 3**, the modal level (n = 1,425). It is the most frequent observed level, consistent with the South German Credit rule, and must be recorded in the coefficient-table metadata. Changing it is a specification change, not a tuning choice.

**Why `categories` is declared explicitly rather than inferred.** All five bands are present with at least 123 rows across 3,150, so every stratified fold of a 2,520-row development set contains all five and no unknown level can arise. Declaring the category list fixes both the column order and the identity of the dropped reference across folds, which matters because the band 1 coefficient is the one carrying the quasi-separation caveat and must be traceable to the same dummy in every fold. It also removes any dependence on how the installed scikit-learn version handles `drop` in combination with `handle_unknown`. South German Credit ran with `handle_unknown='ignore'` alongside a dropped reference, so that combination evidently works in this environment; the explicit list is robustness, not a correction, and Codex should confirm the encoder signature against the installed version rather than assume it.

**Events per parameter: 28.3** across the development set, **22.6** in an outer-fold training set, and roughly **15.1** in an inner-fold training set. This is comfortable, and markedly better than South German Credit's 3.9. Coefficient instability at Stage 16, if observed, is therefore attributable to the quasi-separation in Section 5.5 rather than to thin events, and should be reported that way.

**No log or signed-log transform is applied to any predictor.** Log1p was evaluated and rejected on measured skew: it *worsens* the skew of four of the six quantitative predictors, because the zero mass documented at Section 5.4 becomes a left-hand spike. `Seconds of Use` moves from +1.322 to -2.650, `Frequency of use` from +1.144 to -1.443, `Distinct Called Numbers` from +1.029 to -1.231, `Subscription Length` from -1.300 to -2.400. It improves `Frequency of SMS` (+1.974 to -0.188) and `Call Failure` (+1.090 to -0.445). A transform applied to some columns and not others on a per-variable skew test is not a coherent specification, so none is applied. This diverges from both South German Credit (`log1p` on credit amount) and Bank Marketing (signed-log on balance), and the divergence is a property of the zero-inflation, not an inconsistency. Logged.

**Scaling is standardisation**, matching South German Credit rather than the robust scaling used for the Default of Credit Card Clients utilisation feature. Recorded as a limitation: mean call duration has skew 3.79 and a maximum of 628 seconds, so its standardised scale is influenced by a small number of high values.

### 7.3 Decision tree, random forest, XGBoost

| Feature group | Treatment |
|---|---|
| `Call Failure`, `Subscription Length`, `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers` | Raw values, no scaling, no transformation |
| `Customer Value` | **Excluded from all tree-family matrices under R17** |
| `Charge Amount` | Raw ordered code 0–5 after R7 |
| `Age Group` | Raw integer code 1–5 |
| `Complains`, `Tariff Plan`, `Status` | Explicit 0/1 indicators using the mappings in Section 3.2 |
| Correlated blocks | **No additional collapse.** All remaining original columns are retained after R4 and R17 |
| Mean call duration | **Not constructed.** The tree family can represent the ratio through splits on its components |

**The tree-family design matrix contains 11 terms and requires no one-hot encoding at all** — `Age Group` is the only multi-level categorical and enters as an ordered integer code, which permits the non-monotone structure in Section 5.5 to be captured through repeated splits.

This is the first of the four datasets on which the tree-family matrix needs no expansion, so encoded-term level and original-variable level coincide for those three models. Stage 16 reduction is required for logistic regression only.

### 7.4 Feature-set asymmetry — required record for Stage 16

Two features are not shared across model families, and `feature_rankings_combined.csv` must handle them explicitly rather than by writing a zero. `Customer Value` is absent from every model and therefore does not appear as a ranking row:

| Feature | LR | DT / RF / XGB | Rule for the combined rankings file |
|---|---|---|---|
| Total seconds of use | absent | present | LR rank and direction columns **blank**; reason recorded |
| Mean call duration | present | absent | Tree-family rank and direction columns **blank**; reason recorded |

A blank means *this model never received this feature*. It must not be filled with a zero, a bottom rank, or an imputed value, since any of those would read as *this model found the feature uninformative*, which is a different and false statement. `top10_overlap_*` counts are computed over each pair's shared features and the denominator is recorded alongside the count.

---

## 9. Hyperparameter search [Stage 9]

Randomised search, `n_iter = 40`, seed 42, identical budget across all four models, scored on PR-AUC in the inner loop.

| Model | Search space |
|---|---|
| Logistic regression | `C`: log-uniform 1e-3 to 1e2; `penalty`: l2; `solver`: lbfgs; `max_iter`: 2000 |
| Decision tree | `max_depth`: 2–**5** (capped); `min_samples_leaf`: 5–60; `min_samples_split`: 10–120; `criterion`: gini, entropy; `ccp_alpha`: 0.0–0.02 |
| Random forest | `n_estimators`: 300–800; `max_depth`: 3–12 or None; `min_samples_leaf`: 1–25; `max_features`: sqrt, log2, 0.5 |
| XGBoost | `n_estimators`: 200–600; `max_depth`: 2–6; `learning_rate`: log-uniform 0.01–0.3; `subsample`: 0.6–1.0; `colsample_bytree`: 0.6–1.0; `min_child_weight`: 1–10; `reg_lambda`: log-uniform 0.1–10 |

Leaf-size ranges are scaled to the 2,520-row development set, sitting between the South German Credit ranges (800 rows) and the Bank Marketing ranges (36,168 rows).

**Decision tree maximum depth is capped at 5**, matching South German Credit and Bank Marketing. The search may not exceed it; realised depth is asserted `<= 5` in the invariant checks. The cap is a specification decision, not a tuning outcome. Record that the ensembles were not similarly constrained.

**Both the inner-loop search score and the nested outer estimate are recorded for every model**, so any divergence between them is visible. `search_score` is defined in workflow Stage 9. The `final_fit_search_score` and `final_fit_params_json` required by workflow Stage 9 are recorded separately.

---

## 12. Threshold selection [Stage 12]

**Rule: floor recall at 0.80, maximise precision subject to that floor.**

Applied identically to all four models, on out-of-fold development predictions, fixed before the sweep is inspected.

**Deterministic tie-break:** among eligible thresholds, choose maximum precision; if tied, choose maximum recall; if still tied, choose the highest threshold.

**Why recall is floored.** The false negative — a customer lost — costs the remaining lifetime value plus reacquisition. The false positive costs one retention offer to a customer who would have stayed. The asymmetry runs strongly towards coverage, so the operating point guarantees coverage of the at-risk base and lets precision follow. Flooring recall is also mechanically safe: recall is monotone in the threshold, so every model can attain any recall floor, the Stage 12 attainment halt cannot fire spuriously, and **workflow amendment 7 does not bind on this dataset**. `floor_attained` is still written to `thresholds_selected.csv` for schema consistency with Bank Marketing and is expected to be `True` for all four models.

**Why 0.80.** A declared analytical anchor, fixing coverage at four in five churners before precision is considered. It is **not derived from a verified economic cost** and must be described that way.

**Why it is higher than South German Credit's 0.75.** Deliberate, and a substantive point for the discussion chapter rather than an inconsistency. Refusing a creditworthy applicant denies them access to credit and forgoes the bank's interest; sending an unnecessary retention offer costs a discount. The false-positive cost is materially lower here, so the recall floor is set higher. **Operating points are dataset-specific by design; only within-dataset model ordering transfers to the cross-dataset stage.**

**Also recorded:**

- Precision, recall, FP and FN at the conventional `0.50` threshold, **as a reference convention only**.
- **No cost-ratio sensitivity analysis.** No cost ratio is stipulated for this dataset and none may appear in calculations, tables, figures or written outputs.

**One consequence to record rather than argue.** At a 15.7% base rate, an 0.80 recall floor sits well out on the sweep, where precision is falling. Precision at the operating point may therefore be low in absolute terms for every model, and the spread between models may be compressed relative to the full PR curve. If the Stage 15 ranking under precision-at-threshold disagrees with the ranking under PR-AUC, check the threshold sweep files before offering any other reading. Stage 14 takes precedence in any case.

---

## 16. Interpretability evidence [Stage 16]

The governing rules — development-set data source, signed-summation aggregation to variable level, and no variable-level aggregate for logistic regression coefficients — are set out in workflow Stage 16 and apply to every dataset. Four dataset-specific points follow.

**Scale.** Logistic regression is fitted on 14 encoded terms and the tree family on 11, with no encoded expansion at all for the tree family (Section 7.3). Reduction to the original-variable level therefore applies to **`Age Group` in the logistic regression only**. This is the simplest interpretability configuration of the four datasets, and it makes Iranian Churn the cleanest case for reading the four models against each other — worth stating in the cross-dataset chapter.

**Feature-set asymmetry.** The blank-not-zero rule in Section 7.4 is binding on `feature_rankings_combined.csv` and is the single most likely place for a silent error on this dataset.

**Development-set cap.** The development set holds 2,520 rows against the workflow's 2,000-row SHAP cap, so stratified subsampling to 2,000 with seed 42 **is** triggered here. Record the actual row count used, as the workflow requires. The background sample is `min(200, 2520) = 200` rows.

**Two features to inspect explicitly.** The `Age Group = 1` dummy has zero positives in 101 development rows and its logistic regression coefficient must not be read as an estimate (Section 5.5). And `Status` and `Complains` are the two strongest marginal predictors; whatever the four models rank first, the interpretive constraint in Section 3.1 governs how it is described.

---

## 18. Package assembly — one waiver [Stage 18]

**Item 6 of workflow Stage 18, the exact-reproduction re-run into a separate verification directory, is not performed on this dataset.**

This applies the standing cross-dataset ruling that waived the reproducibility re-run for every dataset after South German Credit, and matches Default of Credit Card Clients and Bank Marketing. Items 1 to 5 of Stage 18 are performed in full, so the registry, the expected-key set, the manifest check, the invariant suite, the schema comparison and the environment record are all produced as normal.

`reproducibility.json` is still written, recording that the re-run was waived by this ruling rather than attempted and failed. The absence is a declared decision, not a missing output.

---

## Recorded limitations

1. **Duplicated patterns straddle the split.** 100 of 630 test rows (15.87%) have their exact feature pattern present in the development set. Test metrics are mildly optimistic, and the bootstrap interval does not capture this because it resamples the same test set (Section 5.6).
2. **An irreducible-error floor.** 14 feature patterns, covering 64 rows, carry conflicting targets. No model can separate them; 11 of the 14 patterns, covering 56 rows, sit in the dormant-account block (Section 5.6).
3. **Small test set.** 99 positives (Section 4).
4. **`Customer Value` is source-undocumented but empirically exact.** The UCI page gives no formula, but every supplied row exactly follows the recovered age-group-weighted usage formula in Section 5.3. It is dropped from all four model families under R17. The recovered equality is a property of this file, not a claim that the operator documented or intended that formula.
5. **`Age Group` bands are of unknown width.** The representative ages 15, 25, 30, 45, 55 imply uneven and unstated band boundaries. Band ordering is used; no band is treated as a measured age.
6. **Age 15 appears in the data.** 123 customers sit in band 1. Whether these are minors holding accounts in their own name, or a coding convention for the youngest band, is not stated by the source. No inference is drawn from the band beyond its association with the target.
7. **Quasi-separation in several regions.** Six modelled regions contain zero positives (Section 5.5). Tuned L2 controls coefficient magnitude but the affected odds ratios are not estimates.
8. **Single operator, single country, 2020 donation.** Churn behaviour is specific to one Iranian telecommunications operator over one twelve-month window. Nothing generalises to other operators or markets, and the dissertation must not imply otherwise.
9. **No monetary units anywhere.** `Charge Amount` is a band with no stated currency amount and `Customer Value` has no stated unit. No output may express a cost, a saving or a return in currency.
10. **Heavy-tailed derived feature.** Mean call duration has skew 3.79 and a maximum of 628 seconds; standardisation leaves its scale sensitive to a small number of high values (Section 7.2).
11. **Mean call duration carries two meanings at zero.** 154 dormant rows are assigned `0.0` alongside a genuine low tail. The assignment is justified at Section 3.4 and the affected rows are a known, counted group, but any Stage 16 statement about the low end of this feature must acknowledge that it mixes *no calls made* with *very short calls*.
12. **Call-count semantics are internally unresolved.** `Distinct Called Numbers` exceeds `Frequency of use` in 120 rows (3.81%). The source definitions imply that this should not occur if both fields count calls in the ordinary sense, but no authoritative correction is available. R16 retains the values unchanged, prohibits a ratio between the fields and records the ambiguity.

---

## Deviations to log

**From the workflow text.** Stage 7 instructs "bin ordinal codes into interpretable buckets" for logistic regression. This file one-hot encodes `Age Group` and enters `Charge Amount` as an ordered score, under the rule in Section 7.1. Recorded as a deliberate, dataset-specific divergence.

**From South German Credit — ordinal encoding.** SGC entered all nine ordinal variables as ordered scores for logistic regression, on grounds of parameter economy. That constraint does not bind here, and `Age Group` is one-hot encoded because the observed relationship is non-monotone (Sections 5.5, 7.1).

**From South German Credit and Bank Marketing — skew transforms.** SGC applied `log1p` to credit amount; Bank Marketing applied a signed-log to balance. **No transform is applied on this dataset.** Log1p worsens measured skew on four of six quantitative predictors because of the zero mass at Section 5.4 (Section 7.2).

**From Default of Credit Card Clients — scaling of a derived feature.** CCD robust-scaled its utilisation feature. Mean call duration is standardised here, for consistency with the rest of the logistic regression numeric block. Recorded as a limitation rather than defended as superior.

**From the UCI documentation — `Charge Amount` range.** The page documents `0`–`9`; the file contains a code 10. Resolved by the collapse at R7 (R6).

**From the UCI documentation — an attribute not in the description block.** `Age` appears in the Variables Table but not in the Additional Information block, and is an exact functional duplicate of `Age Group`. Dropped at R4 (Section 3.3).

**From the UCI documentation — a column named but absent.** The description block names an "Anonymous Customer ID". No such column is in the file (R5).

**From the source-undocumented derived field.** `Customer Value` is exactly recoverable from age group and three usage variables in this CSV but the source publishes no formula. It is dropped from all four model families under R17 so that no family uniquely receives the pre-engineered interaction.

**From the source variable semantics.** `Distinct Called Numbers` exceeds `Frequency of use` in 120 rows. The values are retained unchanged under R16 because the source offers no authoritative resolution.

---

## Verification log

Computed directly from `data/Customer Churn.csv` before this file was written:

- Shape 3150 × 14; thirteen `int64` columns and one `float64`; zero missing values; zero negative values
- Header whitespace irregularity confirmed in three columns (Section 2)
- `Churn`: 495 ones, 2655 zeros; base rate 0.157143
- SHA-256 and file size of the source file
- `Age` × `Age Group` crosstab exactly diagonal; Spearman 1.000, Pearson 0.9608 (Section 3.3)
- 300 exact duplicate rows; 314 feature-identical duplicate rows; 2,836 distinct feature patterns; 476 rows in repeated patterns; 14 conflicting patterns over 64 rows; 11 conflicting patterns covering 56 rows in the dormant block (Section 5.6)
- Split at seed 42 reproduced: 2,520 / 630 rows, 396 / 99 positives; 100 test rows sharing a feature pattern with development (Sections 4, 5.6)
- Target rate by position decile and by blocks of 100; Spearman with position -0.014 (Section 5.1)
- Pearson and Spearman correlation matrices over all predictors; VIF and condition number before and after the Section 7.2 restructuring (Sections 5.3, 7.2)
- Exact `Customer Value` reconstruction on all 3,150 rows using the age-group factors and usage formula in Section 5.3; maximum absolute floating-point residual `4.55e-13`, below tolerance `5e-10`
- Zero-usage index sets confirmed identical across the three call columns, and the `Customer Value = 0` set confirmed equal to their intersection with zero SMS (Section 5.4)
- Zero counts for all six columns carrying a zero mass, and confirmation that the `Call Failure` and `Frequency of SMS` zero sets are distinct from the dormant block (Section 3.0)
- Distribution of mean call duration at a non-zero denominator — minimum 2.0 s, median 60.1 s, six rows below 10 s at a churn rate of 0.500 (Section 3.4)
- `Distinct Called Numbers > Frequency of use` in 120 rows (3.81%); retained under R16 as source-semantic ambiguity
- Absent integers within the observed range of each count variable: none for `Call Failure` (0–36) or `Subscription Length` (3–47); 14 for `Frequency of use`; 6 for `Distinct Called Numbers`; 118 for `Frequency of SMS`. All are sparse-tail gaps, not coding gaps
- Decimal precision of `Customer Value` is irregular — 562 values at one decimal place, 1,791 at two, 797 at three — supporting recorded limitation 4
- Churn rate by decile of every quantitative predictor and by level of every categorical predictor (Sections 5.5, 7.1)
- Skew of every quantitative predictor, raw and under `log1p` (Section 7.2)
- Level counts for `Charge Amount` before and after the R7 collapse
