# Development-set EDA findings

All statements below describe observed development or source structure. Hypotheses are labelled and were recorded before model results were examined.

- The development set contains 240 bad-credit outcomes among 800 observations (0.300).
- The largest absolute pairwise Spearman correlation is 0.626, between **Credit duration (months)** and **Credit amount (DM)** (signed value 0.626). The locked decision retains both variables for logistic regression.
- Across source rows 751–1000, the bad-credit rate is 0.996. Row position is excluded from every feature matrix and the split is shuffled and stratified.
- Credit amount ranges from 250 to 18424; age ranges from 19 to 75. No observations were removed.
- **Pre-model hypothesis:** ordered-score predictors may provide monotonic signal suitable for logistic regression, while quantitative quintile-rate changes may also permit tree thresholds.
- **Pre-model hypothesis:** sparse nominal levels may yield unstable encoded-term coefficients; they are retained and flagged rather than collapsed.
- **Pre-model hypothesis:** the depth-five decision tree may express fewer distinct structures than the two ensembles; this is a model constraint, not a result.

Quantitative positive-rate sequences by development-set quintile were recorded as: Credit duration [0.21107266435986158, 0.2033898305084746, 0.32209737827715357, 0.2926829268292683, 0.4791666666666667]; Credit amount [0.30625, 0.25625, 0.25, 0.26875, 0.41875]; Age [0.3978494623655914, 0.3269230769230769, 0.23870967741935484, 0.25, 0.2641509433962264].
