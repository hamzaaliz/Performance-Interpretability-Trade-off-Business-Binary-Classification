# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Default next month.
- **Operating rule:** Maximise precision subject to recall of at least 0.65; ties use maximum recall and then the highest threshold.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Below-baseline handling:** A valid PR-AUC at or below the matched prevalence/dummy baseline is warned about and investigated, not suppressed.
- **Sampling caveat:** PR-AUC, precision and calibration describe the delivered sample. Its sampling design is undocumented, so no population-prevalence claim is made.
- **Test status:** The test set is a one-time held-out confirmation under a pre-inspected public dataset. Development evidence governs comparative claims.
