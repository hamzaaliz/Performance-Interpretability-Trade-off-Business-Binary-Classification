# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss, and confusion-matrix counts.
- **Positive class:** Bad credit.
- **Operating rule:** Maximise precision subject to recall of at least 0.75; ties use maximum recall and then the highest threshold.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Below-baseline handling:** A valid PR-AUC at or below the matched prevalence/dummy baseline is warned about and investigated, not suppressed.
- **Sampling caveat:** PR-AUC, precision and calibration refer to the 30% positive-rate sample and are not population estimates.
