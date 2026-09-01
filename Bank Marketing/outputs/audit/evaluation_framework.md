# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Term-deposit subscription.
- **Operating rule:** Floor precision at 0.50 and maximise recall; constrained ties use precision and then highest threshold. Non-attainment uses the locked maximum-precision fallback.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **Sampling caveat:** PR-AUC, precision and calibration describe the pooled, date-ordered delivered sample and are not forward-campaign estimates.
- **Test status:** The test set is a one-time in-period held-out confirmation; development evidence governs comparisons.
