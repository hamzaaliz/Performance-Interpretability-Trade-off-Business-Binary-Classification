# Evaluation framework

- **Primary model-selection metric:** PR-AUC computed as average precision.
- **Supporting positive-class metrics:** precision and recall at the declared operating threshold.
- **Additional recorded metrics:** ROC-AUC, calibration slope and intercept, Brier score, log loss and confusion-matrix counts.
- **Positive class:** Customer churn.
- **Operating rule:** Require recall of at least 0.80, maximise precision, and break ties by recall and then the highest threshold. The guaranteed-floor fallback is labelled explicitly if ever invoked.
- **Imbalance handling:** Metric and threshold selection only; no class weighting, resampling or loss modification.
- **False positive:** A customer who stayed, targeted with a retention offer.
- **False negative:** A customer who churned, not targeted.
- **Test status:** The test set is evaluated once after the development-only gate passes.
