# Structural observations

These statements report measured development-set evidence. They do not explain performance differences, recommend a model or answer a research question.

- The development evidence contains 24,000 accounts and 5,309 default outcomes.
- **Logistic regression:** the highest cross-fitted permutation rank was Repayment status, September (mean average-precision decrease 0.189693); the highest variable-level SHAP rank was Repayment status, September (mean absolute contribution 0.417332, direction positive, scale: log-odds (linear model raw output)). The top three variables account for 0.8617 of summed positive permutation importance.
- **Decision tree:** the highest cross-fitted permutation rank was Repayment status, September (mean average-precision decrease 0.169194); the highest variable-level SHAP rank was Repayment status, September (mean absolute contribution 0.080545, direction positive, scale: raw class-1 probability output). The top three variables account for 0.8887 of summed positive permutation importance.
- **Random forest:** the highest cross-fitted permutation rank was Repayment status, September (mean average-precision decrease 0.203731); the highest variable-level SHAP rank was Repayment status, September (mean absolute contribution 0.068242, direction positive, scale: raw class-1 probability output). The top three variables account for 0.8397 of summed positive permutation importance.
- **XGBoost:** the highest cross-fitted permutation rank was Repayment status, September (mean average-precision decrease 0.186452); the highest variable-level SHAP rank was Repayment status, September (mean absolute contribution 0.344815, direction positive, scale: raw margin (log-odds)). The top three variables account for 0.8162 of summed positive permutation importance.
- The final decision tree had realised depth 4 and 9 leaves. Stability across resamples was not measured, so stable thresholds are not asserted; the full fitted rules are recorded in the structure file.
- Pairwise permutation top-ten overlap counts were: LR–DT 7/10; LR–RF 8/10; LR–XGB 8/10; DT–RF 8/10; DT–XGB 8/10; RF–XGB 8/10.
- **Decision tree:** the highest-ranked older bill omitted from LR was Bill amount, August (NT$) at permutation rank 14. This records model use of an available historical input; it does not establish that the input caused a score difference.
- **Random forest:** the highest-ranked older bill omitted from LR was Bill amount, April (NT$) at permutation rank 15. This records model use of an available historical input; it does not establish that the input caused a score difference.
- **XGBoost:** the highest-ranked older bill omitted from LR was Bill amount, June (NT$) at permutation rank 16. This records model use of an available historical input; it does not establish that the input caused a score difference.
- Nominal-variable SHAP directions are category-dependent; no single direction is assigned.
- Quantitative, ordinal and binary directions use the locked Spearman-rho convention in each model's SHAP summary.
- The zero-recorded segment, bill trajectory and bill-to-limit curvature remain pre-model hypotheses. Standard EDA and importance evidence do not by themselves isolate a causal performance explanation.
- The 5,309 development defaults provide substantial positive-class support for the fitted rankings; this is an observed count, not an explanation for any model ordering.
