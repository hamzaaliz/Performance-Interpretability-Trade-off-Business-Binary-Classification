# Structural observations

These statements report measured development-set evidence. They do not explain performance differences, recommend a model, or answer a research question.

- The development evidence contains 2520 observations and 396 positive outcomes.
- **Logistic regression:** the highest cross-fitted grouped permutation rank was Complaint registered (mean average-precision decrease 0.305298); the highest variable-level SHAP rank was Number of calls (mean absolute contribution 1.297296, scale: log-odds (linear model raw output)).
- **Decision tree:** the highest cross-fitted grouped permutation rank was Complaint registered (mean average-precision decrease 0.325858); the highest variable-level SHAP rank was Account inactive (mean absolute contribution 0.130363, scale: raw class-1 probability output).
- **Random forest:** the highest cross-fitted grouped permutation rank was Complaint registered (mean average-precision decrease 0.255513); the highest variable-level SHAP rank was Account inactive (mean absolute contribution 0.108533, scale: raw class-1 probability output).
- **XGBoost:** the highest cross-fitted grouped permutation rank was Complaint registered (mean average-precision decrease 0.170590); the highest variable-level SHAP rank was Number of calls (mean absolute contribution 1.465548, scale: raw margin (log-odds)).
- The final decision tree had realised depth 5 and 7 leaves; its full rules are recorded in `outputs/interpretability/dt_structure.txt`.
- Pairwise permutation top-ten overlap counts were: LR–DT 10/10 shared-feature slots; LR–RF 9/10 shared-feature slots; LR–XGB 9/10 shared-feature slots; DT–RF 9/10 shared-feature slots; DT–XGB 9/10 shared-feature slots; RF–XGB 10/10 shared-feature slots.
- Nominal-variable SHAP directions are recorded as category-dependent; no single direction was assigned.
- Quantitative, ordinal and binary directions use the locked Spearman-rho convention and are recorded in each model's SHAP summary.
- The Stage 5 statements remain pre-model hypotheses; this file records no claim that a structural pattern caused a metric difference.
