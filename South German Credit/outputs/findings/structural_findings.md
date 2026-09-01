# Structural observations

These statements report measured development-set evidence. They do not explain performance differences, recommend a model, or answer a research question.

- The development evidence contains 800 observations and 240 positive outcomes.
- **Logistic regression:** the highest cross-fitted grouped permutation rank was Checking account status (mean average-precision decrease 0.111656); the highest variable-level SHAP rank was Checking account status (mean absolute contribution 0.607008, scale: log-odds (linear model raw output)).
- **Decision tree:** the highest cross-fitted grouped permutation rank was Checking account status (mean average-precision decrease 0.097966); the highest variable-level SHAP rank was Checking account status (mean absolute contribution 0.134659, scale: raw class-1 probability output).
- **Random forest:** the highest cross-fitted grouped permutation rank was Checking account status (mean average-precision decrease 0.119936); the highest variable-level SHAP rank was Checking account status (mean absolute contribution 0.079654, scale: raw class-1 probability output).
- **XGBoost:** the highest cross-fitted grouped permutation rank was Checking account status (mean average-precision decrease 0.120469); the highest variable-level SHAP rank was Checking account status (mean absolute contribution 0.664653, scale: raw margin (log-odds)).
- The final decision tree had realised depth 5 and 14 leaves; its full rules are recorded in `outputs/interpretability/dt_structure.txt`.
- Pairwise permutation top-ten overlap counts were: LR–DT 8/10; LR–RF 6/10; LR–XGB 8/10; DT–RF 7/10; DT–XGB 9/10; RF–XGB 7/10.
- Nominal-variable SHAP directions are recorded as category-dependent; no single direction was assigned.
- Quantitative, ordinal and binary directions use the locked Spearman-rho convention and are recorded in each model's SHAP summary.
- The Stage 5 statements remain pre-model hypotheses; this file records no claim that a structural pattern caused a metric difference.
