# Structural observations

These statements report measured development-set evidence. They do not explain performance differences, recommend a model, or answer a research question.

- The development evidence contains 36168 observations and 4231 positive outcomes.
- **Logistic regression:** the highest cross-fitted grouped permutation rank was Previous campaign outcome (mean average-precision decrease 0.100093); the highest variable-level SHAP rank was Contact channel (mean absolute contribution 0.567999, scale: log-odds (linear model raw output)).
- **Decision tree:** the highest cross-fitted grouped permutation rank was Previous campaign outcome (mean average-precision decrease 0.108801); the highest variable-level SHAP rank was Previous campaign outcome (mean absolute contribution 0.066836, scale: raw class-1 probability output).
- **Random forest:** the highest cross-fitted grouped permutation rank was Previous campaign outcome (mean average-precision decrease 0.124651); the highest variable-level SHAP rank was Previous campaign outcome (mean absolute contribution 0.043094, scale: raw class-1 probability output).
- **XGBoost:** the highest cross-fitted grouped permutation rank was Month of planned contact (mean average-precision decrease 0.098099); the highest variable-level SHAP rank was Contact channel (mean absolute contribution 0.514049, scale: raw margin (log-odds)).
- The final decision tree had realised depth 4 and 5 leaves; its full rules are recorded in `outputs/interpretability/dt_structure.txt`.
- Pairwise permutation top-ten overlap counts were: LR–DT 6/10; LR–RF 8/10; LR–XGB 8/10; DT–RF 7/10; DT–XGB 8/10; RF–XGB 9/10.
- Nominal-variable SHAP directions are recorded as category-dependent; no single direction was assigned.
- Quantitative, ordinal and binary directions use the locked Spearman-rho convention and are recorded in each model's SHAP summary.
- The Stage 5 statements remain pre-model hypotheses; this file records no claim that a structural pattern caused a metric difference.
