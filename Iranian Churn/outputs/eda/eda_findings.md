# Development-set EDA findings

All predictor–target statements below use development data only. Hypotheses were recorded before model results were examined.

- The development set contains 396 churn outcomes among 2,520 rows (0.157143).
- The largest absolute development-set Spearman association in the shared modelling input frame is 0.935711, for `seconds_of_use` and `frequency_of_use` (signed value 0.935711).
- Development-set churn rates by age band are {"1": 0.0, "2": 0.171913, "3": 0.165347, "4": 0.199377, "5": 0.014815}.
- Development-set churn rates by collapsed charge band are {"0": 0.240515, "1": 0.068627, "2": 0.053797, "3": 0.04321, "4": 0.016393, "5": 0.0}.
- The development set contains 122 rows with zero call usage; their churn rate is 0.516393.
- Quantitative churn-rate sequences by development-set quintile are {"call_failure": [0.173288, 0.139842, 0.136179, 0.152361], "distinct_called_numbers": [0.344231, 0.175047, 0.138614, 0.105263, 0.006211], "frequency_of_sms": [0.228516, 0.223108, 0.2249, 0.09901, 0.00994], "frequency_of_use": [0.358121, 0.201942, 0.165992, 0.054435, 0.0], "mean_call_duration": [0.271287, 0.125249, 0.140873, 0.130952, 0.117063], "seconds_of_use": [0.375, 0.222222, 0.134921, 0.053571, 0.0], "subscription_length": [0.119461, 0.209524, 0.188725, 0.157706, 0.091358]}.
- No observations were removed; substantive zero values are retained.
- `Distinct Called Numbers` and `Frequency of use` are retained without a derived ratio because their source semantics are unresolved in 120 source rows.
- **Pre-model hypothesis:** the zero-usage block may be isolated by a tree threshold, whereas the additive logistic specification represents it through several lower-tail values.
- **Pre-model hypothesis:** subscription length may exhibit a threshold or U-shaped relationship that a single linear logistic term cannot express directly.
- **Pre-model hypothesis:** age-band one-hot terms permit non-monotone logistic effects, while repeated tree splits can also represent them.
