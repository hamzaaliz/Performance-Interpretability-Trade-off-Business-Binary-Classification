# Development-set EDA findings

All predictor–target statements below use development data only. Hypotheses are labelled and were recorded before model results were examined.

- The development set contains 4,231 subscriptions among 36,168 records (0.116982).
- The long-form mixed-type association file contains 36 Spearman pairs and 15 Cramer's V pairs. Mixed quantitative–nominal pairs are not computed.
- The largest absolute development-set Spearman association among quantitative and binary predictors is 0.985784 for `pdays` and `previous` (signed value 0.985784).
- The largest development-set Cramer's V among nominal predictors is 0.511174 for `contact` and `month`.
- The development-set `pdays`–`previous` Spearman association is 0.985784; among previously contacted development rows it is -0.098913. The locked LR decomposition is unchanged.
- Quantitative positive-rate sequences by development-set quintile are: {"age": [0.15129, 0.104005, 0.09505, 0.090482, 0.147774], "balance": [0.068222, 0.100818, 0.117215, 0.134836, 0.164132], "campaign": [0.131565, 0.103783, 0.064871], "day": [0.125208, 0.135507, 0.112663, 0.098321, 0.112956], "pdays": [0.116982], "previous": [0.116982]}.
- The source is date ordered. The first and last row-position deciles have subscription rates 0.029854 and 0.470914. Row position is excluded and the split is shuffled and stratified.
- `month` and `day` are treated only as campaign-timing indicators; no seasonality variable is inferred.
- No observations were removed. The retained development values include negative balances and the `pdays = -1` sentinel.
- `duration` remains excluded from every model. Its development-only Spearman association with subscription is 0.342683; its development quintile subscription-rate sequence is [0.007268, 0.038483, 0.077742, 0.130731, 0.331765].
- **Pre-model hypothesis:** the shared never-contacted state may create a large level shift, while `pdays` and `previous` may retain distinct within-contacted information.
- **Pre-model hypothesis:** the tree family may represent category interactions and threshold patterns that the additive logistic-regression specification does not encode directly.
- **Pre-model hypothesis:** the date-ordered campaign structure may be reflected in `month` and `day`, so their evidence is not interpreted as portable seasonality.
