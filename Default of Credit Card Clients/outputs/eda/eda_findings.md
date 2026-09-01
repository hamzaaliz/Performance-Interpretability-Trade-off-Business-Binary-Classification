# Development-set EDA findings

All statements below describe observed development or source structure. Hypotheses are labelled and were locked before model results were examined.

- The development set contains 5,309 defaults among 24,000 accounts (0.2212).
- The largest absolute pairwise Spearman correlation is 0.9098, between **Bill amount, September (NT$)** and **Bill amount, August (NT$)** (signed value 0.9098).
- Across the six bill amounts, mean absolute Pearson correlation is 0.8897 and the maximum is 0.9517. This verifies the locked LR pruning decision; trees retain the full block.
- Across the six raw repayment-status variables, mean absolute Pearson correlation is 0.6545 and the maximum is 0.8173.
- Latest-month repayment buckets show a threshold pattern: non-positive status: 18,506 rows, rate 0.1373; one-month delay: 3,000 rows, rate 0.3437; two-month delay: 2,138 rows, rate 0.6936; delay of three months or more: 356 rows, rate 0.7163.
- The audit-only average bill-to-limit decile default-rate sequence is [0.255, 0.1875, 0.146667, 0.14125, 0.165417, 0.194167, 0.233333, 0.2575, 0.299583, 0.331667]. It is not an engineered model feature.
- The zero-recorded bill-and-payment segment contains 642 development accounts, with default rate 0.3738, against 0.2170 outside the segment. No segment indicator is engineered.
- The development-only rule flagging any positive repayment-status code marks 8,104 accounts, with recall 0.6553 and precision 0.4293. It is contextual evidence only and cannot alter the 0.65 recall floor.
- The final source-position decile has default rate 0.2210. Row position is excluded from all models; the split is shuffled and stratified.
- No observations were removed. Negative bills and extreme bill/payment amounts remain in the model inputs specified by the decisions file.
- **Pre-model hypothesis:** the repayment-status cliff can provide strong monotonic or threshold signal to all four pipelines, with LR expressing the locked buckets and trees using raw codes.
- **Pre-model hypothesis:** the tree family may use historical bill trajectory, bill-to-limit interactions or the zero-recorded segment that are not explicitly represented by the LR pipeline.
- **Pre-model hypothesis:** correlated bill variables may distribute tree native importance differently from grouped permutation importance.

Development default-rate sequences by quintile were: Credit limit [0.321673, 0.25923, 0.199512, 0.169178, 0.130784]; Age [0.242401, 0.197572, 0.201085, 0.2174, 0.243522].
