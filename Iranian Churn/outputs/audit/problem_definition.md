# Problem definition

- **Target:** Customer churn; churn is the positive class.
- **Observation:** One Iranian telecommunications customer, described by aggregates over months 1–9; churn is recorded at month 12.
- **Primary decision:** Whether to target a customer with a retention intervention before the churn outcome.
- **Stakeholders:** The telecommunications provider and the customer considered for retention contact.
- **False positive:** A customer who stayed, targeted with a retention offer.
- **False negative:** A customer who churned, not targeted.
- **Evaluation stance:** Models are evaluated as ranking and classification systems; probability statements are limited to the supplied sample.
- **Holdout status:** The stratified test set is a one-time held-out confirmation; development evidence governs comparison.
