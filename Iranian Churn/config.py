from pathlib import Path


SEED = 42
TEST_SIZE = 0.20
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEARCH_ITERATIONS = 40
BOOTSTRAP_RESAMPLES = 5000
PERMUTATION_REPEATS = 30
SHAP_ROW_CAP = 2000
SHAP_BACKGROUND = 200
RECALL_FLOOR = 0.80
CALIBRATION_EPSILON = 1e-6
CALIBRATION_BINS = 10
CODE_VERSION_LABEL = "iranian-churn-workflow-2026-08-25-v1"

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "Customer Churn.csv"

TARGET_RAW = "Churn"
RAW_COLUMNS = [
    "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age",
    "Customer Value", TARGET_RAW,
]
INTERNAL_COLUMNS = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status", "age",
    "customer_value", "churn",
]
COLUMN_MAP = dict(zip(RAW_COLUMNS, INTERNAL_COLUMNS))

PREDICTORS = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "mean_call_duration", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status",
]
LR_FEATURES = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "mean_call_duration", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status",
]
TREE_FEATURES = [
    "call_failure", "complains", "subscription_length", "charge_amount",
    "seconds_of_use", "frequency_of_use", "frequency_of_sms",
    "distinct_called_numbers", "age_group", "tariff_plan", "status",
]
MODEL_FEATURES = {
    "lr": LR_FEATURES,
    "dt": TREE_FEATURES,
    "rf": TREE_FEATURES,
    "xgb": TREE_FEATURES,
}
QUANTITATIVE = [
    "call_failure", "subscription_length", "seconds_of_use", "frequency_of_use",
    "frequency_of_sms", "distinct_called_numbers",
]
BINARY = ["complains", "tariff_plan", "status"]
NOMINAL = ["age_group"]
ORDINAL = ["charge_amount"]

DISPLAY_NAMES = {
    "call_failure": "Call failures",
    "complains": "Complaint registered",
    "subscription_length": "Subscription length (months)",
    "charge_amount": "Charge amount band",
    "seconds_of_use": "Seconds of use",
    "mean_call_duration": "Mean call duration (seconds)",
    "frequency_of_use": "Number of calls",
    "frequency_of_sms": "Number of text messages",
    "distinct_called_numbers": "Distinct called numbers",
    "age_group": "Age band",
    "tariff_plan": "Contractual plan",
    "status": "Account inactive",
    "age": "Representative age",
    "customer_value": "Customer value score",
    "churn": "Churn",
}

NOMINAL_CATEGORIES = {"age_group": [1, 2, 3, 4, 5]}
LR_REFERENCE_LEVELS = {"age_group": 3}
LEVEL_LABELS = {"age_group": {value: f"Band {value}" for value in [1, 2, 3, 4, 5]}}
EXPECTED_LEVELS = {
    "complains": [0, 1],
    "tariff_plan": [1, 2],
    "status": [1, 2],
    "age_group": [1, 2, 3, 4, 5],
    "churn": [0, 1],
}

MODEL_ORDER = ["lr", "dt", "rf", "xgb"]
MODEL_DISPLAY = {
    "lr": "Logistic regression",
    "dt": "Decision tree",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "dummy": "Dummy classifier",
}
