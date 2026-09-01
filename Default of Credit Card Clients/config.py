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
RECALL_FLOOR = 0.65
CALIBRATION_EPSILON = 1e-6
CALIBRATION_BINS = 10
CODE_VERSION_LABEL = "ccd-workflow-2026-08-24-v1"

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "credit_card_default.csv"
SHARED_WORKFLOW_PATH = PROJECT_ROOT / "South German Credit" / "modules" / "workflow.py"
SHARED_EVALUATION_PATH = PROJECT_ROOT / "South German Credit" / "modules" / "evaluation.py"

TARGET_RAW = "default payment next month"
RAW_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    TARGET_RAW,
]

RAW_TO_FEATURE = {
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education_level",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "PAY_0": "repayment_status_september",
    "PAY_2": "repayment_status_august",
    "PAY_3": "repayment_status_july",
    "PAY_4": "repayment_status_june",
    "PAY_5": "repayment_status_may",
    "PAY_6": "repayment_status_april",
    "BILL_AMT1": "bill_amount_september",
    "BILL_AMT2": "bill_amount_august",
    "BILL_AMT3": "bill_amount_july",
    "BILL_AMT4": "bill_amount_june",
    "BILL_AMT5": "bill_amount_may",
    "BILL_AMT6": "bill_amount_april",
    "PAY_AMT1": "payment_amount_september",
    "PAY_AMT2": "payment_amount_august",
    "PAY_AMT3": "payment_amount_july",
    "PAY_AMT4": "payment_amount_june",
    "PAY_AMT5": "payment_amount_may",
    "PAY_AMT6": "payment_amount_april",
}

PREDICTORS = [RAW_TO_FEATURE[column] for column in RAW_COLUMNS if column not in {"ID", TARGET_RAW}]

BILL_FEATURES = [
    "bill_amount_september", "bill_amount_august", "bill_amount_july",
    "bill_amount_june", "bill_amount_may", "bill_amount_april",
]
PAYMENT_FEATURES = [
    "payment_amount_september", "payment_amount_august", "payment_amount_july",
    "payment_amount_june", "payment_amount_may", "payment_amount_april",
]
ORDINAL = [
    "repayment_status_september", "repayment_status_august", "repayment_status_july",
    "repayment_status_june", "repayment_status_may", "repayment_status_april",
]
QUANTITATIVE = ["credit_limit", "age", *BILL_FEATURES, *PAYMENT_FEATURES]
BINARY = ["sex"]
NOMINAL = ["education_level", "marital_status"]
LR_NUMERIC = ["credit_limit", "age", "bill_amount_september"]
LR_USED_FEATURES = [
    "credit_limit", "sex", "education_level", "marital_status", "age",
    *ORDINAL, "bill_amount_september", *PAYMENT_FEATURES,
]
LR_PRUNED_FEATURES = [
    "bill_amount_august", "bill_amount_july", "bill_amount_june",
    "bill_amount_may", "bill_amount_april",
]

DISPLAY_NAMES = {
    "credit_limit": "Credit limit (NT$)",
    "sex": "Sex",
    "education_level": "Education level",
    "marital_status": "Marital status",
    "age": "Age (years)",
    "repayment_status_september": "Repayment status, September",
    "repayment_status_august": "Repayment status, August",
    "repayment_status_july": "Repayment status, July",
    "repayment_status_june": "Repayment status, June",
    "repayment_status_may": "Repayment status, May",
    "repayment_status_april": "Repayment status, April",
    "bill_amount_september": "Bill amount, September (NT$)",
    "bill_amount_august": "Bill amount, August (NT$)",
    "bill_amount_july": "Bill amount, July (NT$)",
    "bill_amount_june": "Bill amount, June (NT$)",
    "bill_amount_may": "Bill amount, May (NT$)",
    "bill_amount_april": "Bill amount, April (NT$)",
    "payment_amount_september": "Payment made, September (NT$)",
    "payment_amount_august": "Payment made, August (NT$)",
    "payment_amount_july": "Payment made, July (NT$)",
    "payment_amount_june": "Payment made, June (NT$)",
    "payment_amount_may": "Payment made, May (NT$)",
    "payment_amount_april": "Payment made, April (NT$)",
    "default": "Default next month",
}

FEATURE_KEYS = {feature: feature for feature in PREDICTORS}

NOMINAL_CATEGORIES = {
    "education_level": [1, 2, 3, 4],
    "marital_status": [1, 2, 3],
}
LR_REFERENCE_LEVELS = {"education_level": 2, "marital_status": 2}

REPAYMENT_CATEGORIES = {
    "repayment_status_september": [
        "non_positive_status", "delay_1m", "delay_2m", "delay_3m_plus"
    ],
    "repayment_status_august": ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
    "repayment_status_july": ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
    "repayment_status_june": ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
    "repayment_status_may": ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
    "repayment_status_april": ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
}

LEVEL_LABELS = {
    "education_level": {
        1: "graduate school", 2: "university", 3: "high school",
        4: "other or undocumented",
    },
    "marital_status": {1: "married", 2: "single", 3: "other or undocumented"},
    "repayment_status_september": {
        "non_positive_status": "non-positive status",
        "delay_1m": "one-month delay",
        "delay_2m": "two-month delay",
        "delay_3m_plus": "delay of three months or more",
    },
}
for _feature in ORDINAL[1:]:
    LEVEL_LABELS[_feature] = {
        "non_positive_status": "non-positive status",
        "delay_1_2m": "delay of one or two months",
        "delay_3m_plus": "delay of three months or more",
    }

MODEL_ORDER = ["lr", "dt", "rf", "xgb"]
MODEL_DISPLAY = {
    "lr": "Logistic regression",
    "dt": "Decision tree",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "dummy": "Dummy classifier",
}
