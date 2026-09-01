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
PRECISION_FLOOR = 0.50
CALIBRATION_EPSILON = 1e-6
CALIBRATION_BINS = 10
CODE_VERSION_LABEL = "bank-marketing-workflow-2026-08-24-v1"

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "bank-full.csv"

TARGET_RAW = "y"
RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", TARGET_RAW,
]
PREDICTORS = [
    "age", "balance", "day", "campaign", "pdays", "previous", "default",
    "housing", "loan", "job", "marital", "education", "contact", "month",
    "poutcome",
]
QUANTITATIVE = ["age", "balance", "day", "campaign", "pdays", "previous"]
BINARY = ["default", "housing", "loan"]
NOMINAL = ["job", "marital", "education", "contact", "month", "poutcome"]
ORDINAL = []

DISPLAY_NAMES = {
    "age": "Age (years)",
    "balance": "Average yearly balance (EUR)",
    "day": "Day of month of planned contact",
    "campaign": "Contacts this campaign",
    "pdays": "Days since previous contact",
    "previous": "Contacts before this campaign",
    "previously_contacted": "Previously contacted",
    "default": "Credit in default",
    "housing": "Housing loan",
    "loan": "Personal loan",
    "job": "Occupation",
    "marital": "Marital status",
    "education": "Education level",
    "contact": "Contact channel",
    "month": "Month of planned contact",
    "poutcome": "Previous campaign outcome",
    "duration": "Last contact duration (s)",
    "subscribed": "Term deposit subscribed",
}

NOMINAL_CATEGORIES = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
    "poutcome": ["failure", "other", "success", "unknown"],
}
LR_REFERENCE_LEVELS = {
    "job": "blue-collar",
    "marital": "married",
    "education": "secondary",
    "contact": "cellular",
    "month": "may",
    "poutcome": "unknown",
}
LEVEL_LABELS = {feature: {level: level for level in levels} for feature, levels in NOMINAL_CATEGORIES.items()}

EXPECTED_LEVELS = {
    **NOMINAL_CATEGORIES,
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "y": ["no", "yes"],
}

MODEL_ORDER = ["lr", "dt", "rf", "xgb"]
MODEL_DISPLAY = {
    "lr": "Logistic regression",
    "dt": "Decision tree",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "dummy": "Dummy classifier",
}