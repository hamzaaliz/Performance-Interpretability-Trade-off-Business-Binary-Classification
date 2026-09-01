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
RECALL_FLOOR = 0.75
CALIBRATION_EPSILON = 1e-6
CALIBRATION_BINS = 10
CODE_VERSION_LABEL = "sgc-workflow-2026-08-22-v1"

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SouthGermanCredit.asc"

RAW_COLUMNS = [
    "laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont",
    "beszeit", "rate", "famges", "buerge", "wohnzeit", "verm",
    "alter", "weitkred", "wohn", "bishkred", "beruf", "pers",
    "telef", "gastarb", "kredit",
]

QUANTITATIVE = ["laufzeit", "hoehe", "alter"]
ORDINAL = [
    "laufkont", "moral", "sparkont", "beszeit", "rate", "wohnzeit",
    "verm", "bishkred", "beruf",
]
BINARY = ["pers", "telef", "gastarb"]
NOMINAL = ["verw", "famges", "buerge", "weitkred", "wohn"]
PREDICTORS = QUANTITATIVE + ORDINAL + BINARY + NOMINAL

DISPLAY_NAMES = {
    "laufzeit": "Credit duration (months)",
    "hoehe": "Credit amount (DM)",
    "alter": "Age (years)",
    "laufkont": "Checking account status",
    "moral": "Credit history",
    "sparkont": "Savings",
    "beszeit": "Employment duration",
    "rate": "Instalment rate (% of income)",
    "wohnzeit": "Time at present residence",
    "verm": "Most valuable property",
    "bishkred": "Number of credits at this bank",
    "beruf": "Job quality",
    "pers": "Three or more dependants",
    "telef": "Telephone registered",
    "gastarb": "Foreign worker",
    "verw": "Purpose",
    "famges": "Personal status and sex",
    "buerge": "Other debtors or guarantor",
    "weitkred": "Other instalment plans",
    "wohn": "Housing",
    "kredit": "Credit risk",
}

NOMINAL_CATEGORIES = {
    "verw": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
    "famges": [1, 2, 3, 4],
    "buerge": [1, 2, 3],
    "weitkred": [1, 2, 3],
    "wohn": [1, 2, 3],
}

LR_REFERENCE_LEVELS = {
    "verw": 3,
    "famges": 3,
    "buerge": 1,
    "weitkred": 3,
    "wohn": 2,
}

LEVEL_LABELS = {
    "verw": {
        0: "others", 1: "car (new)", 2: "car (used)",
        3: "furniture/equipment", 4: "radio/television",
        5: "domestic appliances", 6: "repairs", 8: "vacation",
        9: "retraining", 10: "business",
    },
    "famges": {
        1: "male: divorced/separated",
        2: "female: non-single or male: single",
        3: "male: married/widowed",
        4: "female: single",
    },
    "buerge": {1: "none", 2: "co-applicant", 3: "guarantor"},
    "weitkred": {1: "bank", 2: "stores", 3: "none"},
    "wohn": {1: "for free", 2: "rent", 3: "own"},
}

MODEL_ORDER = ["lr", "dt", "rf", "xgb"]
MODEL_DISPLAY = {
    "lr": "Logistic regression",
    "dt": "Decision tree",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "dummy": "Dummy classifier",
}
