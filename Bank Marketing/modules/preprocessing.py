import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (
    BINARY,
    DISPLAY_NAMES,
    LEVEL_LABELS,
    LR_REFERENCE_LEVELS,
    NOMINAL,
    NOMINAL_CATEGORIES,
    PREDICTORS,
)


def apply_locked_rulings(frame):
    out = frame.copy()
    for feature in BINARY:
        out[feature] = out[feature].map({"no": 0, "yes": 1})
        if out[feature].isna().any():
            raise RuntimeError(f"Unexpected binary level in {feature}")
        out[feature] = out[feature].astype(int)
    return out[PREDICTORS].copy()


def _nominal_encoder(drop_reference):
    categories = [NOMINAL_CATEGORIES[column] for column in NOMINAL]
    drop = [LR_REFERENCE_LEVELS[column] for column in NOMINAL] if drop_reference else None
    return OneHotEncoder(
        categories=categories,
        drop=drop,
        handle_unknown="ignore",
        sparse_output=False,
        dtype=float,
    )


def _signed_log(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _pdays_clean(values):
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0, values, 0.0)


def _previous_log1p(values):
    return np.log1p(np.asarray(values, dtype=float))


def _previously_contacted(values):
    return (np.asarray(values, dtype=float) >= 0).astype(float)


def _transform_scale(function):
    return Pipeline(
        [
            (
                "transform",
                FunctionTransformer(function, validate=False, feature_names_out="one-to-one"),
            ),
            ("scale", StandardScaler()),
        ]
    )


def make_lr_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["age", "day", "campaign"]),
            ("balance_signed_log", _transform_scale(_signed_log), ["balance"]),
            ("pdays_clean", _transform_scale(_pdays_clean), ["pdays"]),
            ("previous_log1p", _transform_scale(_previous_log1p), ["previous"]),
            (
                "previously_contacted",
                FunctionTransformer(
                    _previously_contacted,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
                ["pdays"],
            ),
            ("binary", "passthrough", BINARY),
            ("nominal", _nominal_encoder(drop_reference=True), NOMINAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_tree_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("quantitative", "passthrough", ["age", "balance", "day", "campaign", "pdays", "previous"]),
            ("binary", "passthrough", BINARY),
            ("nominal", _nominal_encoder(drop_reference=False), NOMINAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encoded_feature_metadata(fitted_preprocessor):
    names = list(fitted_preprocessor.get_feature_names_out())
    raw_names = []
    display_terms = []
    for encoded_name in names:
        transformer, remainder = encoded_name.split("__", 1)
        if transformer == "nominal":
            raw = None
            for candidate in sorted(NOMINAL, key=len, reverse=True):
                prefix = candidate + "_"
                if remainder.startswith(prefix):
                    raw = candidate
                    level = remainder[len(prefix):]
                    display_terms.append(
                        f"{DISPLAY_NAMES[candidate]} = {LEVEL_LABELS[candidate].get(level, level)}"
                    )
                    break
            if raw is None:
                raise RuntimeError(f"Unmapped nominal encoded term: {encoded_name}")
        elif transformer == "previously_contacted":
            raw = "pdays"
            display_terms.append(DISPLAY_NAMES["previously_contacted"])
        elif transformer == "pdays_clean":
            raw = "pdays"
            display_terms.append("Days since previous contact (cleaned)")
        elif transformer == "previous_log1p":
            raw = "previous"
            display_terms.append("Contacts before this campaign (log1p)")
        elif transformer == "balance_signed_log":
            raw = "balance"
            display_terms.append("Average yearly balance (signed-log, standardised)")
        else:
            raw = remainder
            display_terms.append(DISPLAY_NAMES.get(raw, raw))
        raw_names.append(raw)
    if any(raw not in PREDICTORS for raw in raw_names):
        unknown = sorted({raw for raw in raw_names if raw not in PREDICTORS})
        raise RuntimeError(f"Unmapped encoded features: {unknown}")
    return names, raw_names, display_terms
