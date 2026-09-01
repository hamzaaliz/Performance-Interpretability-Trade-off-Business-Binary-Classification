import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (
    DISPLAY_NAMES, LEVEL_LABELS, LR_REFERENCE_LEVELS, NOMINAL_CATEGORIES,
    PREDICTORS, TREE_FEATURES,
)


def apply_locked_rulings(frame):
    out = frame.copy()
    out["charge_amount"] = out["charge_amount"].clip(upper=5).astype(int)
    out["complains"] = out["complains"].astype(int)
    out["tariff_plan"] = out["tariff_plan"].eq(2).astype(int)
    out["status"] = out["status"].eq(2).astype(int)
    out["mean_call_duration"] = np.divide(
        out["seconds_of_use"].to_numpy(dtype=float),
        out["frequency_of_use"].to_numpy(dtype=float),
        out=np.zeros(len(out), dtype=float),
        where=out["frequency_of_use"].to_numpy() != 0,
    )
    return out[PREDICTORS].copy()


def _age_encoder():
    return OneHotEncoder(
        categories=[NOMINAL_CATEGORIES["age_group"]],
        drop=[LR_REFERENCE_LEVELS["age_group"]],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=float,
    )


def make_lr_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "quantitative",
                StandardScaler(),
                [
                    "call_failure", "subscription_length", "frequency_of_use",
                    "frequency_of_sms", "distinct_called_numbers", "mean_call_duration",
                ],
            ),
            ("ordinal", StandardScaler(), ["charge_amount"]),
            ("binary", "passthrough", ["complains", "tariff_plan", "status"]),
            ("age_group", _age_encoder(), ["age_group"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_tree_preprocessor():
    return ColumnTransformer(
        transformers=[("raw", "passthrough", TREE_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encoded_feature_metadata(fitted_preprocessor):
    names = list(fitted_preprocessor.get_feature_names_out())
    raw_names = []
    display_terms = []
    for encoded_name in names:
        transformer, remainder = encoded_name.split("__", 1)
        if transformer == "age_group":
            prefix = "age_group_"
            if not remainder.startswith(prefix):
                raise RuntimeError(f"Unmapped age-band term: {encoded_name}")
            level = int(float(remainder[len(prefix):]))
            raw = "age_group"
            display_terms.append(f"{DISPLAY_NAMES[raw]} = {LEVEL_LABELS[raw][level]}")
        else:
            raw = remainder
            display_terms.append(DISPLAY_NAMES.get(raw, raw))
        raw_names.append(raw)
    if any(raw not in PREDICTORS for raw in raw_names):
        unknown = sorted({raw for raw in raw_names if raw not in PREDICTORS})
        raise RuntimeError(f"Unmapped encoded features: {unknown}")
    return names, raw_names, display_terms
