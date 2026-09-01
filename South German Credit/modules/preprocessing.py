import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (
    BINARY,
    DISPLAY_NAMES,
    LR_REFERENCE_LEVELS,
    NOMINAL,
    NOMINAL_CATEGORIES,
    ORDINAL,
)


def apply_locked_rulings(df):
    out = df.copy()
    out["bishkred"] = out["bishkred"].replace({4: 3})
    out["pers"] = out["pers"].map({1: 1, 2: 0}).astype(int)
    out["telef"] = out["telef"].map({1: 0, 2: 1}).astype(int)
    out["gastarb"] = out["gastarb"].map({1: 1, 2: 0}).astype(int)
    return out


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


def make_lr_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("quant", StandardScaler(), ["laufzeit", "alter"]),
            (
                "amount",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["hoehe"],
            ),
            ("ordinal", StandardScaler(), ORDINAL),
            ("binary", "passthrough", BINARY),
            ("nominal", _nominal_encoder(drop_reference=True), NOMINAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_tree_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("quant", "passthrough", ["laufzeit", "hoehe", "alter"]),
            ("ordinal", "passthrough", ORDINAL),
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
    for name in names:
        remainder = name.split("__", 1)[1]
        raw = None
        for candidate in NOMINAL:
            if remainder.startswith(candidate + "_"):
                raw = candidate
                level_text = remainder[len(candidate) + 1 :]
                try:
                    level = int(float(level_text))
                except ValueError:
                    level = level_text
                label = str(level)
                from config import LEVEL_LABELS

                label = LEVEL_LABELS.get(candidate, {}).get(level, label)
                display_terms.append(f"{DISPLAY_NAMES[candidate]} = {label}")
                break
        if raw is None:
            raw = remainder
            display_terms.append(DISPLAY_NAMES.get(raw, raw))
        raw_names.append(raw)
    return names, raw_names, display_terms
