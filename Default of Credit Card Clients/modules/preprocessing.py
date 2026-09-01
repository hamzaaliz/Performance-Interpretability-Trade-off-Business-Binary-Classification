import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from config import (
    BINARY,
    BILL_FEATURES,
    DISPLAY_NAMES,
    LEVEL_LABELS,
    LR_NUMERIC,
    LR_REFERENCE_LEVELS,
    NOMINAL,
    NOMINAL_CATEGORIES,
    ORDINAL,
    PAYMENT_FEATURES,
    PREDICTORS,
    RAW_TO_FEATURE,
    REPAYMENT_CATEGORIES,
)


def apply_locked_rulings(frame):
    out = frame.copy()
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    out["SEX"] = out["SEX"].map({1: 0, 2: 1}).astype(int)
    out = out.rename(columns=RAW_TO_FEATURE)
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


def _bucket_repayments(values):
    values = np.asarray(values)
    bucketed = np.empty(values.shape, dtype=object)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        if column_index == 0:
            bucketed[:, column_index] = np.select(
                [column <= 0, column == 1, column == 2, column >= 3],
                ["non_positive_status", "delay_1m", "delay_2m", "delay_3m_plus"],
                default="unexpected",
            )
        else:
            bucketed[:, column_index] = np.select(
                [column <= 0, np.isin(column, [1, 2]), column >= 3],
                ["non_positive_status", "delay_1_2m", "delay_3m_plus"],
                default="unexpected",
            )
    if np.any(bucketed == "unexpected"):
        raise RuntimeError("Unexpected repayment-status value reached the locked bucketing step")
    return bucketed


def _repayment_encoder():
    return Pipeline(
        [
            (
                "bucket",
                FunctionTransformer(
                    _bucket_repayments,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "encode",
                OneHotEncoder(
                    categories=[REPAYMENT_CATEGORIES[column] for column in ORDINAL],
                    drop=["non_positive_status"] * len(ORDINAL),
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )


def make_lr_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", RobustScaler(), LR_NUMERIC),
            (
                "payment",
                Pipeline(
                    [
                        (
                            "log1p",
                            FunctionTransformer(
                                np.log1p,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("scale", RobustScaler()),
                    ]
                ),
                PAYMENT_FEATURES,
            ),
            ("binary", "passthrough", BINARY),
            ("nominal", _nominal_encoder(drop_reference=True), NOMINAL),
            ("repayment", _repayment_encoder(), ORDINAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_tree_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "quantitative",
                "passthrough",
                ["credit_limit", "age", *BILL_FEATURES, *PAYMENT_FEATURES],
            ),
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
    categorical = sorted([*NOMINAL, *ORDINAL], key=len, reverse=True)
    for encoded_name in names:
        remainder = encoded_name.split("__", 1)[1]
        raw = None
        for candidate in categorical:
            prefix = candidate + "_"
            if remainder.startswith(prefix):
                raw = candidate
                level_text = remainder[len(prefix):]
                try:
                    level = int(float(level_text))
                except ValueError:
                    level = level_text
                level_label = LEVEL_LABELS.get(candidate, {}).get(level, str(level))
                display_terms.append(f"{DISPLAY_NAMES[candidate]} = {level_label}")
                break
        if raw is None:
            raw = remainder
            display_terms.append(DISPLAY_NAMES.get(raw, raw))
        raw_names.append(raw)
    if any(raw not in PREDICTORS for raw in raw_names):
        unknown = sorted({raw for raw in raw_names if raw not in PREDICTORS})
        raise RuntimeError(f"Unmapped encoded features: {unknown}")
    return names, raw_names, display_terms
