from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import SEED
from modules.preprocessing import make_tree_preprocessor


def make_pipeline():
    return Pipeline(
        [
            ("preprocessor", make_tree_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                ),
            ),
        ]
    )


def search_space():
    return {
        "classifier__n_estimators": randint(200, 601),
        "classifier__max_depth": randint(2, 7),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 11),
        "classifier__reg_lambda": loguniform(0.1, 10),
    }
