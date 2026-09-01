from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import SEED
from modules.preprocessing import make_lr_preprocessor


def make_pipeline():
    return Pipeline(
        [
            ("preprocessor", make_lr_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=2000, random_state=SEED
                ),
            ),
        ]
    )


def search_space():
    return {"classifier__C": loguniform(1e-3, 1e2)}
