from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import SEED
from modules.preprocessing import make_tree_preprocessor


def make_pipeline():
    return Pipeline(
        [
            ("preprocessor", make_tree_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=None),
            ),
        ]
    )


def search_space():
    return {
        "classifier__n_estimators": randint(300, 801),
        "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
        "classifier__min_samples_leaf": randint(1, 26),
        "classifier__max_features": ["sqrt", "log2", 0.5],
    }
