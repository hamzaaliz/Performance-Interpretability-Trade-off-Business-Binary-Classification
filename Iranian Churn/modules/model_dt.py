from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import SEED
from modules.preprocessing import make_tree_preprocessor


def make_pipeline():
    return Pipeline(
        [
            ("preprocessor", make_tree_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ]
    )


def search_space():
    return {
        "classifier__max_depth": randint(2, 6),
        "classifier__min_samples_leaf": randint(5, 61),
        "classifier__min_samples_split": randint(10, 121),
        "classifier__criterion": ["gini", "entropy"],
        "classifier__ccp_alpha": uniform(0.0, 0.02),
    }
