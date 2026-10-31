"""Baselines, the two scikit-learn models, metrics and Optuna tuning."""

from typing import Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from red_tide.features import FEATURES

Kind = Literal["count", "logreg", "hgb"]
# The persistence information alone, as numbers: is weather worth anything on top?
COUNT_FEATURES = ["last_count_log", "last_bloom"]


def persistence(df: pd.DataFrame) -> np.ndarray:
    """A bloom if the last nearby count was one; no history counts as no bloom."""
    return df["last_bloom"].fillna(0.0).to_numpy(dtype=float)


def season_rate(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Bloom rate of each row's calendar month in the training years (0 for unseen months)."""
    rate = train.groupby(train["date"].dt.month)["y"].mean()
    return df["date"].dt.month.map(rate).fillna(0.0).to_numpy(dtype=float)


def make_model(kind: Kind, params: dict, seed: int = 0):
    if kind in ("count", "logreg"):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, **params),
        )
    # Handles NaN natively. No early stopping: it would hold out a random 10 % of the
    # training rows, and the split here is by time, never at random.
    return HistGradientBoostingClassifier(
        class_weight="balanced", early_stopping=False, random_state=seed, **params
    )


def features_for(kind: Kind) -> list[str]:
    return COUNT_FEATURES if kind == "count" else FEATURES


def recall_at_precision(y: np.ndarray, scores: np.ndarray, p: float = 0.5) -> float:
    """Best recall over thresholds whose precision is at least p; 0.0 if none."""
    if not np.any(y):
        return 0.0
    precision, recall, _ = precision_recall_curve(y, scores)
    ok = precision >= p
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y, scores) -> dict:
    y = np.asarray(y).astype(int)
    positives = int(y.sum())
    if positives == 0:
        # PR-AUC is undefined without a single bloom.
        return {"pr_auc": np.nan, "recall_at_p50": np.nan, "positives": 0}
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "recall_at_p50": recall_at_precision(y, np.asarray(scores)),
        "positives": positives,
    }


def suggest(kind: Kind, trial: optuna.Trial) -> dict:
    if kind in ("count", "logreg"):
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-4, 10, log=True
        ),
        "max_iter": trial.suggest_int("max_iter", 100, 400),
    }


def sample_params(kind: Kind, n: int, seed: int) -> list[dict]:
    """n parameter sets drawn at random from the search space, the same for a given seed."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    return [suggest(kind, study.ask()) for _ in range(n)]


def objective(
    kind: Kind, train: pd.DataFrame, val: pd.DataFrame, trial: optuna.Trial
) -> float:
    """Validation PR-AUC of one trial's parameters."""
    cols = features_for(kind)
    model = make_model(kind, suggest(kind, trial))
    model.fit(train[cols], train["y"])
    scores = model.predict_proba(val[cols])[:, 1]
    return float(average_precision_score(val["y"], scores))


def tune(
    kind: Kind,
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_trials: int,
    n_jobs: int,
    seed: int,
    enqueue: list[dict] | None = None,
) -> optuna.Study:
    """`enqueue`: run exactly these parameter sets first (the benchmark uses it so every
    thread count does the same work)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    for params in enqueue or []:
        study.enqueue_trial(params)
    study.optimize(
        lambda trial: objective(kind, train, val, trial),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    return study
