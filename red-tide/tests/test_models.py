from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from red_tide import features, models

FIXTURE = Path(__file__).parent / "fixtures" / "features.csv"


@pytest.fixture(scope="module")
def parts():
    return features.split(pd.read_csv(FIXTURE, parse_dates=["date"]))


def test_recall_at_precision():
    y = np.array([1, 0, 1, 0])
    assert models.recall_at_precision(y, np.array([0.9, 0.8, 0.7, 0.1])) == 1.0
    y5, s5 = np.array([1, 0, 0, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert models.recall_at_precision(y5, s5) == 0.5  # all-in gives precision 2/5
    assert models.recall_at_precision(np.array([0, 0]), np.array([0.5, 0.6])) == 0.0


def test_metrics_with_no_positives():
    m = models.evaluate(np.zeros(5), np.linspace(0, 1, 5))
    assert m["positives"] == 0
    assert np.isnan(m["pr_auc"]) and np.isnan(m["recall_at_p50"])


def test_evaluate_perfect_scores():
    m = models.evaluate(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert m == {"pr_auc": 1.0, "recall_at_p50": 1.0, "positives": 2}


def test_no_history_is_not_a_bloom_prediction():
    df = pd.DataFrame({"last_bloom": [np.nan, 1.0, 0.0]})
    assert models.persistence(df).tolist() == [0.0, 1.0, 0.0]


def test_season_rate_uses_train_months_only():
    train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-09-01", "2010-09-15", "2011-03-01"]),
            "y": [True, False, False],
        }
    )
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2018-09-20", "2018-03-02", "2018-06-01"])}
    )
    assert models.season_rate(train, df).tolist() == [0.5, 0.0, 0.0]


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
def test_models_handle_nan_features(parts, kind):
    model = models.make_model(kind, {})
    tr = parts["train"]
    model.fit(tr[features.FEATURES], tr["y"])
    scores = model.predict_proba(parts["test"][features.FEATURES])[:, 1]
    assert np.isfinite(scores).all()


def test_tune_runs_small_study_on_fixture(parts):
    study = models.tune(
        "hgb", parts["train"], parts["val"], n_trials=4, n_jobs=2, seed=1
    )
    assert len(study.trials) == 4
    assert 0.0 <= study.best_value <= 1.0


def test_hgb_has_no_random_early_stopping():
    assert models.make_model("hgb", {}).early_stopping is False


def test_tune_with_enqueued_params_runs_exactly_those(parts):
    params = models.sample_params("hgb", 3, seed=1)
    study = models.tune(
        "hgb", parts["train"], parts["val"], 3, 2, seed=1, enqueue=params
    )
    key = lambda p: sorted(p.items())
    assert sorted(map(key, (t.params for t in study.trials))) == sorted(
        map(key, params)
    )


def test_count_only_model_uses_two_features(parts):
    model = models.make_model("count", {})
    tr = parts["train"]
    model.fit(tr[models.COUNT_FEATURES], tr["y"])
    assert models.COUNT_FEATURES == ["last_count_log", "last_bloom"]
