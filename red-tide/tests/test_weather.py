import io
import json
import urllib.error
from datetime import date

import numpy as np
import pytest

from red_tide import weather

CELL = (27.375, -82.625)
PAYLOAD = {
    "daily": {
        "time": ["2005-01-01", "2005-01-02", "2005-01-03"],
        "temperature_2m_mean": [18.3, None, 19.7],
        "wind_speed_10m_mean": [7.0, 9.4, 10.0],
        "wind_direction_10m_dominant": [141, 90, None],
        "precipitation_sum": [0.0, 1.5, 0.0],
    }
}


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(json.dumps(PAYLOAD).encode())

    monkeypatch.setattr(weather.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    return calls


def test_cell_of_snaps_to_center():
    assert weather.cell_of(27.30, -82.60) == (27.375, -82.625)
    assert weather.cell_of(26.0, -82.0) == (26.125, -81.875)


def test_cache_hit_makes_no_request(tmp_path, fake_api):
    first = weather.fetch_cell(CELL, date(2005, 1, 1), date(2005, 1, 3), tmp_path)
    second = weather.fetch_cell(CELL, date(2005, 1, 1), date(2005, 1, 3), tmp_path)
    assert len(fake_api) == 1
    assert first.equals(second)
    assert list(first.columns) == ["date", "temp", "wind_speed", "wind_dir", "rain"]


def test_nulls_kept_as_nan(tmp_path, fake_api):
    df = weather.fetch_cell(CELL, date(2005, 1, 1), date(2005, 1, 3), tmp_path)
    assert len(df) == 3
    assert np.isnan(df["temp"][1]) and np.isnan(df["wind_dir"][2])


def test_retries_then_raises(tmp_path, monkeypatch):
    tries = []

    def urlopen(url, timeout=None):
        tries.append(url)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(weather.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="27.375"):
        weather.fetch_cell(CELL, date(2005, 1, 1), date(2005, 1, 3), tmp_path)
    assert len(tries) == 3


def test_load_all_adds_cell_columns(tmp_path, fake_api):
    df = weather.load_all([CELL], date(2005, 1, 1), date(2005, 1, 3), tmp_path)
    assert (df["cell_lat"] == 27.375).all() and (df["cell_lon"] == -82.625).all()
