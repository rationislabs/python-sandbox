import numpy as np
import pandas as pd
import pytest

from red_tide import features, weather

T = pd.Timestamp("2010-06-30")
SITE = (27.30, -82.60)  # cell (27.375, -82.625)
WX_CELL = weather.cell_of(*SITE, weather.WX_DEG)


def samples(*rows):
    """rows of (date, lat, lon, cells)."""
    return pd.DataFrame(rows, columns=["date", "lat", "lon", "cells"])


def wx(days=60, temp=None, wind_dir=None, cell=WX_CELL):
    dates = pd.date_range(T - pd.Timedelta(days=days - 1), T)
    df = pd.DataFrame(
        {
            "date": dates,
            "temp": 20.0 if temp is None else temp,
            "wind_speed": 10.0,
            "wind_dir": 90.0 if wind_dir is None else wind_dir,
            "rain": 1.0,
        }
    )
    return df.assign(cell_lat=cell[0], cell_lon=cell[1])


def row_on(df, date):
    match = df[df["date"] == date]
    assert len(match) == 1
    return match.iloc[0]


def km_north(km):
    return SITE[0] + km / 111.2


def test_no_feature_uses_data_after_t_minus_7():
    temp = np.full(60, 20.0)
    temp[-7:] = 40.0  # spike on t-6 .. t only
    s = samples(
        (T, *SITE, 50_000),
        (T - pd.Timedelta(days=3), *SITE, 900_000),  # too recent for t
    )
    row = row_on(features.build(s, wx(temp=temp)), T)
    assert row.temp_mean == pytest.approx(20.0)
    assert row.temp_trend == pytest.approx(0.0)
    assert np.isnan(row.last_count_log)
    assert list(features.build(s, wx(temp=temp)).columns) == [
        "date",
        "cell_lat",
        "cell_lon",
        "y",
        *features.FEATURES,
    ]


def test_window_is_the_30_days_ending_at_t_minus_7():
    temp = np.arange(
        60, dtype=float
    )  # day index; t is 59, window t-36 .. t-7 = 23 .. 52
    row = row_on(features.build(samples((T, *SITE, 0)), wx(temp=temp)), T)
    assert row.temp_mean == pytest.approx(np.mean(np.arange(23, 53)))
    assert row.temp_trend == pytest.approx(1.0)
    assert row.rain_sum == pytest.approx(30.0)


def test_persistence_uses_latest_within_20km():
    s = samples(
        (T, *SITE, 0),
        (T - pd.Timedelta(days=8), km_north(25), SITE[1], 5_000_000),  # 25 km: ignored
        (
            T - pd.Timedelta(days=10),
            km_north(15),
            SITE[1],
            200_000,
        ),  # 15 km, 10 days old
        (T - pd.Timedelta(days=30), *SITE, 10),  # older than the 15 km one
    )
    row = row_on(features.build(s, wx()), T)
    assert row.last_count_log == pytest.approx(np.log10(1 + 200_000))
    assert row.last_bloom == 1.0


def test_window_with_gaps():
    temp = np.full(60, 20.0)
    window = slice(60 - 37, 60 - 7)  # t-36 .. t-7
    temp_19 = temp.copy()
    temp_19[window] = np.nan
    temp_19[window.start : window.start + 19] = 21.0
    temp_20 = temp_19.copy()
    temp_20[window.start + 19] = 21.0
    s = samples((T, *SITE, 0))
    assert np.isnan(row_on(features.build(s, wx(temp=temp_19)), T).temp_mean)
    assert row_on(features.build(s, wx(temp=temp_20)), T).temp_mean == pytest.approx(
        21.0
    )


def test_upwelling_share():
    dirs = np.full(60, 90.0)
    dirs[-37:-7:2] = 330.0  # every other window day from the NNW
    row = row_on(features.build(samples((T, *SITE, 0)), wx(wind_dir=dirs)), T)
    assert row.upwelling_share == pytest.approx(0.5)


def test_same_cell_same_day_collapses():
    s = samples((T, *SITE, 50_000), (T, SITE[0] + 0.01, SITE[1], 150_000))
    df = features.build(s, wx())
    assert len(df) == 1
    assert bool(df.iloc[0].y) is True


def test_sample_cell_finer_than_weather_cell():
    s = samples((T, *SITE, 0), (T, 27.10, -82.60, 0))  # same 0.5° weather cell
    df = features.build(s, wx())
    assert len(df) == 2
    assert df["temp_mean"].notna().all()


def test_season_features():
    row = row_on(features.build(samples((T, *SITE, 0)), wx()), T)
    assert row.doy_sin**2 + row.doy_cos**2 == pytest.approx(1.0)


def test_split_has_no_shared_dates_and_fixed_bounds():
    dates = pd.date_range("2014-01-01", "2019-12-31", freq="7D")
    df = pd.DataFrame({"date": dates, "y": False})
    s = features.split(df)
    assert s["train"].date.max() <= pd.Timestamp("2015-12-31")
    assert s["val"].date.between("2016-01-01", "2017-12-31").all()
    assert s["test"].date.min() >= pd.Timestamp("2018-01-01")
    assert sum(len(part) for part in s.values()) == len(df)
