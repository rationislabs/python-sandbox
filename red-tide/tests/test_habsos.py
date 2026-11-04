from pathlib import Path

import pandas as pd

from red_tide import habsos

FIXTURE = Path(__file__).parent / "fixtures" / "habsos.csv"


def test_threshold():
    assert list(habsos.is_bloom(pd.Series([99_999, 100_000]))) == [False, True]


def test_load_filters_region_and_years():
    df = habsos.load(FIXTURE)
    assert df["date"].min() >= pd.Timestamp("2000-01-01")
    assert df["lat"].between(25.8, 28.0).all()
    assert df["lon"].between(-83.5, -81.7).all()


def test_load_skips_bad_counts():
    df = habsos.load(FIXTURE)
    assert df["cells"].notna().all()
    assert len(df) == 16  # 25 rows: 2 bad counts, 1 from 1999, 6 outside the region
    assert list(df.columns) == ["date", "lat", "lon", "cells"]


def test_load_reports_unreadable_date_or_position(tmp_path, capsys):
    raw = pd.read_csv(FIXTURE, dtype=str)
    inside = raw[raw["LATITUDE"].astype(float).between(25.8, 28.0)].iloc[0]
    bad = pd.DataFrame([inside, inside, inside])
    bad.iloc[0, bad.columns.get_loc("SAMPLE_DATE")] = "not a date"
    bad.iloc[1, bad.columns.get_loc("LATITUDE")] = ""
    bad.iloc[2, bad.columns.get_loc("LONGITUDE")] = "x"
    path = tmp_path / "habsos.csv"
    pd.concat([raw, bad]).to_csv(path, index=False)
    assert len(habsos.load(path)) == 16
    assert (
        "skipped 3 rows with an unreadable date or position" in capsys.readouterr().err
    )
