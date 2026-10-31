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
