"""Karenia brevis cell counts from HABSOS (NOAA NCEI accession 0120767, public domain)."""

import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HABSOS_URL = "https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv"
BLOOM_CELLS = 100_000
# Southwest Florida shelf, Tampa Bay to Naples: lat_min, lat_max, lon_min, lon_max.
REGION = (25.8, 28.0, -83.5, -81.7)
FIRST_DATE = pd.Timestamp("2000-01-01")


def is_bloom(cells: pd.Series) -> pd.Series:
    return cells >= BLOOM_CELLS


def load(path: Path) -> pd.DataFrame:
    """Samples in REGION since 2000 as date, lat, lon, cells; rows without a numeric count are dropped."""
    raw = pd.read_csv(
        path,
        usecols=["LATITUDE", "LONGITUDE", "SAMPLE_DATE", "CELLCOUNT"],
        low_memory=False,
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                raw["SAMPLE_DATE"], format="%Y/%m/%d", errors="coerce"
            ),
            "lat": pd.to_numeric(raw["LATITUDE"], errors="coerce"),
            "lon": pd.to_numeric(raw["LONGITUDE"], errors="coerce"),
            "cells": pd.to_numeric(raw["CELLCOUNT"], errors="coerce"),
        }
    )
    lat_min, lat_max, lon_min, lon_max = REGION
    df = df[
        df["lat"].between(lat_min, lat_max)
        & df["lon"].between(lon_min, lon_max)
        & (df["date"] >= FIRST_DATE)
    ]
    bad = df["cells"].isna() | (df["cells"] < 0)
    if bad.any():
        print(
            f"habsos: skipped {bad.sum()} rows without a usable cell count",
            file=sys.stderr,
        )
    return df[~bad].reset_index(drop=True)


def download(dest: Path) -> Path:
    """Fetch the HABSOS CSV to dest (3 tries, backoff 2/4/8 s), written atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(HABSOS_URL, timeout=120) as resp:
                tmp.write_bytes(resp.read())
            tmp.replace(dest)
            return dest
        except OSError as e:
            if attempt == 2:
                raise RuntimeError(f"HABSOS download failed: {e}") from e
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")
