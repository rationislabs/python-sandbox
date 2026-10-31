"""Daily weather per 0.25° cell from the Open-Meteo historical archive (ERA5, CC BY 4.0)."""

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CELL_DEG = 0.25  # sample cells
# Weather cells: 25 years of daily data counts as hundreds of calls against Open-Meteo's
# free limits, so the weather grid is coarser (20 cells over the region, not 63).
WX_DEG = 0.5
PAUSE_S = (
    200  # between uncached requests: about 18 an hour stays under the hourly limit
)
RATE_LIMIT_WAIT_S = 65
RATE_LIMIT_TRIES = 60
DAILY = {
    "temperature_2m_mean": "temp",
    "wind_speed_10m_mean": "wind_speed",
    "wind_direction_10m_dominant": "wind_dir",
    "precipitation_sum": "rain",
}


def cell_of(lat: float, lon: float, deg: float = CELL_DEG) -> tuple[float, float]:
    """Center of the deg-sized cell holding the point."""

    def center(x: float) -> float:
        return math.floor(x / deg) * deg + deg / 2

    return center(lat), center(lon)


def _request(cell: tuple[float, float], start: date, end: date) -> dict:
    query = urllib.parse.urlencode(
        {
            "latitude": cell[0],
            "longitude": cell[1],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(DAILY),
            "timezone": "GMT",
        }
    )
    failures = rate_limited = 0
    while True:
        try:
            with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 and rate_limited < RATE_LIMIT_TRIES:
                rate_limited += 1
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
            failures += 1
            error = e
        except (OSError, ValueError) as e:
            failures += 1
            error = e
        if failures == 3:
            raise RuntimeError(f"Open-Meteo failed for cell {cell}: {error}") from error
        time.sleep(2**failures)


def fetch_cell(
    cell: tuple[float, float], start: date, end: date, cache_dir: Path
) -> pd.DataFrame:
    """date, temp, wind_speed, wind_dir, rain for one cell; nulls stay NaN.

    One request per cell for the whole range, cached as <lat>_<lon>.csv.
    """
    # ponytail: the cache key is the cell only; delete the cache to change the range.
    path = cache_dir / f"{cell[0]}_{cell[1]}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    daily = _request(cell, start, end)["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_name, column in DAILY.items():
        df[column] = pd.to_numeric(pd.Series(daily[api_name], dtype=object))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    time.sleep(PAUSE_S)  # stay polite to the free API between uncached requests
    return df


def load_all(
    cells: Iterable[tuple[float, float]], start: date, end: date, cache_dir: Path
) -> pd.DataFrame:
    frames = [
        fetch_cell(cell, start, end, cache_dir).assign(
            cell_lat=cell[0], cell_lon=cell[1]
        )
        for cell in cells
    ]
    return pd.concat(frames, ignore_index=True)
