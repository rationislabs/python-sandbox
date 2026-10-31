"""One feature row per (cell, sample day), built only from data up to t - 7 days."""

import numpy as np
import pandas as pd

from red_tide import habsos, weather

LEAD_DAYS = 7
WINDOW_DAYS = 30
NEAR_KM = 20
MIN_VALID_DAYS = 20
FEATURES: list[str] = [
    "temp_mean",
    "temp_trend",
    "wind_mean",
    "upwelling_share",
    "rain_sum",
    "doy_sin",
    "doy_cos",
    "last_count_log",
    "last_bloom",
]
SPLITS = {
    "train": ("2000-01-01", "2015-12-31"),
    "val": ("2016-01-01", "2017-12-31"),
    "test": ("2018-01-01", "2100-12-31"),
}
EARTH_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _upwelling(wind_dir: pd.Series) -> pd.Series:
    """1.0 for winds from the NNW (300°–30°), alongshore southward on this coast."""
    up = ((wind_dir >= 300) | (wind_dir <= 30)).astype(float)
    return up.where(wind_dir.notna())


def _window_stats(cell_wx: pd.DataFrame) -> pd.DataFrame:
    """Weather features for each day d from the WINDOW_DAYS days ending at d."""
    daily = cell_wx.set_index("date").sort_index().asfreq("D")
    roll = {"window": WINDOW_DAYS, "min_periods": 1}

    def enough(stat: pd.Series, values: pd.Series) -> pd.Series:
        return stat.where(values.notna().rolling(**roll).sum() >= MIN_VALID_DAYS)

    temp = daily["temp"]
    # Least-squares slope over the valid days, from rolling sums (°C per day).
    x = pd.Series(np.arange(len(daily), dtype=float), index=daily.index)
    valid = temp.notna().astype(float)
    n = valid.rolling(**roll).sum()
    sx = (x * valid).rolling(**roll).sum()
    sy = temp.fillna(0).rolling(**roll).sum()
    sxx = (x * x * valid).rolling(**roll).sum()
    sxy = (x * temp.fillna(0)).rolling(**roll).sum()
    trend = (n * sxy - sx * sy) / (n * sxx - sx**2)

    return pd.DataFrame(
        {
            "temp_mean": enough(temp.rolling(**roll).mean(), temp),
            "temp_trend": enough(trend, temp),
            "wind_mean": enough(
                daily["wind_speed"].rolling(**roll).mean(), daily["wind_speed"]
            ),
            "upwelling_share": enough(
                _upwelling(daily["wind_dir"]).rolling(**roll).mean(), daily["wind_dir"]
            ),
            "rain_sum": enough(daily["rain"].rolling(**roll).sum(), daily["rain"]),
        }
    )


def _weather_features(wx: pd.DataFrame) -> pd.DataFrame:
    """Stats for the window ending at d, keyed by the sample day d + LEAD_DAYS."""
    frames = []
    for (cell_lat, cell_lon), cell_wx in wx.groupby(["cell_lat", "cell_lon"]):
        stats = _window_stats(cell_wx)
        stats.index = stats.index + pd.Timedelta(days=LEAD_DAYS)
        frames.append(
            stats.rename_axis("date")
            .reset_index()
            .assign(cell_lat=cell_lat, cell_lon=cell_lon)
        )
    return pd.concat(frames, ignore_index=True)


def _persistence(targets: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Most recent count within NEAR_KM taken on or before t - LEAD_DAYS, per target row."""
    stations = samples.groupby(["lat", "lon"], as_index=False).size()[["lat", "lon"]]
    stations["station"] = np.arange(len(stations))
    history = (
        samples.merge(stations, on=["lat", "lon"])
        .groupby(["station", "date"], as_index=False)["cells"]
        .max()
    )
    # Every (target, station) pair within NEAR_KM. ponytail: per-cell blocks keep the
    # pair matrix small; a KD-tree would scale further if the region grows.
    pairs = []
    for _, block in targets.groupby(["cell_lat", "cell_lon"]):
        d = _haversine_km(
            block["lat"].to_numpy()[:, None],
            block["lon"].to_numpy()[:, None],
            stations["lat"].to_numpy()[None, :],
            stations["lon"].to_numpy()[None, :],
        )
        ti, si = np.nonzero(d <= NEAR_KM)
        pairs.append(
            pd.DataFrame({"target": block.index.to_numpy()[ti], "station": si})
        )
    pairs = pd.concat(pairs, ignore_index=True)
    pairs["cutoff"] = targets.loc[pairs["target"], "date"].to_numpy() - pd.Timedelta(
        days=LEAD_DAYS
    )
    matched = pd.merge_asof(
        pairs.sort_values("cutoff"),
        history.sort_values("date"),
        left_on="cutoff",
        right_on="date",
        by="station",
        direction="backward",
    ).dropna(subset=["cells"])
    latest = matched.sort_values(["date", "cells"]).groupby("target").tail(1)
    out = pd.DataFrame(
        index=targets.index, columns=["last_count_log", "last_bloom"], dtype=float
    )
    out.loc[latest["target"], "last_count_log"] = np.log10(
        1 + latest["cells"].to_numpy()
    )
    out.loc[latest["target"], "last_bloom"] = (
        habsos.is_bloom(latest["cells"]).astype(float).to_numpy()
    )
    return out


def build(samples: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Columns date, cell_lat, cell_lon, y + FEATURES; y = bloom of the day's max count."""
    cells = [
        weather.cell_of(lat, lon) for lat, lon in zip(samples["lat"], samples["lon"])
    ]
    tagged = samples.assign(
        cell_lat=[c[0] for c in cells], cell_lon=[c[1] for c in cells]
    )
    targets = tagged.groupby(["cell_lat", "cell_lon", "date"], as_index=False).agg(
        lat=("lat", "mean"), lon=("lon", "mean"), cells=("cells", "max")
    )
    targets["y"] = habsos.is_bloom(targets["cells"])
    targets = targets.join(_persistence(targets, samples))
    wx_cells = [
        weather.cell_of(lat, lon, weather.WX_DEG)
        for lat, lon in zip(targets["cell_lat"], targets["cell_lon"])
    ]
    targets["wx_lat"] = [c[0] for c in wx_cells]
    targets["wx_lon"] = [c[1] for c in wx_cells]
    wx_features = _weather_features(wx).rename(
        columns={"cell_lat": "wx_lat", "cell_lon": "wx_lon"}
    )
    targets = targets.merge(wx_features, on=["wx_lat", "wx_lon", "date"], how="left")
    angle = 2 * np.pi * targets["date"].dt.day_of_year / 365.25
    targets["doy_sin"] = np.sin(angle)
    targets["doy_cos"] = np.cos(angle)
    out = targets[["date", "cell_lat", "cell_lon", "y", *FEATURES]]
    return out.sort_values(["date", "cell_lat", "cell_lon"]).reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fixed time split: train to 2015, validate 2016-2017, test from 2018."""
    return {
        name: df[df["date"].between(start, end)].reset_index(drop=True)
        for name, (start, end) in SPLITS.items()
    }
