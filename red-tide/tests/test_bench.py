import json
import os
import subprocess
import sys
from pathlib import Path

from red_tide import bench

FIXTURE = Path(__file__).parent / "fixtures" / "features.csv"


def test_bench_once_in_child_on_fixture():
    env = {**os.environ, **bench.ONE_THREAD}
    out = subprocess.run(
        [sys.executable, "-X", "gil=0", "-m", "red_tide.bench", "--n-jobs", "2"]
        + ["--n-trials", "4", "--seed", "1", "--features", str(FIXTURE)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    row = json.loads(out.stdout)
    assert row["gil"] is False
    assert row["n_jobs"] == 2 and row["n_trials"] == 4
    assert set(row) >= {"seconds", "trials_per_s"}


def test_grid_columns(tmp_path):
    df = bench.run_grid(FIXTURE, threads=(1, 2), n_trials=2, seed=0, out_dir=tmp_path)
    assert len(df) == 4
    assert set(df.columns) >= {
        "gil",
        "n_jobs",
        "seconds",
        "trials_per_s",
        "speedup",
        "cpu",
        "cores",
    }
    assert sorted(df["gil"].unique()) == [False, True]
    assert (tmp_path / "bench.csv").exists() and (tmp_path / "bench.png").exists()


def test_startup_times_keys(tmp_path):
    times = bench.startup_times(runs=1, out_dir=tmp_path)
    assert set(times) == {"lazy", "eager"}
    assert times["lazy"] > 0 and times["eager"] > 0
    assert (tmp_path / "startup.csv").exists()
