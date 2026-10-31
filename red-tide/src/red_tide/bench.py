"""Does Optuna tuning get faster with threads when the GIL is off?

Each (threads, GIL mode) cell runs in its own interpreter: the GIL cannot change inside a running
process. scikit-learn's own OpenMP/BLAS threads are pinned to 1 so only Optuna's threads count.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ONE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
# The eager run turns every lazy import off with a filter (3.15 has no "-X lazy_imports=none").
EAGER = (
    "import runpy, sys; sys.set_lazy_imports_filter(lambda *a: False); "
    "sys.argv = ['bloom', '--help']; runpy.run_module('red_tide.cli', run_name='__main__')"
)


def bench_once(n_jobs: int, n_trials: int, seed: int, features_csv: Path) -> dict:
    from red_tide import features, models

    parts = features.split(pd.read_csv(features_csv, parse_dates=["date"]))
    start = time.perf_counter()
    models.tune("hgb", parts["train"], parts["val"], n_trials, n_jobs, seed)
    seconds = time.perf_counter() - start
    return {
        "gil": sys._is_gil_enabled(),
        "n_jobs": n_jobs,
        "n_trials": n_trials,
        "seconds": round(seconds, 3),
        "trials_per_s": round(n_trials / seconds, 3),
    }


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_grid(
    features_csv: Path,
    threads=(1, 2, 4, 8),
    n_trials: int = 64,
    seed: int = 0,
    out_dir: Path = Path("results"),
) -> pd.DataFrame:
    env = {**os.environ, **ONE_THREAD}
    rows = []
    for gil in (0, 1):
        for n in threads:
            cmd = [sys.executable, "-X", f"gil={gil}", "-m", "red_tide.bench"]
            cmd += [
                "--n-jobs",
                str(n),
                "--n-trials",
                str(n_trials),
                "--seed",
                str(seed),
            ]
            cmd += ["--features", str(features_csv)]
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env
            )
            row = json.loads(out.stdout)
            print(
                f"GIL {'on' if row['gil'] else 'off'}, {n} threads: {row['seconds']} s"
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    base = df[df["n_jobs"] == min(threads)].set_index("gil")["seconds"]
    df["speedup"] = (df["gil"].map(base) / df["seconds"]).round(2)
    df["cpu"] = cpu_name()
    df["cores"] = os.cpu_count()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bench.csv", index=False)
    chart(df, out_dir / "bench.png")
    return df


def chart(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for gil, label, color in (
        (False, "GIL off (3.15t)", "#2f6fdb"),
        (True, "GIL on (-X gil=1)", "#c2571a"),
    ):
        part = df[df["gil"] == gil].sort_values("n_jobs")
        ax.plot(part["n_jobs"], part["speedup"], marker="o", label=label, color=color)
    threads = sorted(df["n_jobs"].unique())
    ax.plot(threads, threads, linestyle=":", color="#999999", label="perfect scaling")
    ax.set_xticks(
        threads,
        [f"{n}\n(SMT)" if n > (os.cpu_count() or 8) // 2 else str(n) for n in threads],
    )
    ax.set_xlabel("Optuna threads (n_jobs)")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title(f"Optuna + HistGradientBoosting, {df['n_trials'].iloc[0]} trials")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def startup_times(runs: int = 15, out_dir: Path = Path("results")) -> dict:
    """Median seconds of `bloom --help` with lazy imports (default) and with them all eager."""
    commands = {
        "lazy": [sys.executable, "-m", "red_tide.cli", "--help"],
        "eager": [sys.executable, "-c", EAGER],
    }
    times = {}
    for name, cmd in commands.items():
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            subprocess.run(cmd, capture_output=True, check=True)
            samples.append(time.perf_counter() - start)
        times[name] = round(statistics.median(samples), 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"mode": k, "median_seconds": v, "runs": runs} for k, v in times.items()]
    ).to_csv(out_dir / "startup.csv", index=False)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One benchmark cell; prints one JSON line."
    )
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bench_once(args.n_jobs, args.n_trials, args.seed, args.features)))


if __name__ == "__main__":
    main()
