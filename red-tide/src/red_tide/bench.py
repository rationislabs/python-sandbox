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


def bench_once(
    n_jobs: int,
    n_trials: int,
    seed: int,
    features_csv: Path,
    params: list[dict] | None = None,
) -> dict:
    from red_tide import features, models

    parts = features.split(pd.read_csv(features_csv, parse_dates=["date"]))
    start = time.perf_counter()
    models.tune(
        "hgb", parts["train"], parts["val"], n_trials, n_jobs, seed, enqueue=params
    )
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
    n_trials: int = 32,
    seed: int = 0,
    repeats: int = 3,
    out_dir: Path = Path("results"),
) -> pd.DataFrame:
    """Every cell runs the same pre-drawn parameter sets, so only threads and the GIL differ.

    GIL modes alternate and the whole grid repeats, so heat and background load hit both
    modes alike; bench.csv holds the medians, bench_runs.csv every run.
    """
    from red_tide import models

    out_dir.mkdir(parents=True, exist_ok=True)
    params_file = out_dir / "bench_params.json"
    params_file.write_text(json.dumps(models.sample_params("hgb", n_trials, seed)))
    env = {**os.environ, **ONE_THREAD}
    rows = []
    for repeat in range(repeats):
        for n in threads:
            for gil in (0, 1):
                cmd = [sys.executable, "-X", f"gil={gil}", "-m", "red_tide.bench"]
                cmd += ["--n-jobs", str(n), "--n-trials", str(n_trials)]
                cmd += ["--seed", str(seed), "--features", str(features_csv)]
                cmd += ["--params", str(params_file)]
                out = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, env=env
                )
                row = json.loads(out.stdout) | {"repeat": repeat}
                mode = "on" if row["gil"] else "off"
                print(f"run {repeat + 1}, GIL {mode}, {n} threads: {row['seconds']} s")
                rows.append(row)
    runs = pd.DataFrame(rows).assign(cpu=cpu_name(), cores=os.cpu_count())
    runs.to_csv(out_dir / "bench_runs.csv", index=False)
    summary = (
        runs.groupby(["gil", "n_jobs"], as_index=False)
        .agg(
            n_trials=("n_trials", "first"),
            runs=("seconds", "size"),
            seconds=("seconds", "median"),
        )
        .assign(cpu=cpu_name(), cores=os.cpu_count())
    )
    summary["trials_per_s"] = (summary["n_trials"] / summary["seconds"]).round(3)
    base = summary[summary["n_jobs"] == min(threads)].set_index("gil")["seconds"]
    summary["speedup"] = (summary["gil"].map(base) / summary["seconds"]).round(2)
    summary.to_csv(out_dir / "bench.csv", index=False)
    chart(summary, out_dir / "bench.png")
    return runs


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
    ax.set_title(
        f"Optuna + HistGradientBoosting, the same {df['n_trials'].iloc[0]} trials"
        f" (median of {df['runs'].iloc[0]})"
    )
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
    parser.add_argument(
        "--params", type=Path, help="JSON list of parameter sets to run"
    )
    args = parser.parse_args()
    params = json.loads(args.params.read_text()) if args.params else None
    row = bench_once(args.n_jobs, args.n_trials, args.seed, args.features, params)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
