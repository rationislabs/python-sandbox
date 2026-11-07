"""Time every variant with the GIL off (in this process) and on (a child with PYTHON_GIL=1)."""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from kmeans_threads import variants
from kmeans_threads.kmeans import ITERS, RUNS, SEED, K, agree, kmeanspp, make_data

COLUMNS = [
    "variant",
    "workers",
    "gil",
    "seconds",
    "startup_seconds",
    "speedup",
    "runs",
    "points",
    "iters",
    "cpu",
    "cores",
    "python",
    "numpy",
]
WORKERS = tuple(w for w in (1, 2, 4, 8) if w <= (os.process_cpu_count() or 1))
__all__ = ["COLUMNS", "RUNS", "WORKERS", "bench", "cpu_name", "run_half", "table"]


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_half(points: int, workers: Sequence[int], runs: int) -> list[dict]:
    """One row per variant and worker count, in the current GIL mode only."""
    data = make_data(points, SEED)
    init = kmeanspp(data.points, K, SEED)
    ref = None
    rows = []
    for name in variants.VARIANTS:
        fit = variants.VARIANTS[name]
        for w in workers if name in variants.PARALLEL else (1,):
            times, startups, gil_on = [], [], False
            for _ in range(runs):
                t0 = perf_counter()
                with variants.open_pool(name, w, data) as pool:
                    startup = perf_counter() - t0 if pool is not None else 0.0
                    t1 = perf_counter()
                    centres, labels = fit(data, init, ITERS, w, pool)
                    times.append(perf_counter() - t1)
                startups.append(startup)
                gil_on |= sys._is_gil_enabled()
                if ref is None:
                    ref = (centres, labels)
                elif not (agree(centres, ref[0]) and labels == ref[1]):
                    raise SystemExit(
                        f"{name} with {w} workers disagrees with loop; no results written"
                    )
            seconds = statistics.median(times)
            rows.append(
                {
                    "variant": name,
                    "workers": w,
                    "gil": "on" if gil_on else "off",
                    "seconds": round(seconds, 3),
                    "startup_seconds": round(statistics.median(startups), 3),
                    "speedup": round(rows[0]["seconds"] / seconds, 2) if rows else 1.0,
                    "runs": runs,
                    "points": points,
                    "iters": ITERS,
                    "cpu": cpu_name(),
                    "cores": os.process_cpu_count(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                }
            )
            print(f"{name} {w} workers: {seconds:.3f} s", file=sys.stderr, flush=True)
    return rows


def bench(
    points: int, out: Path, workers: Sequence[int] = WORKERS, runs: int = RUNS
) -> list[dict]:
    """Both halves; `out` is written only when both are clean."""
    off = run_half(points, workers, runs)
    if any(r["gil"] == "on" for r in off):
        raise SystemExit(
            "GIL is on in the GIL-off half (an extension module turned it back on); "
            "no results written"
        )
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "kmeans_threads.bench",
            "--points",
            str(points),
            "--runs",
            str(runs),
            "--workers",
            ",".join(map(str, workers)),
        ],
        env={**os.environ, "PYTHON_GIL": "1"},
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    on = json.loads(child.stdout)
    if any(r["gil"] == "off" for r in on):
        raise SystemExit("PYTHON_GIL=1 did not turn the GIL on; no results written")
    rows = off + on
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def table(rows: list[dict]) -> str:
    lines = [
        "| variant | workers | GIL | seconds | pool start-up s | speedup vs loop |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['variant']} | {r['workers']} | {r['gil']} | {r['seconds']:.3f} "
        f"| {r['startup_seconds']:.3f} | {r['speedup']:.2f} |"
        for r in rows
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=int, required=True)
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--workers", required=True)
    a = p.parse_args()
    ws = [int(w) for w in a.workers.split(",")]
    print(json.dumps(run_half(a.points, ws, a.runs)))
