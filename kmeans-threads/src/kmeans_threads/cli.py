"""`kmeans bench` and `kmeans check`."""

import argparse
import os
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kmeans_threads import bench, variants
from kmeans_threads.kmeans import (
    CHECK_POINTS,
    ITERS,
    POINTS,
    SEED,
    K,
    agree,
    kmeanspp,
    make_data,
)

ROOT = Path(__file__).resolve().parents[2]


def points_arg(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {s}") from None
    if n < K:
        raise argparse.ArgumentTypeError(f"must be at least {K}")
    return n


def check(points: int = CHECK_POINTS, workers: Sequence[int] | None = None) -> bool:
    """Every variant and worker count against loop; prints one line per run."""
    workers = bench.WORKERS if workers is None else workers
    data = make_data(points, SEED)
    init = kmeanspp(data.points, K, SEED)
    ref_c, ref_l = variants.fit_loop(data, init, ITERS, 1, None)
    runs, bad = 0, 0
    for name, fit in variants.VARIANTS.items():
        for w in workers if name in variants.PARALLEL else (1,):
            with variants.open_pool(name, w, data) as pool:
                centres, labels = fit(data, init, ITERS, w, pool)
            ok = agree(centres, ref_c) and labels == ref_l
            print(f"{name} {w} {'ok' if ok else 'DIFFERS'}")
            runs, bad = runs + 1, bad + (not ok)
    if bad:
        print(f"{bad} of {runs} runs differ from loop")
        return False
    print(f"all {runs} runs agree with loop (relative 1e-9)")
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kmeans", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "bench", help="time every variant, GIL off then on; writes results.csv"
    )
    b.add_argument("--points", type=points_arg, default=POINTS)
    sub.add_parser("check", help="every variant agrees with loop on a small data set")
    args = p.parse_args(argv)
    if args.cmd == "check":
        if not check():
            raise SystemExit(1)
        return
    print(
        f"{bench.cpu_name()}, {os.process_cpu_count()} CPUs, "
        f"Python {platform.python_version()}, numpy {np.__version__}"
    )
    rows = bench.bench(args.points, ROOT / "results.csv", bench.WORKERS, bench.RUNS)
    print(bench.table(rows))
    print("wrote results.csv")
