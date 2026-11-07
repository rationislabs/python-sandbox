"""The k-means math every variant shares: data, k-means++, chunks, partial sums, update."""

import math
import random
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

POINTS = 200_000
DIMS = 4
K = 8
ITERS = 10
SEED = 0
RUNS = 3
CHECK_POINTS = 2_000
REL_TOL = 1e-9

type Point = tuple[float, ...]
type Centres = list[Point]
# per-centre sums, per-centre counts, labels (only when asked for)
type Part = tuple[list[list[float]], list[int], list[int] | None]


class Data(NamedTuple):
    points: list[Point]
    array: np.ndarray  # float64, the same values


def blobs(n_points: int, dims: int = DIMS, k: int = K, seed: int = SEED) -> list[Point]:
    """Points around k random centres, unit Gaussian noise."""
    rng = random.Random(seed)
    true = [tuple(rng.uniform(-10, 10) for _ in range(dims)) for _ in range(k)]
    out = []
    for _ in range(n_points):
        c = true[rng.randrange(k)]
        out.append(tuple(x + rng.gauss(0.0, 1.0) for x in c))
    return out


def make_data(n_points: int, seed: int = SEED) -> Data:
    points = blobs(n_points, seed=seed)
    return Data(points, np.array(points, dtype=np.float64))


def _d2(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def kmeanspp(points: list[Point], k: int, seed: int) -> Centres:
    """k-means++ with a fixed sequence of random.Random calls, so a seed gives fixed centres."""
    m = len(set(points))
    if m < k:
        raise ValueError(f"k-means++ needs at least {k} distinct points, got {m}")
    rng = random.Random(seed)
    n = len(points)
    centres = [points[rng.randrange(n)]]
    d2 = [_d2(p, centres[0]) for p in points]
    while len(centres) < k:
        c = points[rng.choices(range(n), weights=d2)[0]]
        centres.append(c)
        d2 = [min(d, _d2(p, c)) for d, p in zip(d2, points, strict=True)]
    return centres


def chunks(n: int, workers: int) -> list[tuple[int, int]]:
    b = [n * i // workers for i in range(workers + 1)]
    return [(b[i], b[i + 1]) for i in range(workers)]


def chunk_sums(
    points: Sequence[Point], centres: Sequence[Point], lo: int, hi: int, labels: bool
) -> Part:
    """Assign points[lo:hi] to their nearest centre; ties keep the lower index."""
    k, dims = len(centres), len(centres[0])
    sums = [[0.0] * dims for _ in range(k)]
    counts = [0] * k
    lab: list[int] | None = [] if labels else None
    for i in range(lo, hi):
        p = points[i]
        best, best_d = 0, math.inf
        for j, c in enumerate(centres):
            d = 0.0
            for x, y in zip(p, c, strict=True):
                d += (x - y) * (x - y)
            if d < best_d:
                best, best_d = j, d
        s = sums[best]
        for t in range(dims):
            s[t] += p[t]
        counts[best] += 1
        if lab is not None:
            lab.append(best)
    return sums, counts, lab


def np_chunk_sums(
    x: np.ndarray, centres: np.ndarray, labels: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """numpy version of chunk_sums. Broadcasting, not @, so BLAS threads never take part."""
    k = len(centres)
    dist = ((x[:, None, :] - centres[None]) ** 2).sum(axis=2)
    lab = dist.argmin(axis=1)
    counts = np.bincount(lab, minlength=k)
    sums = np.zeros_like(centres)
    for j in range(k):
        sums[j] = x[lab == j].sum(axis=0)
    return sums, counts, lab if labels else None


def update(sums: list[list[float]], counts: list[int], old: Centres) -> Centres:
    """Mean of each centre's points; a centre with no points keeps its old position."""
    return [
        tuple(v / n for v in s) if n else o
        for s, n, o in zip(sums, counts, old, strict=True)
    ]


def np_update(sums: np.ndarray, counts: np.ndarray, old: np.ndarray) -> np.ndarray:
    out = old.copy()
    has = counts > 0
    out[has] = sums[has] / counts[has, None]
    return out


def agree(a: Centres, b: Centres) -> bool:
    return len(a) == len(b) and all(
        len(p) == len(q)
        and all(
            math.isclose(x, y, rel_tol=REL_TOL, abs_tol=1e-12)
            for x, y in zip(p, q, strict=True)
        )
        for p, q in zip(a, b, strict=True)
    )
