"""One k-means, five ways. Every variant has the same signature and returns the same result."""

import multiprocessing
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from kmeans_threads.kmeans import (
    Centres,
    Data,
    Part,
    Point,
    chunk_sums,
    chunks,
    np_chunk_sums,
    np_update,
    update,
)

type Result = tuple[Centres, list[int]]
type Fit = Callable[[Data, Centres, int, int, Executor | None], Result]


def _reduce(
    parts: list[Part], k: int, dims: int
) -> tuple[list[list[float]], list[int], list[int]]:
    """Add partial sums and counts in chunk order; concatenate labels in chunk order."""
    sums = [[0.0] * dims for _ in range(k)]
    counts = [0] * k
    labels: list[int] = []
    for s, n, lab in parts:
        for j in range(k):
            counts[j] += n[j]
            row = sums[j]
            for t in range(dims):
                row[t] += s[j][t]
        if lab is not None:
            labels.extend(lab)
    return sums, counts, labels


def _pure(
    run: Callable[[Centres, bool], list[Part]], init: Centres, iters: int
) -> Result:
    centres, labels = list(init), []
    for it in range(iters):
        last = it == iters - 1
        sums, counts, lab = _reduce(run(centres, last), len(init), len(init[0]))
        centres = update(sums, counts, centres)
        if last:
            labels = lab
    return centres, labels


def _numpy(
    run: Callable[
        [np.ndarray, bool], list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]
    ],
    init: Centres,
    iters: int,
) -> Result:
    centres = np.array(init, dtype=np.float64)
    labels: list[int] = []
    for it in range(iters):
        last = it == iters - 1
        parts = run(centres, last)
        sums, counts = parts[0][0].copy(), parts[0][1].copy()
        for s, n, _ in parts[1:]:
            sums += s
            counts += n
        centres = np_update(sums, counts, centres)
        if last:
            labels = [int(v) for p in parts for v in p[2].tolist()]
    return [tuple(r) for r in centres.tolist()], labels


def fit_loop(
    data: Data, init: Centres, iters: int, workers: int, pool: Executor | None
) -> Result:
    n = len(data.points)
    return _pure(lambda c, last: [chunk_sums(data.points, c, 0, n, last)], init, iters)


def fit_threads(
    data: Data, init: Centres, iters: int, workers: int, pool: Executor | None
) -> Result:
    assert pool is not None
    bounds = chunks(len(data.points), workers)

    def run(c: Centres, last: bool) -> list[Part]:
        return list(
            pool.map(lambda b: chunk_sums(data.points, c, b[0], b[1], last), bounds)
        )

    return _pure(run, init, iters)


_POINTS: list[Point] = []
_WORKERS = 1
_BARRIER = None


def _init(points: list[Point], workers: int, barrier) -> None:
    global _POINTS, _WORKERS, _BARRIER
    _POINTS, _WORKERS, _BARRIER = points, workers, barrier


def _wait(_: int) -> None:
    _BARRIER.wait()


def _task(job: tuple[int, Centres, bool]) -> Part:
    index, centres, last = job
    lo, hi = chunks(len(_POINTS), _WORKERS)[index]
    return chunk_sums(_POINTS, centres, lo, hi, last)


def fit_processes(
    data: Data, init: Centres, iters: int, workers: int, pool: Executor | None
) -> Result:
    """Each worker already holds every point (pool initializer); a task is a chunk index plus
    the centres, because the pool cannot send a task to a chosen worker."""
    assert pool is not None
    return _pure(
        lambda c, last: list(pool.map(_task, [(i, c, last) for i in range(workers)])),
        init,
        iters,
    )


def fit_numpy(
    data: Data, init: Centres, iters: int, workers: int, pool: Executor | None
) -> Result:
    return _numpy(lambda c, last: [np_chunk_sums(data.array, c, last)], init, iters)


def fit_numpy_threads(
    data: Data, init: Centres, iters: int, workers: int, pool: Executor | None
) -> Result:
    assert pool is not None
    bounds = chunks(len(data.points), workers)

    def run(c: np.ndarray, last: bool):
        return list(
            pool.map(lambda b: np_chunk_sums(data.array[b[0] : b[1]], c, last), bounds)
        )

    return _numpy(run, init, iters)


VARIANTS: dict[str, Fit] = {
    "loop": fit_loop,
    "threads": fit_threads,
    "processes": fit_processes,
    "numpy": fit_numpy,
    "numpy-threads": fit_numpy_threads,
}
PARALLEL = frozenset({"threads", "processes", "numpy-threads"})


@contextmanager
def open_pool(name: str, workers: int, data: Data) -> Iterator[Executor | None]:
    """A pool whose workers are all running (and, for processes, hold the data), or None."""
    if name not in PARALLEL:
        yield None
        return
    if name == "processes":
        ctx = multiprocessing.get_context("forkserver")
        pool: Executor = ProcessPoolExecutor(
            workers,
            mp_context=ctx,
            initializer=_init,
            initargs=(data.points, workers, ctx.Barrier(workers)),
        )
        warm = _wait
    else:
        pool = ThreadPoolExecutor(workers)
        barrier = threading.Barrier(workers)

        def warm(_: int) -> None:
            barrier.wait()

    with pool:
        list(pool.map(warm, range(workers)))
        yield pool
