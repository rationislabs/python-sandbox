import sys

import numpy as np
import pytest

from kmeans_threads import kmeans


def test_runs_without_the_gil():
    # numpy is imported by now; it must not turn the GIL back on.
    assert not sys._is_gil_enabled()


def test_blobs_are_seeded():
    a = kmeans.blobs(50, seed=1)
    assert a == kmeans.blobs(50, seed=1) and a != kmeans.blobs(50, seed=2)
    assert len(a) == 50 and all(len(p) == kmeans.DIMS for p in a)


def test_make_data_array_matches_points():
    d = kmeans.make_data(20, seed=0)
    assert d.array.shape == (20, 4) and d.array.dtype == np.float64
    assert d.array.tolist() == [list(p) for p in d.points]


def test_kmeanspp_fixed_seed_fixed_centres():
    pts = kmeans.blobs(500, seed=0)
    a = kmeans.kmeanspp(pts, 8, seed=0)
    assert a == kmeans.kmeanspp(pts, 8, seed=0)
    assert a != kmeans.kmeanspp(pts, 8, seed=1)
    assert len(set(a)) == 8 and set(a) <= set(pts)


def test_kmeanspp_second_centre_is_the_other_point():
    lo, hi = (0.0,) * 4, (5.0,) * 4
    for s in range(5):
        assert sorted(kmeans.kmeanspp([lo, hi], 2, seed=s)) == [lo, hi]


def test_kmeanspp_needs_k_distinct_points():
    with pytest.raises(ValueError, match="at least 2 distinct points"):
        kmeans.kmeanspp([(1.0,) * 4] * 10, 2, seed=0)


def test_chunks():
    assert kmeans.chunks(10, 4) == [(0, 2), (2, 5), (5, 7), (7, 10)]
    assert kmeans.chunks(3, 4) == [(0, 0), (0, 1), (1, 2), (2, 3)]


def test_chunk_sums_by_hand():
    pts = [(0.0, 0, 0, 0), (1.0, 0, 0, 0), (10.0, 0, 0, 0)]
    c = [(0.0, 0, 0, 0), (10.0, 0, 0, 0)]
    assert kmeans.chunk_sums(pts, c, 0, 3, labels=True) == (
        [[1.0, 0, 0, 0], [10.0, 0, 0, 0]],
        [2, 1],
        [0, 0, 1],
    )
    assert kmeans.chunk_sums(pts, c, 1, 3, labels=False) == (
        [[1.0, 0, 0, 0], [10.0, 0, 0, 0]],
        [1, 1],
        None,
    )


def test_tie_goes_to_lowest_index():
    c = [(0.0, 0, 0, 0), (10.0, 0, 0, 0)]
    assert kmeans.chunk_sums([(5.0, 0, 0, 0)], c, 0, 1, labels=True)[2] == [0]
    got = kmeans.np_chunk_sums(np.array([[5.0, 0, 0, 0]]), np.array(c), labels=True)
    assert got[2].tolist() == [0]


def test_np_chunk_sums_matches_chunk_sums():
    d = kmeans.make_data(500, seed=0)
    c = kmeans.kmeanspp(d.points, 8, seed=0)
    s, n, lab = kmeans.chunk_sums(d.points, c, 0, 500, labels=True)
    ns, nn, nlab = kmeans.np_chunk_sums(d.array, np.array(c), labels=True)
    assert nn.tolist() == n and nlab.tolist() == lab
    assert np.allclose(ns, s, rtol=1e-12, atol=0)


def test_update_keeps_empty_centre():
    old = [(9.0, 9, 9, 9), (7.0, 7, 7, 7)]
    assert kmeans.update([[2.0, 4, 6, 8], [0.0, 0, 0, 0]], [2, 0], old) == [
        (1.0, 2, 3, 4),
        old[1],
    ]
    got = kmeans.np_update(
        np.array([[2.0, 4, 6, 8], [0, 0, 0, 0]]), np.array([2, 0]), np.array(old)
    )
    assert got.tolist() == [[1.0, 2, 3, 4], [7.0, 7, 7, 7]]


def test_agree():
    assert kmeans.agree([(1.0, 2.0)], [(1.0 + 1e-12, 2.0)])
    assert not kmeans.agree([(1.0, 2.0)], [(1.0 + 1e-6, 2.0)])
    # abs floor for near-zero coordinates
    assert kmeans.agree([(0.0, 1.0)], [(1e-13, 1.0)])
    assert not kmeans.agree([(1.0, 2.0)], [(1.0, 2.0), (3.0, 4.0)])
