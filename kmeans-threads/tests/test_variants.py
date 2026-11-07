import pytest

from kmeans_threads import kmeans, variants

CASES = [
    (n, w)
    for n in variants.VARIANTS
    for w in ((1, 2, 3, 4) if n in variants.PARALLEL else (1,))
]


@pytest.fixture(scope="module")
def small():
    data = kmeans.make_data(kmeans.CHECK_POINTS, seed=0)
    init = kmeans.kmeanspp(data.points, kmeans.K, seed=0)
    return data, init, variants.fit_loop(data, init, kmeans.ITERS, 1, None)


def run(name, data, init, iters, workers):
    with variants.open_pool(name, workers, data) as pool:
        return variants.VARIANTS[name](data, init, iters, workers, pool)


def test_loop_moves_the_centres(small):
    _data, init, (centres, labels) = small
    assert centres != init and len(labels) == kmeans.CHECK_POINTS


@pytest.mark.parametrize(("name", "workers"), CASES)
def test_matches_loop(small, name, workers):
    data, init, (ref_c, ref_l) = small
    centres, labels = run(name, data, init, kmeans.ITERS, workers)
    assert kmeans.agree(centres, ref_c) and labels == ref_l


@pytest.mark.parametrize(("name", "workers"), CASES)
def test_empty_cluster_keeps_its_centre(name, workers):
    data = kmeans.make_data(40, seed=3)
    far = (1000.0,) * kmeans.DIMS
    init = [data.points[0], data.points[1], far]
    centres, labels = run(name, data, init, 3, workers)
    assert centres[2] == far and 2 not in labels


@pytest.mark.parametrize("name", sorted(variants.PARALLEL))
def test_more_workers_than_points(name):
    data = kmeans.make_data(3, seed=0)
    init = list(data.points)
    ref_c, ref_l = variants.fit_loop(data, init, 2, 1, None)
    centres, labels = run(name, data, init, 2, 4)
    assert kmeans.agree(centres, ref_c) and labels == ref_l
