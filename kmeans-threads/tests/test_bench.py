import csv
import sys

import pytest

from kmeans_threads import bench, variants


def test_half_rows():
    rows = bench.run_half(300, (1, 2), 1)
    assert [(r["variant"], r["workers"]) for r in rows] == [
        ("loop", 1),
        ("threads", 1),
        ("threads", 2),
        ("processes", 1),
        ("processes", 2),
        ("numpy", 1),
        ("numpy-threads", 1),
        ("numpy-threads", 2),
    ]
    assert {r["gil"] for r in rows} == {"off"}
    assert all(
        r["startup_seconds"] == 0.0 for r in rows if r["variant"] in ("loop", "numpy")
    )
    assert rows[0]["speedup"] == 1.0


def test_bench_writes_both_halves(tmp_path):
    rows = bench.bench(300, tmp_path / "results.csv", workers=(1, 2), runs=1)
    assert len(rows) == 16
    assert {r["gil"] for r in rows[:8]} == {"off"} and {r["gil"] for r in rows[8:]} == {
        "on"
    }
    with open(tmp_path / "results.csv", newline="") as f:
        assert next(csv.reader(f)) == bench.COLUMNS


def test_refuses_gil_off_label_when_gil_is_on(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True)
    with pytest.raises(SystemExit, match="GIL"):
        bench.bench(300, tmp_path / "r.csv", workers=(1,), runs=1)
    assert not (tmp_path / "r.csv").exists()


def test_disagreement_aborts(monkeypatch, tmp_path):
    real = variants.VARIANTS["numpy"]

    def off(*args):
        centres, labels = real(*args)
        return [(c[0] + 1e-6, *c[1:]) for c in centres], labels

    monkeypatch.setitem(variants.VARIANTS, "numpy", off)
    with pytest.raises(SystemExit, match="numpy with 1 workers disagrees with loop"):
        bench.bench(300, tmp_path / "r.csv", workers=(1,), runs=1)
    assert not (tmp_path / "r.csv").exists()


def test_table():
    rows = bench.run_half(300, (1,), 1)
    lines = bench.table(rows).splitlines()
    assert (
        lines[0]
        == "| variant | workers | GIL | seconds | pool start-up s | speedup vs loop |"
    )
    assert len(lines) == 2 + len(rows)
