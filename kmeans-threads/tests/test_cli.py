import pytest

from kmeans_threads import bench, cli


def test_check_passes(capsys):
    cli.main(["check"])
    out = capsys.readouterr().out
    assert "DIFFERS" not in out
    assert out.splitlines()[-1].endswith("runs agree with loop (relative 1e-9)")


@pytest.mark.parametrize("bad", ["7", "0", "x"])
def test_points_must_be_at_least_k(bad):
    with pytest.raises(SystemExit) as e:
        cli.main(["bench", "--points", bad])
    assert e.value.code == 2


def test_bench_writes_results_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "WORKERS", (1,))  # cli reads bench.WORKERS at call time
    monkeypatch.setattr(bench, "RUNS", 1)
    cli.main(["bench", "--points", "300"])
    assert (tmp_path / "results.csv").exists()
    assert "| variant | workers | GIL |" in capsys.readouterr().out
