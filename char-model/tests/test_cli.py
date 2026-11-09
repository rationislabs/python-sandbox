import json

import pytest

from char_model import cli

TEXT = "the quick brown fox jumps over a lazy dog. " * 15
FAST = ["--steps", "20", "--eval-every", "10", "--hidden", "32"]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cli, "CORPUS", tmp_path / "data" / "corpus.txt")


def default_corpus():
    cli.CORPUS.parent.mkdir(exist_ok=True)
    cli.CORPUS.write_text(TEXT)


def test_custom_corpus_writes_only_runs(paths, tmp_path):
    custom = tmp_path / "notes.txt"
    custom.write_text(TEXT)
    cli.main(["train", "--corpus", str(custom), *FAST])
    (run,) = cli.RUNS.iterdir()
    assert {f.name for f in run.iterdir()} == {
        "model.npz",
        "samples.txt",
        "metrics.json",
    }
    assert not cli.DATA.exists() or not any(cli.DATA.iterdir())


def test_custom_corpus_at_the_default_path_still_writes_only_runs(paths):
    default_corpus()
    cli.main(["train", "--corpus", str(cli.CORPUS), *FAST])
    assert not (cli.DATA / "model.npz").exists()
    assert len(list(cli.RUNS.iterdir())) == 1


def test_default_corpus_also_writes_data_model(paths):
    default_corpus()
    cli.main(["train", *FAST])
    assert (cli.DATA / "model.npz").exists()
    m = json.loads(next(cli.RUNS.glob("*/metrics.json")).read_text())
    assert m["mlp_val_bpc"] > 0 and m["bigram_val_bpc"] > 0 and m["steps"] == 20


def test_train_short_corpus_exits_cleanly(paths, tmp_path):
    short = tmp_path / "s.txt"
    short.write_text("abcabc")
    with pytest.raises(SystemExit, match="too short"):
        cli.main(["train", "--corpus", str(short), *FAST])
    assert not cli.RUNS.exists() or not any(cli.RUNS.iterdir())


def test_missing_default_corpus_says_run_corpus(paths):
    with pytest.raises(SystemExit, match="char-model corpus"):
        cli.main(["train", *FAST])


def test_sample_falls_back_to_data_model(paths, capsys):
    default_corpus()
    cli.main(["train", *FAST])
    for f in cli.RUNS.glob("*/*"):
        f.unlink()
    capsys.readouterr()
    cli.main(["sample", "--n", "40"])
    assert len(capsys.readouterr().out.rstrip("\n")) == 40


def test_sample_is_deterministic(paths, capsys):
    default_corpus()
    cli.main(["train", *FAST])
    capsys.readouterr()
    cli.main(["sample", "--n", "30", "--seed", "4"])
    a = capsys.readouterr().out
    cli.main(["sample", "--n", "30", "--seed", "4"])
    assert capsys.readouterr().out == a


def test_sample_without_any_model_exits(paths):
    with pytest.raises(SystemExit, match="char-model train"):
        cli.main(["sample"])


def test_sample_rejects_nonpositive_temp_and_n(paths):
    for args in (["--temp", "0"], ["--temp", "-1"], ["--n", "0"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["sample", *args])
        assert e.value.code == 2
