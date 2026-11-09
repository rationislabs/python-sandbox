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


@pytest.mark.filterwarnings("error")
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


@pytest.mark.parametrize("name", ["../data", "/tmp/elsewhere", "a/b", ".."])
def test_run_name_must_stay_inside_runs(paths, tmp_path, name):
    custom = tmp_path / "notes.txt"
    custom.write_text(TEXT)
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "--corpus", str(custom), "--name", name, *FAST])
    assert e.value.code == 2
    assert not cli.DATA.exists() or not any(cli.DATA.iterdir())


def test_existing_run_folder_is_refused_before_training(paths, tmp_path, monkeypatch):
    default_corpus()
    (cli.RUNS / "taken").mkdir(parents=True)

    def boom(*a, **k):
        raise AssertionError("trained although the run folder exists")

    monkeypatch.setattr(cli.train, "train", boom)
    with pytest.raises(SystemExit, match="taken"):
        cli.main(["train", "--name", "taken", *FAST])


def test_default_learning_rate_is_the_published_one():
    assert cli.train.LR_MAX == 1e-3


def test_train_prints_baseline_before_training(paths, capsys, monkeypatch):
    default_corpus()
    real = cli.train.train

    def spy(*a, **k):
        out = capsys.readouterr().out
        assert "vocabulary" in out and "bigram" in out and "MLP" not in out
        return real(*a, **k)

    monkeypatch.setattr(cli.train, "train", spy)
    cli.main(["train", *FAST])
    assert "MLP" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_diverged_training_exits_cleanly(paths):
    default_corpus()
    with pytest.raises(SystemExit, match="validation never finite"):
        cli.main(["train", *FAST, "--lr-max", "1e30"])
    assert not cli.RUNS.exists() or not any(cli.RUNS.iterdir())
    assert not (cli.DATA / "model.npz").exists()
