import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from red_tide import cli

FEATURES_FIXTURE = Path(__file__).parent / "fixtures" / "features.csv"


PROBE = (
    "import sys, red_tide.cli; print('sklearn' in sys.modules, 'pandas' in sys.modules)"
)


def test_help_does_not_import_sklearn():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"


def test_train_on_fixture_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA", tmp_path / "data")
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")
    (tmp_path / "data").mkdir()
    shutil.copy(FEATURES_FIXTURE, tmp_path / "data" / "features.csv")
    cli.main(["train", "--trials", "3", "--jobs", "2"])
    metrics = pd.read_csv(tmp_path / "results" / "metrics.csv")
    assert list(metrics.columns) == [
        "model",
        "subset",
        "pr_auc",
        "recall_at_p50",
        "positives",
    ]
    assert len(metrics) == 8
    assert set(metrics.model) == {"persistence", "season", "logreg", "hgb"}
    assert set(metrics.subset) == {"test", "onset"}


def test_report_prints_markdown_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    pd.DataFrame(
        {
            "model": ["persistence", "hgb"],
            "subset": ["test", "test"],
            "pr_auc": [0.51, 0.62],
            "recall_at_p50": [0.4, 0.55],
            "positives": [100, 100],
        }
    ).to_csv(tmp_path / "metrics.csv", index=False)
    cli.main(["report"])
    out = capsys.readouterr().out
    assert "| model" in out and "| hgb" in out and "0.62" in out


def test_report_without_results_says_what_to_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(["report"])
    assert "bloom train" in capsys.readouterr().out
