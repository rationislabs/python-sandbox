import subprocess
from pathlib import Path

import pytest

from char_model import corpus


def make_repo(path: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    git = ["git", "-C", str(path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "user.name=t", "-c", "user.email=t@example.com"]
        + ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "x"],
        check=True,
    )


def test_sources_are_exactly_the_public_list():
    assert set(corpus.SOURCES) == {
        "ai-language-tutor",
        "anastasis",
        "databricks-data-pipeline-app",
        "hearth",
        "python-etl-pipeline-dashboard",
        "sentiment-analysis",
        "telegram_msg_svc_auto",
        "weather-live-dashboard",
        "python-sandbox",
        "rust-sandbox",
    }
    assert not any("404" in p for ps in corpus.SOURCES.values() for p in ps)


def test_strip_html_keeps_text():
    out = corpus.strip_html(
        "<html><head><style>p{color:red}</style><title>T</title></head>"
        "<body><h1>Hi&nbsp;there</h1><p>a <b>b</b> &amp; c</p><script>x()</script></body></html>"
    )
    assert "Hi there" in out and "a b & c" in out and "T" in out
    assert "<" not in out and "color" not in out and "x()" not in out


def test_collect_reads_only_listed_files_at_head(tmp_path):
    doc = "<html><body><p>{}</p></body></html>"
    make_repo(
        tmp_path / "anastasis",
        {
            "README.md": "committed anastasis readme",
            "docs/index.html": doc.format("index text"),
            "docs/guide.html": doc.format("guide text"),
            "docs/reference.html": doc.format("reference text"),
            "docs/404.html": doc.format("NOT FOUND PAGE"),
        },
    )
    (tmp_path / "anastasis/README.md").write_text("UNPUBLISHED")
    make_repo(
        tmp_path / "python-sandbox",
        {
            "README.md": "sandbox readme",
            "exp/README.md": "exp readme",
            "exp/deep/x/README.md": "TOO DEEP",
            "char-model/README.md": "OWN SAMPLES",
        },
    )
    make_repo(tmp_path / "python-webapp", {"README.md": "PRIVATE"})
    out = corpus.collect(
        tmp_path, {k: corpus.SOURCES[k] for k in ("anastasis", "python-sandbox")}
    )
    assert (
        "committed anastasis readme" in out
        and "guide text" in out
        and "exp readme" in out
    )
    for absent in (
        "UNPUBLISHED",
        "NOT FOUND PAGE",
        "TOO DEEP",
        "OWN SAMPLES",
        "PRIVATE",
        "<p>",
    ):
        assert absent not in out


def test_missing_repo_or_path_raises(tmp_path):
    with pytest.raises(RuntimeError, match="hearth"):
        corpus.collect(tmp_path, {"hearth": ["README.md"]})
    make_repo(tmp_path / "anastasis", {"README.md": "x"})
    with pytest.raises(RuntimeError, match="docs/index.html"):
        corpus.collect(tmp_path, {"anastasis": ["docs/index.html"]})
