"""The training text: the public repos' READMEs and the anastasis docs, read from committed HEAD."""

import html
import re
import subprocess
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

SOURCES = {
    "ai-language-tutor": ["README.md"],
    "anastasis": [
        "README.md",
        "docs/index.html",
        "docs/guide.html",
        "docs/reference.html",
    ],
    "databricks-data-pipeline-app": ["README.md"],
    "hearth": ["README.md"],
    "python-etl-pipeline-dashboard": ["README.md"],
    "sentiment-analysis": ["README.md"],
    "telegram_msg_svc_auto": ["README.md"],
    "weather-live-dashboard": ["README.md"],
    "python-sandbox": ["README.md", "*/README.md"],
    "rust-sandbox": ["README.md", "projects/*/README.md"],
}
# Not python-webapp (private), not learning (coursework), not anastasis docs/404.html.
EXCLUDE = {("python-sandbox", "char-model/README.md")}  # holds model samples


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def strip_html(page: str) -> str:
    """Text of an HTML page: no script or style, entities decoded, blank lines collapsed."""
    p = _Text()
    p.feed(page)
    p.close()
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        raise RuntimeError(f"{repo.name}: git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def collect(
    root: Path,
    sources: dict[str, list[str]] = SOURCES,
    log: Callable[[str], None] = print,
) -> str:
    """Every listed file at each repo's HEAD, joined by blank lines, with each repo's KB logged.
    Never reads a working tree."""
    docs = []
    for repo, patterns in sources.items():
        path = root / repo
        if not (path / ".git").exists():
            raise RuntimeError(f"{repo}: no git repo at {path}")
        files = _git(path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        chosen: set[str] = set()
        for pattern in patterns:
            hits = {f for f in files if PurePosixPath(f).full_match(pattern)} - {
                f for r, f in EXCLUDE if r == repo
            }
            if not hits:
                raise RuntimeError(f"{repo}: nothing matches {pattern} at HEAD")
            chosen |= hits
        size = 0
        for f in sorted(chosen):
            body = _git(path, "show", f"HEAD:{f}")
            docs.append(strip_html(body) if f.endswith(".html") else body)
            size += len(docs[-1].strip().encode())
        log(f"{repo}: {size / 1000:.1f} KB")
    return "\n\n".join(d.strip() for d in docs) + "\n"
