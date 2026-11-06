# python-sandbox

Small Python experiments, each in a folder of its own with its own Python version, lock file,
tests and README. Some need Python 3.15, some run on 3.14.

| Experiment | Python | What it shows |
|---|---|---|
| [free-threading](free-threading/) | 3.15t | Threads without the GIL; a PEP 810 lazy import |
| [red-tide](red-tide/) | 3.15t | Red-tide bloom prediction; Optuna threads with the GIL on and off; lazy imports |

Run one:

```sh
cd free-threading
uv run pytest
```

Start a new one: `uv init --package --python 3.14 <name>`, add `ruff` and `pytest` as dev
dependencies (`uv add --dev ruff pytest`), and a row to the table above. CI finds every folder
with a `pyproject.toml` and runs its lint and tests.

Every project here was built with AI assistance.
