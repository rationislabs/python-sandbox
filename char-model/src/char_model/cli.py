"""`char-model corpus`, `train` and `sample`."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from char_model import bigram, corpus, mlp, text, train

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CORPUS = DATA / "corpus.txt"
REPOS = ROOT.parents[1]


def _positive(kind):
    def parse(s: str):
        v = kind(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be greater than 0, got {s}")
        return v

    return parse


def _run_name(s: str) -> str:
    """One folder name inside runs/: no path, so a run can never write anywhere else."""
    if Path(s).name != s or s in ("", ".", ".."):
        raise argparse.ArgumentTypeError(f"must be a plain folder name, got {s}")
    return s


def _corpus(repos: Path) -> None:
    out = corpus.collect(repos)
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text(out, encoding="utf-8")
    print(
        f"wrote {CORPUS}: {len(out.encode()) / 1000:.0f} KB from {len(corpus.SOURCES)} repos"
    )


def _train(a: argparse.Namespace) -> None:
    default = a.corpus is None
    source = CORPUS if default else Path(a.corpus)
    if default and not source.exists():
        raise SystemExit("no data/corpus.txt: run `char-model corpus` first")
    run = RUNS / (a.name or time.strftime("%Y%m%d-%H%M%S"))
    if run.exists():
        raise SystemExit(f"{run} already exists: pick another --name")
    raw = source.read_text(encoding="utf-8")
    vocab = text.build_vocab(raw)
    ids = text.encode(raw, vocab)
    k = text.split_point(len(ids))
    numbers = {
        "context": a.context,
        "embed": a.embed,
        "hidden": a.hidden,
        "batch": a.batch,
        "lr_max": a.lr_max,
        "lr_min": a.lr_min,
    }
    try:
        res = train.train(
            ids,
            len(vocab),
            steps=a.steps,
            eval_every=a.eval_every,
            seed=a.seed,
            **numbers,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from None
    base = bigram.bpc(bigram.fit(ids[:k], len(vocab)), ids, k, len(ids))
    print(f"{len(raw)} characters, vocabulary {len(vocab)}")
    print(f"validation bits per char: bigram {base:.3f}, MLP {res.best_val_bpc:.3f}")
    run.mkdir(parents=True)
    start = ids[: a.context]
    mlp.save(run / "model.npz", res.params, vocab, start)
    with open(run / "samples.txt", "w", encoding="utf-8") as f:
        for temp in (0.5, 0.8, 1.0):
            f.write(f"--- temperature {temp}, seed 1\n")
            f.write(
                mlp.sample(res.params, vocab, start, n=400, temp=temp, seed=1) + "\n"
            )
    metrics = {
        "corpus_chars": len(raw),
        "vocab_size": len(vocab),
        "bigram_val_bpc": base,
        "mlp_val_bpc": res.best_val_bpc,
        "best_step": res.best_step,
        "steps": a.steps,
        "seconds": round(res.seconds, 1),
        "seed": a.seed,
        **numbers,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if default:
        DATA.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "model.npz", DATA / "model.npz")
    print(f"wrote {run}")


def _latest_model() -> Path | None:
    runs = (
        sorted(RUNS.glob("*/model.npz"), key=lambda p: p.stat().st_mtime)
        if RUNS.exists()
        else []
    )
    if runs:
        return runs[-1]
    return DATA / "model.npz" if (DATA / "model.npz").exists() else None


def _sample(a: argparse.Namespace) -> None:
    path = Path(a.model) if a.model else _latest_model()
    if path is None:
        raise SystemExit("no model: run `char-model train` first")
    params, vocab, start = mlp.load(path)
    print(mlp.sample(params, vocab, np.asarray(start), n=a.n, temp=a.temp, seed=a.seed))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="char-model", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "corpus", help="rebuild data/corpus.txt from the public repos' HEAD"
    )
    c.add_argument("--repos", type=Path, default=REPOS)
    t = sub.add_parser(
        "train", help="train the bigram and the MLP; a --corpus run writes only runs/"
    )
    t.add_argument("--corpus", help="any UTF-8 text file; its outputs stay in runs/")
    t.add_argument("--steps", type=_positive(int), default=train.STEPS)
    t.add_argument("--batch", type=_positive(int), default=train.BATCH)
    t.add_argument("--lr-max", type=_positive(float), default=train.LR_MAX)
    t.add_argument("--lr-min", type=_positive(float), default=train.LR_MIN)
    t.add_argument("--eval-every", type=_positive(int), default=train.EVAL_EVERY)
    t.add_argument("--seed", type=int, default=train.SEED)
    t.add_argument("--context", type=_positive(int), default=train.CONTEXT)
    t.add_argument("--embed", type=_positive(int), default=train.EMBED)
    t.add_argument("--hidden", type=_positive(int), default=train.HIDDEN)
    t.add_argument(
        "--name", type=_run_name, help="run folder name (default: the date and time)"
    )
    s = sub.add_parser("sample", help="print text from the newest model")
    s.add_argument(
        "--model", help="a model.npz (default: newest in runs/, else data/model.npz)"
    )
    s.add_argument("--n", type=_positive(int), default=400)
    s.add_argument("--temp", type=_positive(float), default=0.8)
    s.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    if a.cmd == "corpus":
        _corpus(a.repos)
    elif a.cmd == "train":
        _train(a)
    else:
        _sample(a)
