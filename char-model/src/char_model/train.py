"""Training: mini-batches, Adam, a cosine learning rate, and the weights with the best validation."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from char_model import mlp
from char_model.text import bits_per_char, split_point, windows

CONTEXT = 16
EMBED = 24
HIDDEN = 256
BATCH = 128
LR_MAX = 3e-3
LR_MIN = 3e-5
STEPS = 20_000
EVAL_EVERY = 500
SEED = 0


def lr_at(
    step: int, steps: int, lr_max: float = LR_MAX, lr_min: float = LR_MIN
) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / steps))


@dataclass
class TrainResult:
    params: mlp.Params
    best_val_bpc: float
    best_step: int
    history: list[
        tuple[int, float, float]
    ]  # step, train bpc since the last eval, val bpc
    seconds: float


def train(
    ids: np.ndarray,
    vocab_size: int,
    *,
    steps: int = STEPS,
    batch: int = BATCH,
    lr_max: float = LR_MAX,
    lr_min: float = LR_MIN,
    eval_every: int = EVAL_EVERY,
    seed: int = SEED,
    context: int = CONTEXT,
    embed: int = EMBED,
    hidden: int = HIDDEN,
    log: Callable[..., None] = print,
) -> TrainResult:
    n = len(ids)
    n_train = split_point(n)
    if n_train <= context or n_train >= n:
        raise ValueError(
            f"corpus too short: {n} characters, need more than {context} for training "
            "plus some for validation"
        )
    Xt, Yt = windows(ids, context, n_train, context)
    Xv, Yv = windows(ids, n_train, n, context)
    rng = np.random.default_rng(seed)
    params = mlp.init(vocab_size, context, embed, hidden, seed)
    opt = mlp.Adam(params)
    best, best_step, best_params = math.inf, 0, params
    history: list[tuple[int, float, float]] = []
    losses: list[float] = []
    t0 = perf_counter()
    for step in range(steps):
        idx = rng.integers(0, len(Yt), batch)
        loss, grads = mlp.loss_and_grads(params, Xt[idx], Yt[idx])
        lr = lr_at(step, steps, lr_max, lr_min)
        opt.step(grads, lr)
        losses.append(loss / math.log(2))
        done = step + 1
        if done % eval_every == 0 or done == steps:
            val = bits_per_char(mlp.probs(params, Xv)[np.arange(len(Yv)), Yv])
            tr = float(np.mean(losses))
            losses.clear()
            history.append((done, tr, val))
            mark = ""
            if val < best:
                best, best_step, mark = val, done, " *"
                best_params = {k: v.copy() for k, v in params.items()}
            log(
                f"step {done:>6}  train {tr:.3f}  val {val:.3f}  lr {lr:.1e}  "
                f"{perf_counter() - t0:.0f}s{mark}"
            )
    return TrainResult(best_params, best, best_step, history, perf_counter() - t0)
