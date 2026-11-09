"""The model: embeddings of the previous characters, one tanh layer, softmax. Written by hand."""

from pathlib import Path

import numpy as np

type Params = dict[str, np.ndarray]


def init(
    vocab_size: int,
    context: int = 16,
    embed: int = 24,
    hidden: int = 256,
    seed: int = 0,
    dtype=np.float32,
) -> Params:
    rng = np.random.default_rng(seed)
    p = {
        "C": rng.standard_normal((vocab_size, embed)),
        "W1": rng.standard_normal((context * embed, hidden)) / np.sqrt(context * embed),
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((hidden, vocab_size)) * 0.1 / np.sqrt(hidden),
        "b2": np.zeros(vocab_size),
    }
    return {k: v.astype(dtype) for k, v in p.items()}


def _forward(params: Params, X: np.ndarray):
    x = params["C"][X].reshape(len(X), -1)
    h = np.tanh(x @ params["W1"] + params["b1"])
    return x, h, h @ params["W2"] + params["b2"]


def logits(params: Params, X: np.ndarray) -> np.ndarray:
    return _forward(params, X)[2]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def probs(params: Params, X: np.ndarray) -> np.ndarray:
    return _softmax(logits(params, X))


def loss_and_grads(
    params: Params, X: np.ndarray, Y: np.ndarray
) -> tuple[float, Params]:
    """Mean cross-entropy in nats and its gradient for every parameter."""
    B = len(X)
    x, h, z = _forward(params, X)
    shifted = z - z.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(B)
    loss = float(-log_p[rows, Y].mean())
    dz = np.exp(log_p)
    dz[rows, Y] -= 1
    dz /= B
    dh = dz @ params["W2"].T
    da = dh * (1 - h * h)
    dC = np.zeros_like(params["C"])
    np.add.at(dC, X, (da @ params["W1"].T).reshape(*X.shape, -1))
    grads = {
        "C": dC,
        "W1": x.T @ da,
        "b1": da.sum(axis=0),
        "W2": h.T @ dz,
        "b2": dz.sum(axis=0),
    }
    return loss, grads


class Adam:
    def __init__(self, params: Params, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.b1, self.b2, self.eps, self.t = params, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Params, lr: float) -> None:
        """One bias-corrected update, in place."""
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            self.params[k] -= (lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(
                m.dtype
            )


def sample(
    params: Params, vocab: list[str], start: np.ndarray, n: int, temp: float, seed: int
) -> str:
    """`n` characters after `start`, each drawn from softmax(logits / temp)."""
    context = params["W1"].shape[0] // params["C"].shape[1]
    rng = np.random.default_rng(seed)
    ctx = [int(i) for i in start[-context:]]
    out = []
    for _ in range(n):
        z = logits(params, np.array([ctx[-context:]]))[0].astype(np.float64) / temp
        p = np.exp(z - z.max())
        i = int(rng.choice(len(p), p=p / p.sum()))
        out.append(vocab[i])
        ctx.append(i)
    return "".join(out)


def save(path: Path, params: Params, vocab: list[str], start: np.ndarray) -> None:
    """One file: weights, vocabulary and the sampling start context."""
    np.savez_compressed(path, **params, vocab=np.array(vocab, dtype="<U1"), start=start)


def load(path: Path) -> tuple[Params, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        params = {k: f[k] for k in ("C", "W1", "b1", "W2", "b2")}
        return params, f["vocab"].tolist(), f["start"]
