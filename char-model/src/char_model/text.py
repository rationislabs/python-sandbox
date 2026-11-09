"""Characters to integer ids, the train/validation split, and the metric."""

from collections import Counter
from collections.abc import Sequence

import numpy as np

RARE = "\N{REPLACEMENT CHARACTER}"
MIN_COUNT = 3


def build_vocab(text: str, min_count: int = MIN_COUNT) -> list[str]:
    """RARE first, then every character seen at least `min_count` times, sorted."""
    counts = Counter(text)
    return [RARE, *sorted(c for c, n in counts.items() if n >= min_count and c != RARE)]


def encode(text: str, vocab: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(vocab)}
    return np.array([index.get(c, 0) for c in text], dtype=np.int32)


def decode(ids: Sequence[int] | np.ndarray, vocab: list[str]) -> str:
    return "".join(vocab[int(i)] for i in ids)


def split_point(n: int) -> int:
    """The last tenth is for validation."""
    return n - n // 10


def windows(
    ids: np.ndarray, start: int, stop: int, context: int
) -> tuple[np.ndarray, np.ndarray]:
    """X[k] = the `context` ids before target t, Y[k] = ids[t], for t in start..stop-1."""
    if start < context:
        raise ValueError(f"the first target needs {context} ids before it")
    X = np.lib.stride_tricks.sliding_window_view(
        ids[start - context : stop - 1], context
    )
    return X, ids[start:stop]


def bits_per_char(p: np.ndarray) -> float:
    """Mean of -log2 of the probabilities given to the targets."""
    return float(-np.mean(np.log2(p)))
