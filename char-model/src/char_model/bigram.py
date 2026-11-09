"""The baseline: next character from the previous one, add-one smoothing."""

import numpy as np

from char_model.text import bits_per_char


def fit(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return (counts + 1) / (counts.sum(axis=1, keepdims=True) + vocab_size)


def bpc(P: np.ndarray, ids: np.ndarray, start: int, stop: int) -> float:
    """Bits per character on targets ids[start:stop]."""
    return bits_per_char(P[ids[start - 1 : stop - 1], ids[start:stop]])
