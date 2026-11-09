import numpy as np
import pytest

from char_model import bigram, text


def test_vocab_puts_rare_first_and_drops_chars_seen_under_3_times():
    assert text.build_vocab("aaabbbbc!!") == [text.RARE, "a", "b"]


def test_encode_maps_unknown_to_rare():
    vocab = text.build_vocab("aaabbb")
    assert text.encode("ab☃c", vocab).tolist() == [1, 2, 0, 0]
    assert text.decode([1, 0, 2], vocab) == "a�b"


def test_split_keeps_last_tenth_for_validation():
    assert text.split_point(100) == 90 and text.split_point(1005) == 905


def test_windows():
    X, Y = text.windows(np.arange(10), 3, 6, 3)
    assert X.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]] and Y.tolist() == [3, 4, 5]
    with pytest.raises(ValueError):
        text.windows(np.arange(10), 2, 6, 3)


def test_uniform_model_scores_log2_vocab_size():
    assert text.bits_per_char(np.full(50, 1 / 37)) == pytest.approx(np.log2(37))


def test_bigram_add_one_smoothing():
    P = bigram.fit(np.array([0, 1, 0, 1]), 3)
    assert P[0].tolist() == pytest.approx([1 / 5, 3 / 5, 1 / 5])
    assert P[2].tolist() == pytest.approx([1 / 3] * 3)
    assert bigram.bpc(P, np.array([0, 1, 0, 1]), 1, 4) == pytest.approx(
        -np.mean(np.log2([3 / 5, 2 / 4, 3 / 5]))
    )
