import numpy as np
import pytest

from char_model import bigram, mlp, text, train

TEXT = ("the quick brown fox jumps over a lazy dog. " * 5)[:200]
SMALL = {
    "context": 8,
    "embed": 8,
    "hidden": 64,
    "batch": 32,
    "lr_max": 1e-2,
    "lr_min": 1e-3,
    "eval_every": 100,
}
vocab = text.build_vocab(TEXT, min_count=1)
ids = text.encode(TEXT, vocab)
k = text.split_point(len(ids))
p = mlp.init(len(vocab), context=8, embed=8, hidden=64, seed=0)
start = ids[:8]


def quiet(*_):
    pass


def test_lr_schedule_is_cosine():
    assert train.lr_at(0, 100) == pytest.approx(1e-3)
    assert train.lr_at(100, 100) == pytest.approx(3e-5)
    assert train.lr_at(50, 100) == pytest.approx((1e-3 + 3e-5) / 2)


def test_tiny_model_overfits_200_chars_far_below_bigram():
    res = train.train(ids, len(vocab), steps=800, seed=0, **SMALL, log=quiet)
    base = bigram.bpc(bigram.fit(ids[:k], len(vocab)), ids, k, len(ids))
    assert res.best_val_bpc < 0.5 * base


def test_returns_the_best_validation_weights():
    res = train.train(
        ids, len(vocab), steps=300, seed=0, **{**SMALL, "eval_every": 20}, log=quiet
    )
    assert res.best_val_bpc == min(v for _, _, v in res.history)
    X, Y = text.windows(ids, k, len(ids), 8)
    got = text.bits_per_char(mlp.probs(res.params, X)[np.arange(len(Y)), Y])
    assert got == pytest.approx(res.best_val_bpc, rel=1e-5)


def test_same_seed_same_weights():
    a, b = (
        train.train(ids, len(vocab), steps=50, seed=3, **SMALL, log=quiet) for _ in "ab"
    )
    assert all(np.array_equal(a.params[key], b.params[key]) for key in a.params)


def test_too_short_corpus_raises():
    with pytest.raises(ValueError, match="too short"):
        train.train(np.arange(10) % 3, 3, steps=5, log=quiet)


def test_sampling_is_seeded():
    s = mlp.sample(p, vocab, start, n=50, temp=1.0, seed=1)
    assert len(s) == 50 and s == mlp.sample(p, vocab, start, n=50, temp=1.0, seed=1)
    assert s != mlp.sample(p, vocab, start, n=50, temp=1.0, seed=2)


def test_near_zero_temperature_gives_argmax():
    ctx, out = list(start), []
    for _ in range(30):
        i = int(np.argmax(mlp.probs(p, np.array([ctx[-8:]]))[0]))
        out.append(vocab[i])
        ctx.append(i)
    assert mlp.sample(p, vocab, start, n=30, temp=1e-6, seed=1) == "".join(out)
    assert mlp.sample(p, vocab, start, n=30, temp=1e-6, seed=9) == "".join(out)


def test_save_load_roundtrip(tmp_path):
    mlp.save(tmp_path / "m.npz", p, vocab, start)
    q, v, st = mlp.load(tmp_path / "m.npz")
    assert (
        v == vocab
        and np.array_equal(st, start)
        and all(np.array_equal(p[key], q[key]) for key in p)
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_never_finite_validation_raises():
    with pytest.raises(ValueError, match="validation never finite"):
        train.train(
            ids, len(vocab), steps=40, seed=0, **{**SMALL, "lr_max": 1e30}, log=quiet
        )
