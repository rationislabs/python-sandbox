import numpy as np
import pytest

from char_model import mlp


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    p = mlp.init(5, context=3, embed=4, hidden=6, seed=0, dtype=np.float64)
    X, Y = (
        rng.integers(0, 5, (7, 3)),
        rng.integers(0, 5, 7),
    )  # repeated ids: add.at is exercised
    _, grads = mlp.loss_and_grads(p, X, Y)
    eps = 1e-6
    for name, w in p.items():
        num = np.zeros_like(w)
        for i in np.ndindex(w.shape):
            old = w[i]
            w[i] = old + eps
            up, _ = mlp.loss_and_grads(p, X, Y)
            w[i] = old - eps
            down, _ = mlp.loss_and_grads(p, X, Y)
            w[i] = old
            num[i] = (up - down) / (2 * eps)
        # floor 1e-5: finite differences carry ~1e-10 rounding, so a component below 1e-5 is
        # judged by an absolute error under 1e-10 instead of a relative one
        rel = np.abs(grads[name] - num) / np.maximum(
            np.abs(grads[name]) + np.abs(num), 1e-5
        )
        assert rel.max() < 1e-5, name


def test_shapes_and_probabilities():
    p = mlp.init(11, seed=0)
    assert p["W1"].shape == (16 * 24, 256) and p["W2"].shape == (256, 11)
    assert p["C"].dtype == np.float32
    P = mlp.probs(p, np.zeros((4, 16), dtype=np.int32))
    assert P.shape == (4, 11) and np.allclose(P.sum(1), 1)


def test_initial_loss_is_near_uniform():
    p = mlp.init(30, seed=0, dtype=np.float64)
    loss, _ = mlp.loss_and_grads(p, np.zeros((64, 16), dtype=int), np.arange(64) % 30)
    assert loss == pytest.approx(np.log(30), rel=0.05)


def test_adam_first_step_moves_each_weight_by_lr():
    p = {"w": np.array([1.0, -1.0])}
    opt = mlp.Adam(p)
    opt.step({"w": np.array([0.5, -2.0])}, lr=0.1)
    assert p["w"] == pytest.approx([0.9, -0.9], abs=1e-6)
