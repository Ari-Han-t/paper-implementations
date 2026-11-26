import numpy as np
from verkv import VerKV, dense_attention, poisson_pivots, quantize_sketch

rng = np.random.default_rng(1)
K = rng.normal(size=(512, 32)) * 0.6
V = rng.normal(size=(512, 32))
VRMS = np.sqrt((np.linalg.norm(V, axis=1) ** 2).mean())


def test_full_budget_is_exact():
    m = VerKV(K, V, budget=600, recent=16, eps=1e-9)
    q = rng.normal(size=32)
    o, _ = m.step(q)
    assert np.allclose(o, dense_attention(q, K, V))


def test_recent_always_kept_and_budget_roughly_met():
    sizes = [len(poisson_pivots(K, 100, 16, np.random.default_rng(s))[0]) for s in range(30)]
    piv, p = poisson_pivots(K, 100, 16)
    assert set(range(496, 512)) <= set(piv)
    assert np.all(p[496:] == 1) and abs(p.sum() - 100) < 1
    assert abs(np.mean(sizes) - 100) < 10


def test_sketch_close():
    Kq, sc = quantize_sketch(K)
    assert np.abs(Kq * sc - K).max() <= sc.max() / 2 + 1e-9


def test_horvitz_thompson_is_nearly_unbiased():
    q = rng.normal(size=32)
    outs = [VerKV(K, V, budget=128, eps=1e9, seed=s).step(q)[0] for s in range(400)]
    bias = np.linalg.norm(np.mean(outs, 0) - dense_attention(q, K, V)) / VRMS
    assert bias < 0.02


def test_tolerance_mostly_respected():
    eps, delta = 0.05, 0.1
    m = VerKV(K, V, budget=64, eps=eps, delta=delta)
    viol = 0
    for _ in range(100):
        q = rng.normal(size=32) * 3
        o, info = m.step(q)
        assert info["bound"] <= eps or info["fetched"] > 0
        viol += np.linalg.norm(o - dense_attention(q, K, V)) / VRMS > eps
    assert viol / 100 <= 2 * delta


def test_fetch_counter():
    m = VerKV(K, V, budget=32, eps=1e-6)
    _, info = m.step(rng.normal(size=32))
    assert m.host.fetched == info["fetched"] > 0
