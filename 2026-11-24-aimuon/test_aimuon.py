import numpy as np
from aimuon import newton_schulz, exact_msign, Muon, AIMuon
from model import MLP, make_task


def test_newton_schulz_approximates_msign():
    rng = np.random.default_rng(0)
    G = rng.standard_normal((40, 24))
    X = newton_schulz(G, steps=10)
    s = np.linalg.svd(X, compute_uv=False)
    assert np.all((s > 0.6) & (s < 1.25))  # quintic NS gives singular values in ~[0.7, 1.2]
    cos = np.sum(X * exact_msign(G)) / (np.linalg.norm(X) * np.linalg.norm(exact_msign(G)))
    assert cos > 0.98


def test_aimuon_update_norm_matches_muon_and_descends():
    rng = np.random.default_rng(1)
    G = rng.standard_normal((16, 32))
    o, a = Muon(), AIMuon()
    for _ in range(3):
        G2 = G + 0.3 * rng.standard_normal(G.shape)
        d_m, d_a = o.direction("w", G2), a.direction("w", G2)
    assert abs(np.linalg.norm(d_a) - np.linalg.norm(d_m)) < 1e-6
    assert np.sum(G2 * d_a) > 0  # descent direction


def test_mlp_gradients_match_finite_differences():
    X, Y = make_task(0, n=16, d_in=4, d_hidden_teacher=5, d_out=2)
    net = MLP([4, 6, 2], seed=0)
    _, g = net.loss_and_grads(X, Y)
    W, eps = net.params["W0"], 1e-6
    W[1, 2] += eps; lp, _ = net.loss_and_grads(X, Y)
    W[1, 2] -= 2 * eps; lm, _ = net.loss_and_grads(X, Y)
    W[1, 2] += eps
    assert abs((lp - lm) / (2 * eps) - g["W0"][1, 2]) < 1e-6


def test_aimuon_trains():
    X, Y = make_task(0, n=256)
    net = MLP([32, 64, 8], seed=0)
    opt = AIMuon(lr=0.02)
    l0 = net.loss_and_grads(X, Y)[0]
    for _ in range(100):
        _, g = net.loss_and_grads(X, Y)
        for k in g:
            if k.startswith("W"):
                opt.step(k, net.params[k], g[k])
    assert net.loss_and_grads(X, Y)[0] < 0.5 * l0
