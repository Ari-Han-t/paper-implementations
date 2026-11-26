import numpy as np
from optimizers import FOAM, SOAP, Adam, factored_second_moment
from model import init, forward_backward


def test_factored_exact_for_rank1():
    # Under K-FAC + diagonal output curvature, E[G_rot^2] = a b^T (rank 1):
    rng = np.random.default_rng(0)
    a, b = rng.random(7) + 0.1, rng.random(5) + 0.1
    V = np.outer(a, b)
    assert np.allclose(factored_second_moment(V), V)


def test_kfac_diag_output_makes_rotated_second_moment_rank1():
    # G = D^{1/2} Z S^{1/2}, D diagonal (output), S full (input) -> in the
    # eigenbasis of S the expected G_rot^2 is diag(D) x eig(S): rank one.
    rng = np.random.default_rng(1)
    m, n, N = 6, 5, 40000
    d = rng.random(m) + 0.2
    A = rng.standard_normal((n, n)); S = A @ A.T + 0.1 * np.eye(n)
    w, Q = np.linalg.eigh(S)
    Sh = Q @ np.diag(np.sqrt(w)) @ Q.T
    Z = rng.standard_normal((N, m, n))
    G = np.sqrt(d)[None, :, None] * Z @ Sh
    V = ((G @ Q) ** 2).mean(0)
    rel = np.abs(factored_second_moment(V) - V).max() / V.max()
    assert rel < 0.05, rel


def test_foam_state_smaller_than_soap():
    p = {"W": np.zeros((32, 16)), "b": np.zeros(32)}
    g = {"W": np.random.default_rng(0).standard_normal((32, 16)), "b": np.ones(32)}
    f, s = FOAM({k: v.copy() for k, v in p.items()}), SOAP({k: v.copy() for k, v in p.items()})
    f.step({k: v.copy() for k, v in p.items()}, g); s.step({k: v.copy() for k, v in p.items()}, g)
    assert f.state_floats() < s.state_floats()
    # FOAM matrix state: M (m*n) + R,QR (2 n^2) + r (m) + c (n)
    assert f.state["W"]["m"].size + 2 * 16 * 16 + 32 + 16 == sum(
        v.size for v in f.state["W"].values())


def test_rotation_basis_orthonormal():
    p = {"W": np.zeros((8, 6))}
    opt = FOAM(p)
    opt.step(p, {"W": np.random.default_rng(2).standard_normal((8, 6))})
    Q = opt.state["W"]["QR"]
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-10)


def test_foam_dense_identity_basis_matches_adam_first_step():
    # With a dense V, the first step is sign-like in rotated space and its
    # norm matches Adam's first step (both ~ lr*sqrt(numel)).
    rng = np.random.default_rng(3)
    g = {"W": rng.standard_normal((4, 3))}
    pa, pf = {"W": np.zeros((4, 3))}, {"W": np.zeros((4, 3))}
    Adam(pa, lr=0.1, betas=(0.95, 0.95)).step(pa, g)
    FOAM(pf, lr=0.1, factored=False).step(pf, g)
    assert np.isclose(np.linalg.norm(pa["W"]), np.linalg.norm(pf["W"]), rtol=1e-4)


def test_foam_reduces_loss():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((256, 10)); Y = np.sin(X[:, :3])
    p = init(10, 32, 3, rng)
    opt = FOAM(p, lr=0.01)
    l0, _ = forward_backward(p, X, Y)
    for _ in range(200):
        _, g = forward_backward(p, X, Y); opt.step(p, g)
    l1, _ = forward_backward(p, X, Y)
    assert l1 < 0.3 * l0
