import numpy as onp
import autograd.numpy as np
from autograd import grad

import simplehc as shc


def _setup(depth=8, C=16, n=4, B=5, seed=0, variant="simplehc"):
    rng = onp.random.default_rng(seed)
    p = shc.init_params(rng, variant, depth, C, n=n)
    # make gates strongly input-dependent so the test is non-trivial
    for layer in p["layers"]:
        for s in ("pre", "post"):
            layer[s]["alpha"] = onp.array(1.0)
            layer[s]["b"] = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (B, n, C))
    return p, X


def test_grouped_equals_sequential_with_group_entry_gates():
    p, X = _setup()
    a = shc.stream_simplehc(p["layers"], X, K=4)
    b = shc.stream_simplehc_sequential(p["layers"], X, K=4)
    assert onp.max(onp.abs(a - b)) < 1e-10


def test_equivalence_holds_for_ragged_last_group():
    p, X = _setup(depth=7)
    a = shc.stream_simplehc(p["layers"], X, K=3)
    b = shc.stream_simplehc_sequential(p["layers"], X, K=3)
    assert onp.max(onp.abs(a - b)) < 1e-10


def test_K1_reduces_to_identityhc():
    p, X = _setup(depth=6)
    a = shc.stream_simplehc(p["layers"], X, K=1)
    b = shc.stream_identityhc(p["layers"], X)
    assert onp.max(onp.abs(a - b)) < 1e-10


def test_gradients_match_between_grouped_and_sequential():
    p, X = _setup(depth=4)
    f1 = lambda X_: np.sum(np.sin(shc.stream_simplehc(p["layers"], X_, K=4)))
    f2 = lambda X_: np.sum(np.sin(shc.stream_simplehc_sequential(p["layers"], X_, K=4)))
    assert onp.max(onp.abs(grad(f1)(X) - grad(f2)(X))) < 1e-9


def test_access_counts_match_paper_for_K4():
    p, X = _setup(depth=4)
    orig, grouped = shc.AccessCounter(), shc.AccessCounter()
    shc.stream_identityhc(p["layers"], X, counter=orig)
    shc.stream_simplehc(p["layers"], X, K=4, counter=grouped)
    assert (orig.reads, orig.writes) == (8, 4)
    assert (grouped.reads, grouped.writes) == (2, 1)


def test_sinkhorn_is_doubly_stochastic():
    rng = onp.random.default_rng(1)
    M = shc.sinkhorn(rng.normal(0, 2, (3, 4, 4)), iters=50)
    assert onp.allclose(M.sum(-1), 1, atol=1e-6)
    assert onp.allclose(M.sum(-2), 1, atol=1e-6)


def test_gate_ranges():
    p, X = _setup()
    h_pre, h_post = shc._gates(p["layers"][0], X)
    assert onp.all((h_pre > 0) & (h_pre < 1))
    assert onp.all((h_post > 0) & (h_post < 2))


def test_all_variants_forward_shapes():
    rng = onp.random.default_rng(2)
    toks = rng.integers(0, 8, (6, 3))
    for v in shc.STREAMS:
        p = shc.init_params(rng, v, 8, 16)
        assert shc.forward(p, toks, v).shape == (6, 8)
