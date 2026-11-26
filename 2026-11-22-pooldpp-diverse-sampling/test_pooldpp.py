import itertools
import numpy as np
from kdpp import elementary_symmetric, sample_kdpp, greedy_map
from pooldpp import pool_dpp, pool_kernel, target_kernel
from toy_lm import make_toy_lm


def random_psd(n, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    return B @ B.T / n + 0.1 * np.eye(n)


def test_elementary_symmetric_matches_bruteforce():
    lam = np.array([0.5, 2.0, 1.5, 3.0])
    E = elementary_symmetric(lam, 3)
    for k in range(4):
        brute = sum(np.prod(c) for c in itertools.combinations(lam, k))
        assert np.isclose(E[k, -1], brute)


def test_kdpp_returns_k_distinct_items():
    L = random_psd(12)
    rng = np.random.default_rng(0)
    for _ in range(50):
        Y = sample_kdpp(L, 4, rng)
        assert len(Y) == 4 and len(set(Y)) == 4


def test_kdpp_distribution_matches_exact():
    L = random_psd(6, seed=3)
    k = 2
    subsets = list(itertools.combinations(range(6), k))
    dets = np.array([np.linalg.det(L[np.ix_(s, s)]) for s in subsets])
    exact = dets / dets.sum()
    rng = np.random.default_rng(1)
    counts = dict.fromkeys(subsets, 0)
    n = 20000
    for _ in range(n):
        counts[tuple(sample_kdpp(L, k, rng))] += 1
    emp = np.array([counts[s] for s in subsets]) / n
    assert 0.5 * np.abs(emp - exact).sum() < 0.02


def test_greedy_map_first_pick_is_best_singleton_and_nonsingular():
    L = random_psd(10, seed=5)
    Y = greedy_map(L, 3)
    assert np.argmax(np.diag(L)) in Y
    assert np.linalg.det(L[np.ix_(Y, Y)]) > 0


def test_pool_weights_are_unbiased_for_target_quality():
    lm = make_toy_lm()
    rng = np.random.default_rng(0)
    N = 200000
    pool = lm["sample_proposal"](N, rng)
    w = np.exp(lm["logp"][pool] - lm["logq"][pool]) / N  # diagonal of the pool kernel
    est = np.bincount(pool, weights=w, minlength=lm["M"])
    small = pool[:50]
    Lt = pool_kernel(small, lm["logp"], lm["logq"], lm["E"])
    assert np.allclose(np.diag(Lt), np.exp(lm["logp"][small] - lm["logq"][small]) / 50)
    assert np.allclose(est, np.exp(lm["logp"]), atol=3e-3)


def test_pool_dpp_returns_distinct_universe_items():
    lm = make_toy_lm()
    rng = np.random.default_rng(2)
    for mode in ["sample", "map"]:
        Y = pool_dpp(lm["sample_proposal"], lm["logp"], lm["logq"], lm["E"], 3, 50, mode=mode, rng=rng)
        assert len(set(Y)) == 3 and all(0 <= y < lm["M"] for y in Y)
