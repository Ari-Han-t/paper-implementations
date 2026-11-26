import numpy as np
from rocokv import CompressedKVCache, attend, softmax


def _fill(cache, n, d, rng):
    for _ in range(n):
        cache.append(rng.normal(size=d), rng.normal(size=d))
        cache.query(rng.normal(size=d))


def test_attend_matches_manual():
    rng = np.random.default_rng(0)
    q, K, V = rng.normal(size=4), rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    out, w = attend(q, K, V)
    w_ref = softmax(K @ q / 2.0)
    assert np.allclose(w, w_ref) and np.allclose(out, w_ref @ V)


def test_fixed_budget_respected_and_recent_kept():
    rng = np.random.default_rng(1)
    c = CompressedKVCache(8, budget=20, policy="h2o", recent=5)
    _fill(c, 50, 8, rng)
    last_k = c.K[-1].copy()
    c.compress(uncertainty=1.0)
    assert len(c.K) == 20 and np.allclose(c.K[-1], last_k)


def test_redundancy_flags_duplicates():
    c = CompressedKVCache(4, budget=10)
    base = np.array([1.0, 0, 0, 0])
    for k in [base, np.array([0, 1.0, 0, 0]), base * 1.01, np.array([0, 0, 1.0, 0])]:
        c.append(k, k)
    r = c.redundancy()
    assert r[0] > 0.99          # older copy of a repeated key is redundant
    assert r[1] < 0.1 and r[2] < 0.1


def test_redundant_token_evicted_before_unique():
    c = CompressedKVCache(4, budget=3, policy="rocokv", recent=1, adaptive=False, lam=1.0)
    keys = [np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([1.0, 0, 0, 0]), np.array([0, 0, 1.0, 0])]
    for k in keys:
        c.append(k, k)
    c.score[:] = 1.0            # equal attention importance
    c.compress(uncertainty=0.0)
    kept = {tuple(k) for k in c.K}
    assert (0, 1.0, 0, 0) in kept and (1.0, 0, 0, 0) in kept and len(c.K) == 3


def test_adaptive_budget_grows_with_uncertainty():
    c = CompressedKVCache(4, budget=100, adaptive=True, ema=0.0)
    for _ in range(20):
        c.budget_for(1.0 + 0.01 * np.random.default_rng(0).normal())
    hi, lo = c.budget_for(5.0), c.budget_for(-5.0)
    assert hi > 100 > lo and 50 <= lo and hi <= 150
