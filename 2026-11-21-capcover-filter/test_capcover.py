import numpy as np
from capcover import CapCoverFilter, brute_force_exists, _normalize


def _data(seed=0, n=3000, d=32):
    rng = np.random.default_rng(seed)
    t = _normalize(rng.normal(size=(20, d)))
    return _normalize(t[rng.integers(20, size=n)] + 0.1 * rng.normal(size=(n, d))), rng


def test_no_false_negatives_across_tau():
    x, rng = _data()
    f = CapCoverFilter(n_caps=64).fit(x)
    q = np.vstack([_normalize(x[:300] + 0.05 * rng.normal(size=(300, x.shape[1]))),
                   _normalize(rng.normal(size=(300, x.shape[1])))])
    for tau in [0.5, 0.8, 0.9, 0.99]:
        truth = brute_force_exists(f.data, q, tau)
        assert not (truth & ~f.might_contain(q, tau)).any()


def test_indexed_vectors_always_found():
    x, _ = _data(1)
    f = CapCoverFilter(n_caps=32).fit(x)
    assert f.might_contain(x, 0.999).all()


def test_exact_search_matches_bruteforce():
    x, rng = _data(2)
    f = CapCoverFilter(n_caps=48).fit(x)
    q = _normalize(rng.normal(size=(100, x.shape[1])) + x[:100])
    truth = brute_force_exists(f.data, q, 0.85)
    got = [f.exists_exact(qi, 0.85)[0] for qi in q]
    assert np.array_equal(np.array(got), truth)


def test_rejects_far_queries_and_is_small():
    x, rng = _data(3, n=5000, d=64)
    f = CapCoverFilter(n_caps=64).fit(x)
    far = _normalize(rng.normal(size=(500, 64)))
    assert (~f.might_contain(far, 0.95)).mean() > 0.9
    assert f.filter_bytes() < 0.05 * f.index_bytes()
