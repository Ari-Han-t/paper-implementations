import numpy as np

from hdm import Router, World, area_under_tradeoff, build, evaluate, router_features, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 7)) * 10)
    assert np.allclose(p.sum(1), 1)


def test_slow_mode_beats_fast_mode():
    world, route, test = build(seed=1, n_train=2000, n_route=2000, n_test=2000)
    assert test["slow_ok"].mean() > test["fast_ok"].mean() + 0.05


def test_evaluate_extremes():
    fo = np.array([1., 0., 1., 0.])
    so = np.array([1., 1., 1., 1.])
    assert evaluate(np.zeros(4, bool), fo, so, 64) == dict(acc=0.5, slow_frac=0.0, tokens=0.0)
    assert evaluate(np.ones(4, bool), fo, so, 64)["tokens"] == 64.0


def test_oracle_router_has_best_tradeoff():
    rng = np.random.default_rng(0)
    fo = (rng.random(1000) < 0.8).astype(float)
    so = np.maximum(fo, (rng.random(1000) < 0.9).astype(float))
    oracle = area_under_tradeoff(so - fo, fo, so)
    rand = area_under_tradeoff(rng.random(1000), fo, so)
    assert oracle > rand


def test_router_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(20, 6))
    r = Router(6, hidden=5)
    r.value(f)  # fixes normaliser
    dv = rng.normal(size=20)
    g = r._grads(f, dv)
    eps = 1e-6
    i, j = 2, 3
    r.W1[i, j] += eps; up = r.value(f) @ dv
    r.W1[i, j] -= 2 * eps; dn = r.value(f) @ dv
    r.W1[i, j] += eps
    assert abs((up - dn) / (2 * eps) - g[0][i, j]) < 1e-5


def test_supervised_router_beats_random_and_saves_tokens():
    world, route, test = build(seed=2, n_train=3000, n_route=3000, n_test=3000)
    r = Router(route["f"].shape[1]).fit_supervised(route["f"], route["slow_ok"] - route["fast_ok"], epochs=200)
    vh = r.value(test["f"])
    fo, so = test["fast_ok"], test["slow_ok"]
    rnd = np.random.default_rng(0).random(len(fo))
    assert area_under_tradeoff(vh, fo, so) > area_under_tradeoff(rnd, fo, so) + 0.02
    m = evaluate(vh > 0.1, fo, so, world.T)
    assert m["acc"] > fo.mean() + 0.05 and m["slow_frac"] < 0.6


def test_cdpo_router_is_price_responsive():
    world, route, test = build(seed=3, n_train=3000, n_route=3000, n_test=3000)
    r = Router(route["f"].shape[1]).fit_cdpo(route["f"], route["fast_ok"], route["slow_ok"], episodes=20000)
    vh = r.value(test["f"])
    fracs = [evaluate(vh > lam, test["fast_ok"], test["slow_ok"], world.T)["slow_frac"] for lam in (0.02, 0.2, 0.4)]
    assert fracs[0] > fracs[1] > fracs[2]
    assert evaluate(vh > 0.05, test["fast_ok"], test["slow_ok"], world.T)["acc"] > test["fast_ok"].mean() + 0.05
