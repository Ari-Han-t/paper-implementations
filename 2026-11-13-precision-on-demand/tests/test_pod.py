import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gates import (auc, entropy, gated_logp, generate, kl, threshold_for_budget,  # noqa: E402
                   top2_margin)
from model import init_params, log_softmax, loss_and_grads, make_windows  # noqa: E402
from quant import (bytes_per_token, dequant_high, dequant_low, make_views,  # noqa: E402
                   quantize_int8)


def test_int8_roundtrip_error_bounded():
    w = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)
    q, s = quantize_int8(w)
    assert q.dtype == np.int8
    assert np.all(np.abs(dequant_high(q, s) - w) <= s / 2 + 1e-6)


def test_low_pass_uses_only_top_nibble():
    q = np.arange(-127, 128, dtype=np.int8)[:, None]
    s = np.ones((1, 1), np.float32)
    lo = dequant_low(q, s)
    # values sharing the same top nibble map to the same reconstruction
    top = np.right_shift(q.astype(np.int16), 4)
    for t in np.unique(top):
        assert len(np.unique(lo[top == t])) == 1
    # midpoint reconstruction error is at most half a 16-wide bucket
    assert np.all(np.abs(lo - q) <= 8.0)
    assert len(np.unique(top)) <= 16


def test_views_share_storage_semantics():
    p = init_params(10, ctx=4, emb=3, hidden=8)
    hi, lo = make_views(p)
    for k in ("b1", "b2"):
        assert hi[k] is p[k] and lo[k] is p[k]
    assert np.abs(hi["W1"] - p["W1"]).max() < np.abs(lo["W1"] - p["W1"]).max()


def test_gradient_check():
    rng = np.random.default_rng(0)
    p = init_params(7, ctx=3, emb=4, hidden=5, seed=1)
    p = {k: v.astype(np.float64) for k, v in p.items()}
    X = rng.integers(0, 7, (6, 3))
    Y = rng.integers(0, 7, 6)
    _, g = loss_and_grads(p, X, Y)
    eps = 1e-6
    for k in p:
        idx = tuple(rng.integers(0, d) for d in p[k].shape)
        old = p[k][idx]
        p[k][idx] = old + eps
        lp, _ = loss_and_grads(p, X, Y)
        p[k][idx] = old - eps
        lm, _ = loss_and_grads(p, X, Y)
        p[k][idx] = old
        assert abs((lp - lm) / (2 * eps) - g[k][idx]) < 1e-5, k


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 5, 200).astype(float)  # many ties
    y = rng.random(200) < 0.3
    pos, neg = s[y], s[~y]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert auc(s, y) == pytest.approx(brute)
    assert auc(np.arange(10), np.arange(10) >= 5) == 1.0


def test_threshold_hits_budget():
    s = np.random.default_rng(0).normal(size=10_000)
    for b in (0.05, 0.15, 0.5):
        assert (s > threshold_for_budget(s, b)).mean() == pytest.approx(b, abs=0.005)
    assert (s > threshold_for_budget(s, 0)).mean() == 0
    assert (s > threshold_for_budget(s, 1)).mean() == 1


def test_uncertainty_measures():
    sharp = log_softmax(np.array([[10.0, 0, 0, 0]]))
    flat = log_softmax(np.zeros((1, 4)))
    assert entropy(flat)[0] == pytest.approx(np.log(4))
    assert entropy(sharp)[0] < entropy(flat)[0]
    assert top2_margin(sharp)[0] > top2_margin(flat)[0]
    assert top2_margin(flat)[0] == pytest.approx(0)


def test_gated_logp_and_kl():
    lhi = log_softmax(np.random.default_rng(0).normal(size=(5, 6)))
    llo = log_softmax(np.random.default_rng(1).normal(size=(5, 6)))
    esc = np.array([1, 0, 1, 0, 0], bool)
    g = gated_logp(lhi, llo, esc)
    assert np.allclose(kl(lhi, g)[esc], 0)
    assert np.all(kl(lhi, g)[~esc] > 0)


def test_bytes_model():
    assert bytes_per_token(0.0) == 0.5 and bytes_per_token(1.0) == 1.0
    assert bytes_per_token(1.0, "rerun") == 1.5
    assert bytes_per_token(0.2, "rerun") > bytes_per_token(0.2, "nibble")


def test_generate_policies_consistent():
    p = init_params(12, ctx=4, emb=3, hidden=16, seed=2)
    hi, lo = make_views(p)
    prompt = [1, 2, 3, 4]
    ref, esc_hi = generate(hi, lo, prompt, 20, 4, policy="high")
    always, esc_g = generate(hi, lo, prompt, 20, 4, policy="gate", gate="entropy", threshold=-np.inf)
    never, esc_n = generate(hi, lo, prompt, 20, 4, policy="gate", gate="entropy", threshold=np.inf)
    low, _ = generate(hi, lo, prompt, 20, 4, policy="low")
    assert ref == always and esc_hi == esc_g == 20
    assert never == low and esc_n == 0


def test_make_windows():
    X, Y = make_windows([0, 1, 2, 3, 4], 2)
    assert X.tolist() == [[0, 1], [1, 2], [2, 3]] and Y.tolist() == [2, 3, 4]
