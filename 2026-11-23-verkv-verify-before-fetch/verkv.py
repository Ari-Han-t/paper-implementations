"""Minimal NumPy re-implementation of the ideas in
"Verify Before You Fetch: Reliable KV Cache Compression for LLM Inference" (arXiv 2611.11319).

Single attention head, single decode step. The full KV cache lives in "host memory"
(a plain array we count reads from); the "GPU" holds:
  * a query-agnostic set of pivotal tokens chosen by Poisson sampling (PoissonComp-style),
  * a token-level sketch of every key (int4-style per-token quantised keys).
At each step, KVFetch uses the sketch to estimate the attention mass that falls outside
the resident set. If the implied relative-error bound is within tolerance, no host
memory is touched; otherwise it fetches the highest-scoring missing tokens until the
bound is met.
"""
import numpy as np


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def dense_attention(q, K, V):
    w = softmax(K @ q / np.sqrt(K.shape[1]))
    return w @ V


class HostKV:
    """Full KV cache in 'CPU memory'; counts how many token rows are fetched."""

    def __init__(self, K, V):
        self.K, self.V = K, V
        self.fetched = 0

    def fetch(self, idx):
        self.fetched += len(idx)
        return self.K[idx], self.V[idx]


def quantize_sketch(K, bits=4):
    """Per-token symmetric quantisation of keys: the cheap token-level sketch."""
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(K).max(axis=1, keepdims=True) / levels + 1e-12
    return np.round(K / scale).astype(np.int8), scale


def poisson_pivots(K, budget, recent=16, rng=None):
    """Query-agnostic pivotal-token selection by Poisson (independent Bernoulli) sampling.

    Inclusion probability is proportional to squared key norm (tokens with large keys can draw
    large attention for many queries), scaled so the expected size is `budget`, capped at 1.
    The most recent `recent` tokens are always kept.
    """
    rng = rng or np.random.default_rng(0)
    n = K.shape[0]
    score = np.linalg.norm(K, axis=1) ** 2
    p = np.zeros(n)
    m = max(budget - recent, 0)
    old = np.arange(n - recent)
    # water-filling so that sum(min(1, c*score)) == m
    lo, hi = 0.0, 1e6 / (score[old].mean() + 1e-12)
    for _ in range(60):
        c = (lo + hi) / 2
        if np.minimum(1, c * score[old]).sum() > m:
            hi = c
        else:
            lo = c
    p[old] = np.minimum(1, lo * score[old])
    p[n - recent:] = 1.0
    keep = rng.random(n) < p
    return np.flatnonzero(keep), p


class VerKV:
    """eps: tolerance on normalised error ||o_hat - o|| / rms(||v||); delta: failure prob."""

    def __init__(self, K, V, budget=128, recent=16, eps=0.05, delta=0.1, bits=4, seed=0, chebyshev=False):
        self.host = HostKV(K, V)
        self.n, self.d = K.shape
        self.eps, self.delta = eps, delta
        from statistics import NormalDist
        self.z = NormalDist().inv_cdf(1 - delta / 2) if chebyshev is False else 1 / np.sqrt(delta)
        self.piv, self.p = poisson_pivots(K, budget, recent, np.random.default_rng(seed))
        self.Kp, self.Vp = K[self.piv], V[self.piv]  # resident on "GPU"
        self.Kq, self.scale = quantize_sketch(K, bits)  # token-level key sketch
        self.vnorm = np.linalg.norm(V, axis=1)  # one scalar per token (part of sketch)
        self.vrms = np.sqrt((self.vnorm ** 2).mean())

    def step(self, q):
        """Return (output, info) for one decode query.

        Tokens are either *certain* (always-kept or fetched: weight 1) or represented by
        the Poisson sample with Horvitz-Thompson weight 1/p_i. The certain set is chosen
        from the sketch only (independent of the sample), so the estimate stays unbiased
        for numerator and denominator.
        """
        d = np.sqrt(self.d)
        s_sk = (self.Kq * self.scale) @ q / d  # approximate logits for every token
        pi = self.p
        certain = pi >= 1.0
        resident = np.zeros(self.n, bool)
        resident[self.piv] = True
        K, V = self.host.K, self.host.V  # only rows marked `resident` are ever read
        mx = s_sk.max()
        wt = np.exp(s_sk - mx)  # sketch weights for all tokens
        Z = wt.sum()
        fetched_now = 0
        while True:
            unc = ~certain
            # variance proxy uses ||v_i||^2 (stored scalar) in place of ||v_i - o||^2
            spread = self.vnorm ** 2
            contrib = np.where(unc, (1 - pi) / np.maximum(pi, 1e-12) * wt ** 2 * spread, 0.0)
            bound = self.z * np.sqrt(contrib.sum()) / Z / self.vrms  # CLT (eps, delta) bound
            if bound <= self.eps or certain.all():
                break
            k = min(int(unc.sum()), max(8, fetched_now))
            take = np.argsort(-contrib)[:k]
            take = take[unc[take]]
            need = take[~resident[take]]
            if len(need):
                self.host.fetch(need)  # counts host->GPU traffic
            fetched_now += len(need)
            resident[take] = True
            certain[take] = True
        idx = np.flatnonzero(resident & (certain | (pi > 0)))
        s = K[idx] @ q / d
        w = np.exp(s - s.max()) * np.where(certain[idx], 1.0, 1.0 / np.maximum(pi[idx], 1e-12))
        o = (w @ V[idx]) / w.sum()
        return o, dict(fetched=fetched_now, bound=bound)
