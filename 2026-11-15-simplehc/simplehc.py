"""Minimal NumPy implementation of residual-stream variants from
"On the Design of Hyper-Connections: Simplification, Stability, and Scalability"
(arXiv 2611.05984).

All functions are written with ``autograd.numpy`` so the same code is used for
forward passes, exactness tests, and gradient-based training.

Shapes (per token, batched over B tokens):
    X      : (B, n, C)   n residual streams of width C
    h_pre  : (B, n)      read gate   -> u = X^T h_pre          (B, C)
    h_post : (B, n)      write gate  -> X += h_post o^T        (B, n, C)
    H_res  : (B, n, n)   stream mixing (mHC only)

Variants
    vanilla     : standard residual  x <- x + F(x)            (n = 1, no gates)
    mhc         : X <- H_res X + h_post o^T, H_res via Sinkhorn (doubly stochastic)
    identityhc  : H_res = I, gates recomputed from X_j at every sublayer
    simplehc    : H_res = I, gates for a group of K sublayers computed once from
                  the group-entry state X_L, batched base reads V = X_L^T H_pre,
                  scalar corrections D = tril(H_pre^T H_post, -1), and a single
                  deferred write-back per group.
"""
from __future__ import annotations

import autograd.numpy as np
from autograd.scipy.special import expit as sigmoid

EPS = 1e-6


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def rmsnorm(x, axis=-1):
    return x / np.sqrt(np.mean(x * x, axis=axis, keepdims=True) + EPS)


def sinkhorn(logits, iters=20):
    """Project (B, n, n) logits to (approximately) doubly-stochastic matrices."""
    m = np.exp(logits - np.max(logits, axis=(-1, -2), keepdims=True))
    for _ in range(iters):
        m = m / np.sum(m, axis=-1, keepdims=True)
        m = m / np.sum(m, axis=-2, keepdims=True)
    return m


def mlp_sublayer(p, u):
    """F(u) = W2 tanh(W1 rmsnorm(u)).  u: (B, C) -> (B, C)."""
    return np.dot(np.tanh(np.dot(rmsnorm(u), p["W1"])), p["W2"])


def gate_logits(g, X):
    """mHC-style dynamic gate logits: alpha * W^T RMSNorm(vec X) + b."""
    B, n, C = X.shape
    z = rmsnorm(np.reshape(X, (B, n * C)))
    return g["alpha"] * np.dot(z, g["W"]) + g["b"]


# ---------------------------------------------------------------------------
# parameter init
# ---------------------------------------------------------------------------
def init_params(rng, variant, depth, C, n=4, hidden_mult=2, vocab=8, ctx=3):
    p = {
        "emb": rng.normal(0, 0.5, (vocab, C)),
        "ctx_proj": rng.normal(0, 1 / np.sqrt(ctx * C), (ctx * C, C)),
        "unemb": rng.normal(0, 1 / np.sqrt(C), (C, vocab)),
        "layers": [],
    }
    H = hidden_mult * C
    for _ in range(depth):
        layer = {
            "F": {
                "W1": rng.normal(0, 1 / np.sqrt(C), (C, H)),
                "W2": rng.normal(0, 1 / np.sqrt(H) / np.sqrt(depth), (H, C)),
            }
        }
        if variant != "vanilla":
            for s in ("pre", "post"):
                # b=0 -> h_pre = 0.5, h_post = 1.0 at init; small alpha => near-static
                layer[s] = {
                    "W": rng.normal(0, 1 / np.sqrt(n * C), (n * C, n)),
                    "alpha": np.array(0.01),
                    "b": np.zeros(n),
                }
        if variant == "mhc":
            # bias toward identity so mixing starts close to a plain residual
            layer["res"] = {
                "W": rng.normal(0, 1 / np.sqrt(n * C), (n * C, n * n)),
                "alpha": np.array(0.01),
                "b": np.reshape(4.0 * np.eye(n), (n * n,)),
            }
        p["layers"].append(layer)
    return p


# ---------------------------------------------------------------------------
# residual-stream forward passes
# ---------------------------------------------------------------------------
class AccessCounter:
    """Counts full residual-state (n x C) reads and writes, the overhead the
    paper identifies as the main cost of hyper-connections."""

    def __init__(self):
        self.reads = 0
        self.writes = 0


def _gates(layer, X):
    h_pre = sigmoid(gate_logits(layer["pre"], X))
    h_post = 2.0 * sigmoid(gate_logits(layer["post"], X))
    return h_pre, h_post


def stream_vanilla(layers, x, counter=None):
    for layer in layers:
        x = x + mlp_sublayer(layer["F"], x)
    return x


def stream_mhc(layers, X, counter=None):
    B, n, C = X.shape
    for layer in layers:
        h_pre, h_post = _gates(layer, X)
        H_res = sinkhorn(np.reshape(gate_logits(layer["res"], X), (B, n, n)))
        u = np.einsum("bnc,bn->bc", X, h_pre)
        o = mlp_sublayer(layer["F"], u)
        X = np.einsum("bij,bjc->bic", H_res, X) + h_post[:, :, None] * o[:, None, :]
        if counter:
            counter.reads += 3  # gates, read, mixing
            counter.writes += 1
    return X


def stream_identityhc(layers, X, counter=None):
    """H_res = I; gates conditioned on the current state X_j (per sublayer)."""
    for layer in layers:
        h_pre, h_post = _gates(layer, X)
        u = np.matmul(h_pre[:, None, :], X)[:, 0, :]
        o = mlp_sublayer(layer["F"], u)
        X = X + h_post[:, :, None] * o[:, None, :]
        if counter:
            counter.reads += 2  # gate input z_j, read u_j
            counter.writes += 1
    return X


def group_gates(group, X_L):
    """Group-entry gates (Eq. 3): all K sublayers' gates from X_L.
    Returns H_pre, H_post with shape (B, n, K)."""
    pres, posts = zip(*[_gates(layer, X_L) for layer in group])
    return np.stack(pres, axis=-1), np.stack(posts, axis=-1)


def stream_simplehc_sequential(layers, X, K=4, counter=None):
    """Reference: sequential execution, but gates taken from the group entry.
    This is what SimpleHC must reproduce exactly (paper Appendix B)."""
    for L in range(0, len(layers), K):
        group = layers[L:L + K]
        H_pre, H_post = group_gates(group, X)
        if counter:
            counter.reads += 1
        for a, layer in enumerate(group):
            u = np.einsum("bnc,bn->bc", X, H_pre[:, :, a])
            o = mlp_sublayer(layer["F"], u)
            X = X + H_post[:, :, a][:, :, None] * o[:, None, :]
            if counter:
                counter.reads += 1
                counter.writes += 1
    return X


def stream_simplehc(layers, X, K=4, counter=None):
    """SimpleHC grouped execution (Algorithm 1 in spirit):
        V = X_L^T H_pre                      one batched read
        D = tril(H_pre^T H_post, -1)         K x K scalar corrections
        u_a = V[:, a] + sum_{b<a} D[a,b] o_b
        X_{L+K} = X_L + sum_k h_post_k o_k^T one deferred write
    """
    for L in range(0, len(layers), K):
        group = layers[L:L + K]
        k = len(group)
        H_pre, H_post = group_gates(group, X)                  # read 1 (z)
        V = np.matmul(np.transpose(H_pre, (0, 2, 1)), X)      # read 2 (batched), (B,k,C)
        D = np.einsum("zni,znj->zij", H_pre, H_post)           # (B, k, k), cheap
        outs = []
        for a, layer in enumerate(group):
            u = V[:, a, :]
            for b_ in range(a):
                u = u + D[:, a, b_][:, None] * outs[b_]
            outs.append(mlp_sublayer(layer["F"], u))
        O = np.stack(outs, axis=1)                             # (B, k, C)
        X = X + np.matmul(H_post, O)                           # single write
        if counter:
            counter.reads += 2
            counter.writes += 1
    return X


STREAMS = {
    "vanilla": stream_vanilla,
    "mhc": stream_mhc,
    "identityhc": stream_identityhc,
    "simplehc": stream_simplehc,
}


# ---------------------------------------------------------------------------
# tiny next-character model
# ---------------------------------------------------------------------------
def forward(p, ctx_tokens, variant, n=4, K=4):
    """ctx_tokens: (B, ctx) int array -> logits (B, vocab)."""
    B = ctx_tokens.shape[0]
    e = np.reshape(p["emb"][ctx_tokens], (B, -1))
    h = np.dot(e, p["ctx_proj"])                               # (B, C)
    if variant == "vanilla":
        out = stream_vanilla(p["layers"], h)
    else:
        X = np.tile(h[:, None, :], (1, n, 1)) / n              # expand to n streams
        if variant == "simplehc":
            X = stream_simplehc(p["layers"], X, K=K)
        else:
            X = STREAMS[variant](p["layers"], X)
        out = np.sum(X, axis=1)                                # collapse streams
    return np.dot(rmsnorm(out), p["unemb"])


def cross_entropy(logits, targets):
    m = np.max(logits, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.sum(np.exp(logits - m), axis=1))
    picked = logits[np.arange(len(targets)), targets]
    return np.mean(lse - picked)
