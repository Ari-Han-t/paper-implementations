"""Demo: train a tiny 8-sublayer next-character model with four residual
variants on a synthetic order-2 Markov source, then benchmark residual-stream
overhead of per-sublayer identityHC vs grouped SimpleHC.

Run:  python demo.py            (about 1-2 min on a laptop CPU)
"""
import time

import numpy as onp
import autograd.numpy as np
from autograd import value_and_grad
from autograd.misc.flatten import flatten

import simplehc as shc

V, ORDER, CTX = 8, 2, 3
DEPTH, C, N, K = 8, 32, 4, 4
STEPS, BATCH, LR = 400, 128, 3e-3


# ---------------------------------------------------------------- data
def make_source(seed=0):
    """Order-2 Markov chain with peaked (Dirichlet 0.3) transition rows."""
    rng = onp.random.default_rng(seed)
    T = rng.dirichlet(onp.full(V, 0.3), size=(V,) * ORDER)
    return T


def sample(T, length, seed):
    rng = onp.random.default_rng(seed)
    s = list(rng.integers(0, V, ORDER))
    for _ in range(length):
        s.append(rng.choice(V, p=T[tuple(s[-ORDER:])]))
    return onp.array(s)


def windows(seq):
    idx = onp.arange(CTX, len(seq))
    ctx = onp.stack([seq[idx - CTX + i] for i in range(CTX)], axis=1)
    return ctx, seq[idx]


def true_entropy(T, ctx):
    """Average conditional entropy (nats) of the source on these contexts:
    the best achievable loss."""
    rows = T[ctx[:, -2], ctx[:, -1]]
    return float(onp.mean(-(rows * onp.log(rows + 1e-12)).sum(1)))


# ---------------------------------------------------------------- training
def adam(loss_grad, w, steps, batches, lr):
    m = onp.zeros_like(w); v = onp.zeros_like(w)
    losses, gnorms = [], []
    for t in range(1, steps + 1):
        xb, yb = batches(t)
        loss, g = loss_grad(w, xb, yb)
        losses.append(float(loss)); gnorms.append(float(onp.linalg.norm(g)))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - lr * (m / (1 - 0.9 ** t)) / (onp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w, losses, gnorms


def train(variant, Xtr, Ytr, Xva, Yva, seed=0):
    rng = onp.random.default_rng(seed)
    p0 = shc.init_params(rng, variant, DEPTH, C, n=N, vocab=V, ctx=CTX)
    w0, unflatten = flatten(p0)
    loss_fn = lambda w, xb, yb: shc.cross_entropy(
        shc.forward(unflatten(w), xb, variant, n=N, K=K), yb)
    lg = value_and_grad(loss_fn)
    brng = onp.random.default_rng(seed + 1)

    def batches(_):
        i = brng.integers(0, len(Ytr), BATCH)
        return Xtr[i], Ytr[i]

    t0 = time.perf_counter()
    w, losses, gnorms = adam(lg, w0, STEPS, batches, LR)
    dt = time.perf_counter() - t0
    val = float(loss_fn(w, Xva, Yva))
    g = onp.array(gnorms)
    jitter = float(onp.median(onp.abs(onp.diff(g)) / (g[:-1] + 1e-12)))
    return dict(val=val, train=onp.mean(losses[-50:]), time=dt,
                gmax=float(g.max()), jitter=jitter, nparams=w0.size)


# ---------------------------------------------------------------- overhead bench
def bench_overhead(B=1024, Cb=512, depth=8, reps=5):
    """Time only the residual-stream bookkeeping: F is the identity so the
    measurement isolates gate computation + reads/writes of the n x C state."""
    rng = onp.random.default_rng(0)
    p = shc.init_params(rng, "identityhc", depth, Cb, n=N, hidden_mult=1)
    orig_F = shc.mlp_sublayer
    shc.mlp_sublayer = lambda _p, u: u
    X = rng.normal(0, 1, (B, N, Cb))
    res = {}
    try:
        for name, fn in [("identityhc (per-sublayer)", lambda: shc.stream_identityhc(p["layers"], X)),
                         ("simplehc (grouped K=4)", lambda: shc.stream_simplehc(p["layers"], X, K=K))]:
            fn(); ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            c = shc.AccessCounter()
            (shc.stream_identityhc(p["layers"], X, counter=c) if "identity" in name
             else shc.stream_simplehc(p["layers"], X, K=K, counter=c))
            res[name] = (min(ts) * 1e3, c.reads, c.writes)
    finally:
        shc.mlp_sublayer = orig_F
    return res


if __name__ == "__main__":
    T = make_source()
    Xtr, Ytr = windows(sample(T, 30000, seed=1))
    Xva, Yva = windows(sample(T, 5000, seed=2))
    print(f"source entropy floor (val): {true_entropy(T, Xva):.4f} nats | "
          f"unigram-uniform: {onp.log(V):.4f}")
    print(f"model: {DEPTH} MLP sublayers, C={C}, n={N} streams, K={K}, {STEPS} Adam steps\n")
    print(f"{'variant':<11} {'params':>7} {'train':>7} {'val':>7} {'max|g|':>7} {'g-jitter':>8} {'time(s)':>7}")
    for v in ["vanilla", "mhc", "identityhc", "simplehc"]:
        vals = [train(v, Xtr, Ytr, Xva, Yva, seed=s) for s in (0, 1)]
        r = {k: onp.mean([x[k] for x in vals]) for k in vals[0]}
        print(f"{v:<11} {int(r['nparams']):>7} {r['train']:>7.4f} {r['val']:>7.4f} "
              f"{r['gmax']:>7.2f} {r['jitter']:>8.3f} {r['time']:>7.1f}")

    print("\nresidual-stream overhead (F = identity, B=1024, C=512, n=4, 8 sublayers):")
    for name, (ms, rd, wr) in bench_overhead().items():
        print(f"  {name:<27} {ms:7.1f} ms   full-state reads={rd:2d} writes={wr:2d}")
