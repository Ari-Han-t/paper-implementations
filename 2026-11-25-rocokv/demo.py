"""Synthetic 'reasoning trace' benchmark for RoCoKV-style KV compression.

A trace is built from a pool of distinct 'facts' (key/value directions). The
model periodically *loops*: it re-emits near-duplicate copies of a recent chunk
(self-verification / repetition), which receive lots of attention but add no new
information. At each decode step a query looks up one fact; we compare the
compressed-cache attention output to the full-cache output through a random
'LM head' and report KL divergence and top-1 agreement of next-token
distributions.
"""
import numpy as np
from rocokv import CompressedKVCache, attend, softmax, entropy


def make_trace(T=800, d=32, n_facts=120, loop_prob=0.35, seed=0):
    rng = np.random.default_rng(seed)
    facts_k = rng.normal(size=(n_facts, d)); facts_v = rng.normal(size=(n_facts, d))
    K, V, fid = [], [], []
    while len(K) < T:
        if len(K) > 40 and rng.random() < loop_prob:      # loop: repeat a recent chunk
            L = rng.integers(8, 25); s = rng.integers(max(0, len(K) - 120), len(K) - L)
            for i in range(s, s + L):
                K.append(K[i] + 0.05 * rng.normal(size=d)); V.append(V[i] + 0.05 * rng.normal(size=d)); fid.append(fid[i])
        else:                                             # new reasoning content
            for _ in range(rng.integers(5, 15)):
                f = rng.integers(n_facts)
                K.append(facts_k[f] + 0.3 * rng.normal(size=d)); V.append(facts_v[f] + 0.3 * rng.normal(size=d)); fid.append(f)
    K, V, fid = np.array(K[:T]), np.array(V[:T]), np.array(fid[:T])
    # queries: look up a fact seen earlier in the trace, with variable sharpness
    Q = np.zeros((T, d))
    sharp = 1.5
    for t in range(T):
        f = fid[rng.integers(0, t + 1)]
        if rng.random() < 0.03:                            # switch between confident / uncertain phases
            sharp = 0.4 if sharp == 1.5 else 1.5
        Q[t] = sharp * facts_k[f] + 0.5 * rng.normal(size=d)
    return K, V, Q


def run(policy, budget, K, V, Q, W, **kw):
    T, d = K.shape
    cache = CompressedKVCache(d, budget, policy=policy, **kw) if policy != "full" else None
    kls, agree = [], []
    for t in range(T):
        full_out, _ = attend(Q[t], K[: t + 1], V[: t + 1])
        p_full = softmax(W @ full_out)
        if cache is None:
            continue
        cache.append(K[t], V[t])
        out = cache.query(Q[t])
        p = softmax(W @ out)
        kls.append(float((p_full * (np.log(p_full + 1e-12) - np.log(p + 1e-12))).sum()))
        agree.append(p.argmax() == p_full.argmax())
        cache.compress(entropy(p))                         # model's own uncertainty
    # mean number of entries actually held once the cache is full (fair memory comparison)
    sizes = np.array(cache.size_trace)
    avg_b = sizes[budget:].mean()
    return np.mean(kls), np.mean(agree), avg_b


def main():
    d, T = 32, 800
    rng = np.random.default_rng(123)
    W = rng.normal(size=(500, d)) * 0.6                    # random LM head, vocab 500
    configs = [
        ("H2O (heavy hitters + recent)", "h2o", {}),
        ("+ redundancy-aware selection", "rocokv", dict(adaptive=False)),
        ("+ uncertainty-adaptive budget (RoCoKV)", "rocokv", dict(adaptive=True)),
    ]
    print(f"Trace length {T}, d={d}, 5 seeds averaged\n")
    for budget in (64, 128, 256):
        print(f"--- average KV budget = {budget} ({budget / T:.0%} of trace) ---")
        print(f"{'method':42s} {'KL(full||comp)':>15s} {'top1 agree':>11s} {'avg KV held':>11s}")
        for name, pol, kw in configs:
            res = np.array([run(pol, budget, *make_trace(T, d, seed=s), W, **kw) for s in range(5)])
            kl, ag, ab = res.mean(0)
            print(f"{name:42s} {kl:15.4f} {ag:11.3f} {ab:11.1f}")
            if kw.get("adaptive"):   # fixed-budget redundancy policy at the SAME memory
                mb = int(round(ab))
                res = np.array([run("rocokv", mb, *make_trace(T, d, seed=s), W, adaptive=False) for s in range(5)])
                kl2, ag2, ab2 = res.mean(0)
                print(f"{'  (redundancy-only, matched memory)':42s} {kl2:15.4f} {ag2:11.3f} {ab2:11.1f}")
        print()


if __name__ == "__main__":
    main()
