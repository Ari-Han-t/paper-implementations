"""Synthetic benchmark: clustered unit-sphere embeddings, mixed query stream.

Online setting (one query at a time, like a semantic cache or a dedup check):
compare a flat brute-force scan against CapCover filter + pruned exact scan.
"""
import time
import numpy as np
from capcover import CapCoverFilter, brute_force_exists, _normalize

rng = np.random.default_rng(42)
D, N, TOPICS = 128, 50_000, 200
topic = _normalize(rng.normal(size=(TOPICS, D)))
lab = rng.integers(TOPICS, size=N)
# each topic varies along a few latent directions (low intrinsic dim) + small noise
basis = rng.normal(size=(TOPICS, 8, D)) / np.sqrt(D)
z = rng.normal(size=(N, 8)) * 0.4
data = _normalize(topic[lab] + np.einsum("nk,nkd->nd", z, basis[lab])
                  + 0.03 * rng.normal(size=(N, D)))

# queries: 50% paraphrase-like (near an indexed vector), 50% off-topic
n_q = 1000
near = _normalize(data[rng.integers(N, size=n_q // 2)] + 0.01 * rng.normal(size=(n_q // 2, D)))
far = _normalize(rng.normal(size=(n_q // 2, D)))
queries = np.vstack([near, far]).astype(np.float32)

for max_deg in [None, 35, 25]:
    t0 = time.perf_counter()
    f = CapCoverFilter(n_caps=256, max_theta=None if max_deg is None else np.radians(max_deg)).fit(data)
    X = f.data
    print(f"\n== max cap radius: {max_deg or 'unbounded'} deg | build {time.perf_counter()-t0:.1f}s | "
          f"caps={f.n_caps} | filter={f.filter_bytes()/1024:.0f} KiB = "
          f"{100*f.filter_bytes()/f.index_bytes():.2f}% of FP32 index")
    print(f"{'tau':>5} {'true+':>6} {'rejected':>9} {'FN':>3} {'FP rate':>8} {'scanned':>8} "
          f"{'brute us/q':>11} {'capcover us/q':>14} {'speedup':>8}")
    for tau in [0.80, 0.90, 0.95, 0.98]:
        truth = brute_force_exists(X, queries, tau)
        t0 = time.perf_counter()
        for q in queries:
            (X @ q >= tau).any()
        tb = (time.perf_counter() - t0) / n_q
        maybe = f.might_contain(queries, tau)
        t0 = time.perf_counter()
        ans = np.zeros(n_q, bool); scanned = 0
        for i, q in enumerate(queries):
            if f.might_contain(q, tau)[0]:
                ans[i], s = f.exists_exact(q, tau); scanned += s
        tf = (time.perf_counter() - t0) / n_q
        fn = int((truth & ~maybe).sum())
        assert fn == 0 and np.array_equal(ans, truth)
        neg = ~truth
        fp = (maybe & neg).sum() / max(neg.sum(), 1)
        print(f"{tau:5.2f} {truth.mean():6.1%} {(~maybe).mean():9.1%} {fn:3d} {fp:8.1%} "
              f"{scanned/(n_q*N):8.2%} {tb*1e6:11.0f} {tf*1e6:14.0f} {tb/tf:7.2f}x")
