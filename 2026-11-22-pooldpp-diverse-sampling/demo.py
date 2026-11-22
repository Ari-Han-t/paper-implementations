"""Experiments for the PoolDPP implementation.

1. Convergence: PoolDPP's item-inclusion marginals approach the exact target
   k-DPP's marginals as the pool size N grows (paper: KL = O(1/N)).
2. Quality vs diversity: i.i.d. sampling from p vs PoolDPP vs PoolDPP-MAP.
"""
import itertools
import time
import numpy as np
from kdpp import sample_kdpp
from pooldpp import pool_dpp, target_kernel
from toy_lm import make_toy_lm

K, R, BW = 3, 3000, 1.0


def exact_marginals(L, k):
    M = L.shape[0]
    marg, Z = np.zeros(M), 0.0
    for S in itertools.combinations(range(M), k):
        d = np.linalg.det(L[np.ix_(S, S)])
        Z += d
        marg[list(S)] += d
    return marg / Z


def empirical_marginals(samples, M):
    m = np.zeros(M)
    for s in samples:
        m[np.unique(s)] += 1
    return m / len(samples)


def tv(a, b, k):  # total variation between inclusion marginals normalised to sum 1
    return 0.5 * np.abs(a / k - b / k).sum()


def main():
    t0 = time.time()
    lm = make_toy_lm()
    E, logp, logq, M = lm["E"], lm["logp"], lm["logq"], lm["M"]
    L = target_kernel(logp, E, bandwidth=BW)
    exact = exact_marginals(L, K)
    rng = np.random.default_rng(1)

    noise = tv(empirical_marginals([sample_kdpp(L, K, rng) for _ in range(R)], M), exact, K)
    print(f"Experiment 1: convergence to target {K}-DPP (universe M={M}, {R} draws per N)")
    print(f"  Monte-Carlo noise floor (exact k-DPP samples): TV = {noise:.4f}")
    print("  N     TV(marginals)")
    rows = []
    for N in [10, 25, 50, 100, 200, 400]:
        samples = [pool_dpp(lm["sample_proposal"], logp, logq, E, K, N, bandwidth=BW, rng=rng)
                   for _ in range(R)]
        d = tv(empirical_marginals(samples, M), exact, K)
        rows.append((N, d))
        print(f"  {N:<5} {d:.4f}")

    print("\nExperiment 2: quality vs diversity of K=3 outputs (1000 trials, N=100)")
    cl = lm["cluster"]

    def stats(sets):
        q = np.mean([logp[s].mean() for s in sets])
        distinct = np.mean([len(np.unique(s)) for s in sets])
        clusters = np.mean([len(np.unique(cl[s])) for s in sets])
        pdist = np.mean([np.mean([np.linalg.norm(E[a] - E[b]) for a, b in itertools.combinations(s, 2)])
                         for s in sets])
        return q, distinct, clusters, pdist

    p = np.exp(logp)
    iid = [rng.choice(M, size=K, p=p) for _ in range(1000)]
    pd = [pool_dpp(lm["sample_proposal"], logp, logq, E, K, 100, bandwidth=BW, rng=rng) for _ in range(1000)]
    mp = [pool_dpp(lm["sample_proposal"], logp, logq, E, K, 100, bandwidth=BW, mode="map", rng=rng)
          for _ in range(1000)]
    print(f"  {'method':<14}{'mean log p':>11}{'distinct':>10}{'clusters':>10}{'pair dist':>11}")
    for name, sets in [("iid from p", iid), ("PoolDPP", pd), ("PoolDPP-MAP", mp)]:
        q, d, c, pdd = stats(sets)
        print(f"  {name:<14}{q:>11.3f}{d:>10.2f}{c:>10.2f}{pdd:>11.3f}")
    print(f"\n(total time {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
