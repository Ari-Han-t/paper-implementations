"""PoolDPP (Prada-Corral et al., 2026), minimal NumPy version.

Target: a k-DPP over the (huge) space of complete strings with kernel
    L(x, y) = p(x)^{b/2} * S(x, y) * p(y)^{b/2}
where p is the LM likelihood (quality) and S an embedding similarity (diversity).

The space can't be enumerated, so PoolDPP:
  1. draws a pool of N strings from a tractable proposal q,
  2. importance-weights each draw: w_i = p(x_i)^b / (N q(x_i)),
  3. builds the pool kernel  L~_ij = sqrt(w_i) S(x_i, x_j) sqrt(w_j),
  4. runs an exact k-DPP sampler (or greedy MAP) on L~.
Because E[sum_{i: x_i = x} w_i] = p(x)^b, the pool k-DPP approaches the target
as N grows (the paper proves KL = O(1/N)).
"""
import numpy as np
from kdpp import sample_kdpp, greedy_map


def rbf_similarity(E, bandwidth=1.0):
    d2 = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2 * bandwidth ** 2))


def target_kernel(logp, E, beta=1.0, bandwidth=1.0):
    q = np.exp(0.5 * beta * logp)
    return q[:, None] * rbf_similarity(E, bandwidth) * q[None, :]


def pool_kernel(pool_ids, logp, logq, E, beta=1.0, bandwidth=1.0):
    N = len(pool_ids)
    logw = beta * logp[pool_ids] - logq[pool_ids] - np.log(N)
    w = np.exp(0.5 * logw)
    S = rbf_similarity(E[pool_ids], bandwidth)
    return w[:, None] * S * w[None, :]


def pool_dpp(sample_proposal, logp, logq, E, k, N, beta=1.0, bandwidth=1.0,
             mode="sample", rng=None):
    """Return k item ids from the universe chosen by PoolDPP.

    sample_proposal(N, rng) -> array of N item ids drawn i.i.d. from q.
    In a real LM setting logp/logq would be sequence log-likelihoods and E
    sentence embeddings; here they are arrays indexed by item id.
    """
    rng = np.random.default_rng(rng)
    pool = sample_proposal(N, rng)
    # a k-DPP needs >= k distinct items in the pool (duplicates are rank-deficient)
    while len(np.unique(pool)) < k:
        pool = np.concatenate([pool, sample_proposal(N, rng)])
    Lt = pool_kernel(pool, logp, logq, E, beta, bandwidth)
    Lt += 1e-10 * np.eye(len(pool))
    sel = greedy_map(Lt, k) if mode == "map" else sample_kdpp(Lt, k, rng)
    return pool[sel]
