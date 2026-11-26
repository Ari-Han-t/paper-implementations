"""A toy 'language model' over a finite universe of M strings.

Each string has a 2-D 'embedding' living in one of several semantic clusters.
The target p is sharply peaked (mode-collapsed) on cluster 0; the proposal q is
p at a higher temperature, which is the usual cheap proposal for an LM.
"""
import numpy as np


def make_toy_lm(M=40, n_clusters=5, temperature=2.0, seed=0):
    rng = np.random.default_rng(seed)
    cluster = np.arange(M) % n_clusters
    centers = 3.0 * np.stack([np.cos(2 * np.pi * np.arange(n_clusters) / n_clusters),
                              np.sin(2 * np.pi * np.arange(n_clusters) / n_clusters)], 1)
    E = centers[cluster] + 0.35 * rng.standard_normal((M, 2))
    logits = -1.2 * cluster + 0.5 * rng.standard_normal(M)
    logp = logits - np.logaddexp.reduce(logits)
    logits_q = logits / temperature
    logq = logits_q - np.logaddexp.reduce(logits_q)
    q = np.exp(logq)

    def sample_proposal(N, rng):
        return rng.choice(M, size=N, p=q)

    return dict(E=E, logp=logp, logq=logq, cluster=cluster,
                sample_proposal=sample_proposal, M=M)
