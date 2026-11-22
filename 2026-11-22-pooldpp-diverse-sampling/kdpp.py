"""Exact k-DPP sampling and greedy MAP in NumPy.

A k-DPP with PSD kernel L assigns a size-k subset Y probability
    P(Y) = det(L_Y) / e_k(lambda_1..lambda_n)
where e_k is the k-th elementary symmetric polynomial of L's eigenvalues.
Sampling follows Kulesza & Taskar (2012), Algorithms 1 and 8.
"""
import numpy as np


def elementary_symmetric(lam, k):
    """E[l, n] = e_l(lam_1..lam_n) for l = 0..k, n = 0..N (dynamic programming)."""
    n = len(lam)
    E = np.zeros((k + 1, n + 1))
    E[0, :] = 1.0
    for l in range(1, k + 1):
        for m in range(1, n + 1):
            E[l, m] = E[l, m - 1] + lam[m - 1] * E[l - 1, m - 1]
    return E


def _select_eigenvectors(lam, k, rng):
    """Pick exactly k eigen-indices; index m is kept w.p. lam_m E[l-1,m-1]/E[l,m]."""
    E = elementary_symmetric(lam, k)
    chosen, l = [], k
    for m in range(len(lam), 0, -1):
        if l == 0:
            break
        if m == l or rng.random() < lam[m - 1] * E[l - 1, m - 1] / E[l, m]:
            chosen.append(m - 1)
            l -= 1
    return np.array(chosen[::-1], dtype=int)


def _sample_projection_dpp(V, rng):
    """Sample from the projection DPP spanned by orthonormal columns of V."""
    V = V.copy()
    Y = []
    while V.shape[1] > 0:
        probs = np.sum(V ** 2, axis=1)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        i = rng.choice(len(probs), p=probs)
        Y.append(i)
        # project the basis onto the subspace orthogonal to e_i, drop one column
        j = np.argmax(np.abs(V[i]))
        vj = V[:, j].copy()
        V = V - np.outer(vj, V[i] / vj[i])
        V = np.delete(V, j, axis=1)
        if V.shape[1] > 0:
            V, _ = np.linalg.qr(V)
    return sorted(Y)


def sample_kdpp(L, k, rng=None):
    """Draw one exact sample of size k from the k-DPP with kernel L."""
    rng = np.random.default_rng(rng)
    L = (L + L.T) / 2
    lam, U = np.linalg.eigh(L)
    lam = np.clip(lam, 0, None)
    if np.sum(lam > 1e-12) < k:
        raise ValueError("kernel rank is smaller than k")
    idx = _select_eigenvectors(lam, k, rng)
    return _sample_projection_dpp(U[:, idx], rng)


def greedy_map(L, k):
    """Greedy MAP: repeatedly add the item that maximizes log det(L_Y)."""
    Y = []
    for _ in range(k):
        best, best_val = None, -np.inf
        for i in range(L.shape[0]):
            if i in Y:
                continue
            S = Y + [i]
            sign, logdet = np.linalg.slogdet(L[np.ix_(S, S)])
            val = logdet if sign > 0 else -np.inf
            if val > best_val:
                best, best_val = i, val
        if best is None or best_val == -np.inf:
            break
        Y.append(best)
    return sorted(Y)
