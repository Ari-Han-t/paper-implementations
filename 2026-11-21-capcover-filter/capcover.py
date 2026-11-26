"""Minimal CapCover-style filter: semantic approximate membership over embeddings.

Question answered: "is there any indexed vector x with cos(q, x) >= tau?"
The filter may say "maybe" when the answer is no (false positive), but it
never says "no" when the answer is yes (no false negatives).

Construction (tau-independent):
  1. L2-normalise the vectors and group them with spherical k-means.
  2. Each group becomes a spherical cap (centre c_k, angular radius theta_k),
     where theta_k is the largest angle between c_k and any member.
  3. Centres are stored as int8 (per-cap scale); theta_k is computed against
     the *dequantised* centre, so quantisation never breaks the guarantee.

Query with threshold tau (phi = arccos tau):
  If cos(q, x) >= tau then angle(q, x) <= phi, and by the triangle inequality
  on the sphere angle(q, c_k) <= theta_k + phi for the cap containing x.
  So if angle(q, c_k) > theta_k + phi for every cap, q is rejected safely.
  Otherwise the surviving caps are the only places a match can live, which
  also gives exact pruned search.
"""
import numpy as np


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def spherical_kmeans(x, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ centers.T, axis=1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].sum(0)
            else:  # re-seed empty cluster
                centers[j] = x[rng.integers(len(x))]
        centers = _normalize(centers)
    assign = np.argmax(x @ centers.T, axis=1)
    return centers, assign


class CapCoverFilter:
    def __init__(self, n_caps=256, max_theta=None, min_size=8, iters=20, seed=0):
        self.n_caps, self.iters, self.seed = n_caps, iters, seed
        self.max_theta, self.min_size = max_theta, min_size

    def _refine(self, x, centers, assign):
        """Split caps wider than max_theta (radians) with 2-means until tight."""
        groups = [np.nonzero(assign == j)[0] for j in range(len(centers))]
        out, stack = [], [g for g in groups if len(g)]
        while stack:
            g = stack.pop()
            c = _normalize(x[g].sum(0))
            width = np.arccos(np.clip(x[g] @ c, -1, 1).min())
            if width <= self.max_theta or len(g) <= self.min_size:
                out.append(g)
                continue
            _, a = spherical_kmeans(x[g], 2, iters=8, seed=len(stack))
            if a.min() == a.max():
                out.append(g)
                continue
            stack += [g[a == 0], g[a == 1]]
        centers = _normalize(np.stack([x[g].sum(0) for g in out]))
        assign = np.empty(len(x), dtype=np.int64)
        for j, g in enumerate(out):
            assign[g] = j
        return centers, assign

    def fit(self, vectors):
        x = _normalize(vectors)
        self.data = x
        centers, assign = spherical_kmeans(x, self.n_caps, self.iters, self.seed)
        if self.max_theta is not None:
            centers, assign = self._refine(x, centers, assign)
        self.n_caps = len(centers)
        # int8 quantisation of centres
        scale = np.abs(centers).max(axis=1, keepdims=True) / 127.0
        self.q_centers = np.round(centers / scale).astype(np.int8)
        self.scale = scale.astype(np.float32)
        deq = _normalize(self.q_centers.astype(np.float32) * self.scale)
        self._deq = deq
        # radius vs. dequantised centre (keeps the guarantee exact)
        theta = np.zeros(self.n_caps, dtype=np.float32)
        self.members = []
        for j in range(self.n_caps):
            idx = np.nonzero(assign == j)[0]
            self.members.append(idx)
            if len(idx):
                cos = np.clip(x[idx] @ deq[j], -1.0, 1.0)
                theta[j] = np.arccos(cos.min())
            else:
                theta[j] = -np.inf  # empty cap never matches
        # tiny float slack so rounding can't create false negatives
        self.theta = theta + 1e-5
        # store vectors grouped by cap so a cap's members are one contiguous slice
        order = np.concatenate(self.members)
        self.sorted_data = x[order]
        sizes = np.array([len(m) for m in self.members])
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        return self

    def filter_bytes(self):
        return self.q_centers.nbytes + self.scale.nbytes + self.theta.nbytes

    def index_bytes(self):
        return self.data.nbytes

    def candidate_caps(self, queries, tau):
        q = _normalize(np.atleast_2d(queries))
        phi = np.arccos(np.clip(tau, -1.0, 1.0))
        ang = np.arccos(np.clip(q @ self._deq.T, -1.0, 1.0))
        return ang <= (self.theta + phi)[None, :]

    def might_contain(self, queries, tau):
        """False => definitely no indexed vector with cos >= tau."""
        return self.candidate_caps(queries, tau).any(axis=1)

    def exists_exact(self, query, tau):
        """Exact answer using only vectors from surviving caps."""
        q = _normalize(query)
        caps = np.nonzero(self.candidate_caps(q, tau)[0])[0]
        if len(caps) == 0:
            return False, 0
        scanned = 0
        for c in caps:  # early exit on first hit
            a, b = self.offsets[c], self.offsets[c + 1]
            scanned += b - a
            if (self.sorted_data[a:b] @ q >= tau).any():
                return True, scanned
        return False, scanned


def brute_force_exists(data, queries, tau):
    return (_normalize(queries) @ data.T >= tau).any(axis=1)
