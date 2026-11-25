"""Minimal NumPy re-implementation of the two ideas in RoCoKV (arXiv 2611.25433):

1. Uncertainty-guided adaptive budget: keep more KV entries at decode steps where
   the model is uncertain (high next-token entropy), prune harder when confident,
   while holding the *average* budget fixed.
2. Redundancy-aware token selection: tokens whose keys are near-duplicates of
   other (recent / already-kept) tokens get their importance score down-weighted,
   so loops and self-verification repeats stop hogging the cache.

Only the abstract was available, so the exact formulas here are our own
interpretation (documented in README.md). The CUDA kernels / vLLM plugin are not
reproduced.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p, eps=1e-12):
    return float(-(p * np.log(p + eps)).sum())


def attend(q, K, V):
    """Single-head scaled dot-product attention for one query."""
    w = softmax(K @ q / np.sqrt(q.shape[0]))
    return w @ V, w


class CompressedKVCache:
    """KV cache with pluggable eviction.

    policy: 'h2o'    -> accumulated attention (heavy hitters) + recent window
            'rocokv' -> heavy hitters with redundancy penalty, optional adaptive budget
    """

    def __init__(self, d, budget, policy="rocokv", recent=16, lam=0.8,
                 adaptive=True, alpha=1.0, min_frac=0.5, max_frac=1.5, sim_thresh=0.0,
                 ema=0.9):
        self.d, self.base_budget, self.policy = d, budget, policy
        self.recent, self.lam = recent, lam
        self.adaptive, self.alpha = adaptive and policy == "rocokv", alpha
        self.min_frac, self.max_frac = min_frac, max_frac
        self.sim_thresh = sim_thresh
        self.K = np.zeros((0, d)); self.V = np.zeros((0, d))
        self.score = np.zeros(0)
        self.u_mean, self.u_var, self.n_u = 0.0, 1.0, 0
        self.ema, self.u_smooth = ema, None
        self.budget_trace, self.size_trace = [], []

    # ---- uncertainty-guided budget -------------------------------------
    def budget_for(self, u):
        if not self.adaptive:
            return self.base_budget
        # Eviction is irreversible, so a budget that flips every step degenerates
        # to its minimum. Smooth the signal so budgets follow *phases* of
        # uncertainty rather than single noisy tokens.
        self.u_smooth = u if self.u_smooth is None else self.ema * self.u_smooth + (1 - self.ema) * u
        u = self.u_smooth
        # running z-score of the uncertainty signal (Welford) -> budget scale
        self.n_u += 1
        delta = u - self.u_mean
        self.u_mean += delta / self.n_u
        self.u_var += (delta * (u - self.u_mean) - self.u_var) / self.n_u
        z = (u - self.u_mean) / (np.sqrt(self.u_var) + 1e-6) if self.n_u > 5 else 0.0
        frac = np.clip(1.0 + self.alpha * np.tanh(z) * 0.5, self.min_frac, self.max_frac)
        return int(round(self.base_budget * frac))

    # ---- redundancy ----------------------------------------------------
    def redundancy(self):
        """For each cached token: max cosine similarity of its key to any *more
        recent* cached key (so the newest copy of a repeated thought is kept)."""
        Kn = self.K / (np.linalg.norm(self.K, axis=1, keepdims=True) + 1e-8)
        S = Kn @ Kn.T
        n = len(S)
        S[np.tril_indices(n)] = -1.0   # only compare against later tokens
        r = S.max(axis=1)
        r[-1] = 0.0
        return np.clip((r - self.sim_thresh) / (1 - self.sim_thresh), 0.0, 1.0)

    def append(self, k, v):
        self.K = np.vstack([self.K, k]); self.V = np.vstack([self.V, v])
        self.score = np.append(self.score, 0.0)

    def query(self, q):
        out, w = attend(q, self.K, self.V)
        self.score += w        # accumulated attention (heavy-hitter statistic)
        return out

    def compress(self, uncertainty):
        b = self.budget_for(uncertainty)
        self.budget_trace.append(b)
        n = len(self.K)
        if n <= b:
            self.size_trace.append(n)
            return
        keep_recent = np.arange(max(0, n - self.recent), n)
        cand = np.arange(0, max(0, n - self.recent))
        imp = self.score[cand] / (n - cand)          # normalise by age
        if self.policy == "rocokv":
            imp = imp * (1.0 - self.lam * self.redundancy()[cand])
        n_keep = max(0, b - len(keep_recent))
        keep = cand[np.argsort(-imp)[:n_keep]]
        idx = np.sort(np.concatenate([keep, keep_recent]))
        self.K, self.V, self.score = self.K[idx], self.V[idx], self.score[idx]
        self.size_trace.append(len(idx))
