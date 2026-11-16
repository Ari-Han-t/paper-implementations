"""Minimal NumPy re-implementation of the core ideas in
"The Value of Thinking: Learning When to Reason in Hybrid Decision Models"
(Yu, Xu, Fang, Ammar, Luo, Wang -- arXiv:2611.16961).

Zero-or-T thinking:
  * fast mode: answer straight from one forward pass of a frozen backbone (0 tokens)
  * slow mode: run an expensive "reasoning" procedure that costs T tokens
A lightweight router reads the backbone's hidden state + fast-head output and
estimates the *value of thinking*  v(x) = E[U_slow - U_fast | implicit pass].
Slow mode runs iff v_hat(x) > lambda, where lambda is a per-request price.

Two ways to train the router:
  * supervised: regress per-decision labels v = 1[slow correct] - 1[fast correct]
  * CDPO-style (our simplified reading): policy-gradient from *episode-level*
    returns only, with the policy parameterised as sigmoid((v_hat - lambda)/tau)
    so that the learned quantity stays a calibrated value in utility units and
    lambda can be changed at test time without retraining.

The LLM is replaced by a synthetic intent-classification world (think BANKING77):
the "slow" mode is a better-informed classifier that sees T extra noisy
re-observations of the input (a stand-in for what explicit reasoning reveals).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------------
@dataclass
class World:
    K: int = 8           # number of intents / classes
    d: int = 16          # input dim
    T: int = 64          # "tokens" consumed by slow reasoning
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.means = rng.normal(0, 1.0, size=(self.K, self.d))

    def sample(self, n: int, rng: np.random.Generator):
        """Each request has a latent difficulty (noise scale) the router never sees."""
        y = rng.integers(0, self.K, size=n)
        sigma = np.where(rng.random(n) < 0.6, rng.uniform(0.3, 0.8, n), rng.uniform(1.2, 2.5, n))
        x = self.means[y] + sigma[:, None] * rng.normal(size=(n, self.d))
        return x, y, sigma

    def slow_answer(self, x, y, sigma, rng: np.random.Generator, n_obs: int = 8):
        """'Reasoning': average n_obs fresh observations then nearest-mean decode.
        Costs self.T tokens. Strictly more informed than the fast pass."""
        obs = self.means[y][:, None, :] + sigma[:, None, None] * rng.normal(size=(len(y), n_obs, self.d))
        z = np.concatenate([x[:, None, :], obs], axis=1).mean(axis=1)
        dist = ((z[:, None, :] - self.means[None]) ** 2).sum(-1)
        return dist.argmin(1)


# ----------------------------------------------------------------------------
# Frozen backbone + fast head
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Backbone:
    """Frozen random-feature network: h = tanh(W x + b). Shared by both modes."""

    def __init__(self, d, hidden=64, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1 / np.sqrt(d), size=(d, hidden))
        self.b = rng.normal(0, 0.1, size=hidden)

    def __call__(self, x):
        return np.tanh(x @ self.W + self.b)


class FastHead:
    """Linear softmax head trained once on top of the frozen backbone (fast mode)."""

    def __init__(self, hidden, K):
        self.W = np.zeros((hidden, K))
        self.b = np.zeros(K)

    def fit(self, h, y, lr=0.5, epochs=300, l2=1e-4):
        n, K = len(y), self.W.shape[1]
        Y = np.eye(K)[y]
        for _ in range(epochs):
            p = softmax(h @ self.W + self.b)
            g = (p - Y) / n
            self.W -= lr * (h.T @ g + l2 * self.W)
            self.b -= lr * g.sum(0)
        return self

    def probs(self, h):
        return softmax(h @ self.W + self.b)


def router_features(h, p):
    """What the 'implicit thinking' already produced: hidden state + output stats."""
    ps = -np.sort(-p, axis=1)
    ent = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
    margin = (ps[:, :1] - ps[:, 1:2])
    return np.concatenate([h, ps[:, :3], margin, ent], axis=1)


# ----------------------------------------------------------------------------
# Self-routing block: tiny MLP predicting value of thinking
# ----------------------------------------------------------------------------
class Router:
    def __init__(self, in_dim, hidden=32, seed=2):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(in_dim), size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), size=hidden)
        self.b2 = 0.0
        self.mu = None
        self.sd = None
        self._adam = None

    # --- forward / backward ---------------------------------------------------
    def _norm(self, f):
        if self.mu is None:
            self.mu, self.sd = f.mean(0), f.std(0) + 1e-6
        return (f - self.mu) / self.sd

    def value(self, f):
        a = np.maximum(0, self._norm(f) @ self.W1 + self.b1)
        return a @ self.w2 + self.b2

    def _grads(self, f, dv):
        """Backprop an upstream gradient dv (dL/dv_hat per sample)."""
        fn = self._norm(f)
        pre = fn @ self.W1 + self.b1
        a = np.maximum(0, pre)
        g_w2 = a.T @ dv
        g_b2 = dv.sum()
        da = np.outer(dv, self.w2) * (pre > 0)
        g_W1 = fn.T @ da
        g_b1 = da.sum(0)
        return [g_W1, g_b1, g_w2, np.array(g_b2)]

    def _step(self, grads, lr, b1=0.9, b2=0.999):
        params = [self.W1, self.b1, self.w2, np.array(self.b2)]
        if self._adam is None:
            self._adam = [[np.zeros_like(p), np.zeros_like(p)] for p in params] + [0]
        self._adam[-1] += 1
        t = self._adam[-1]
        new = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m, v = self._adam[i]
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            new.append(p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8))
        self.W1, self.b1, self.w2, self.b2 = new[0], new[1], new[2], float(new[3])

    # --- supervised training (per-decision labels available) ------------------
    def fit_supervised(self, f, v, epochs=400, lr=3e-3, batch=512, seed=0):
        rng = np.random.default_rng(seed)
        self._norm(f)
        n = len(v)
        for _ in range(epochs):
            idx = rng.choice(n, size=min(batch, n), replace=False)
            dv = 2 * (self.value(f[idx]) - v[idx]) / len(idx)
            self._step(self._grads(f[idx], dv), lr)
        return self

    # --- CDPO-style training (episode-level returns only) ---------------------
    def fit_cdpo(self, f, fast_ok, slow_ok, episodes=3000, L=16, G=8,
                 lam_range=(-0.1, 0.6), tau=0.05, lr=3e-3, calib=1.0, seed=0):
        """Each step: sample B episodes of L requests and a price lambda per episode;
        roll out G independent routing samples per episode (group baseline);
        only the episode return R = sum(correct) - lambda * #slow is observed.

        Policy: pi(slow | x, lambda) = sigmoid((v_hat(x) - lambda) / tau).
        REINFORCE gradient wrt v_hat_i is A * (a_i - p_i) / tau.
        Because lambda enters additively against v_hat, the optimum of this
        objective is v_hat(x) = E[slow_ok - fast_ok | x] -> a calibrated value.
        """
        rng = np.random.default_rng(seed)
        self._norm(f)
        n = len(fast_ok)
        B = 16
        for _ in range(episodes // B):
            idx = rng.integers(0, n, size=(B, L))
            lam = rng.uniform(*lam_range, size=B)
            fi = f[idx.ravel()]
            vh = self.value(fi).reshape(B, L)
            p = 1 / (1 + np.exp(-(vh - lam[:, None]) / tau))
            a = (rng.random((G, B, L)) < p[None]).astype(float)
            correct = np.where(a == 1, slow_ok[idx][None], fast_ok[idx][None])
            R = correct.sum(-1) - lam[None] * a.sum(-1)                    # (G, B)
            A = R - R.mean(0, keepdims=True)                                # group baseline
            A = A / (R.std(0, keepdims=True) + 1.0)                         # mild scaling
            # ascend E[R]: dv = -A * (a - p) / tau (minimise negative), averaged over groups
            dv = -(A[..., None] * (a - p[None]) / tau).mean(0).ravel() / (B * L)
            # calibration term: outside the price range the policy gradient vanishes
            # and v_hat can drift without bound; a squared hinge keeps it in a box.
            lo, hi = lam_range[0] - 2 * tau, lam_range[1] + 2 * tau
            vf = vh.ravel()
            dv = dv + calib * (2 * (np.maximum(vf - hi, 0) - np.maximum(lo - vf, 0))) / (B * L)
            self._step(self._grads(fi, dv), lr)
        return self


# ----------------------------------------------------------------------------
# Hybrid decision model and evaluation
# ----------------------------------------------------------------------------
def evaluate(decide_slow, fast_ok, slow_ok, T):
    """decide_slow: bool array. Returns accuracy, slow fraction, mean tokens."""
    correct = np.where(decide_slow, slow_ok, fast_ok)
    return dict(acc=float(correct.mean()), slow_frac=float(decide_slow.mean()),
                tokens=float(decide_slow.mean() * T))


def build(seed=0, n_train=6000, n_route=6000, n_test=6000, T=64):
    """Builds world, frozen backbone, fast head; returns data splits with
    precomputed fast/slow correctness and router features."""
    world = World(T=T, seed=seed)
    rng = np.random.default_rng(seed + 100)
    bb = Backbone(world.d, seed=seed + 1)

    xtr, ytr, _ = world.sample(n_train, rng)
    head = FastHead(64, world.K).fit(bb(xtr), ytr)

    def split(n):
        x, y, s = world.sample(n, rng)
        h = bb(x)
        p = head.probs(h)
        fast_ok = (p.argmax(1) == y).astype(float)
        slow_ok = (world.slow_answer(x, y, s, rng) == y).astype(float)
        return dict(f=router_features(h, p), p=p, fast_ok=fast_ok, slow_ok=slow_ok, sigma=s)

    return world, split(n_route), split(n_test)


def pareto(scores, fast_ok, slow_ok, T, thresholds):
    """Sweep a threshold on a 'go slow' score -> list of (threshold, metrics)."""
    return [(t, evaluate(scores > t, fast_ok, slow_ok, T)) for t in thresholds]


def area_under_tradeoff(scores, fast_ok, slow_ok):
    """Accuracy averaged over slow-call budgets 0..100% (rank-based, threshold-free).
    Higher = better router (it spends reasoning where it helps most)."""
    order = np.argsort(-scores)
    gain = (slow_ok - fast_ok)[order]
    acc = fast_ok.mean() + np.concatenate([[0], np.cumsum(gain)]) / len(gain)
    return float(acc.mean())
