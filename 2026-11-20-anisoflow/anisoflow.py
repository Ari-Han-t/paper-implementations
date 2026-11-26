"""Minimal NumPy implementation of anisotropic learnable prediction targets for flow matching.

Interpolant (data x, noise e ~ N(0, I), t in [0, 1]):
    x_t = t * x + (1 - t) * e,        velocity v = x - e

Anisotropic target with basis Q (orthonormal columns) and per-direction weights w >= 0:
    W = Q diag(w) Q^T,                y = x - W e
w = 0 is x-prediction, w = 1 is v-prediction; each direction gets its own weight.

Closed-form map back to velocity (same network, loss and sampler):
    e = (t W + (1 - t) I)^{-1} (x_t - t y),   v = y + (W - I) e
"""
import numpy as np


def make_W(Q, w):
    return Q @ np.diag(w) @ Q.T


def target(x, e, W):
    return x - e @ W.T


def pred_to_velocity(y, xt, t, W):
    """Map predicted target y to velocity. t has shape (n, 1)."""
    d = xt.shape[1]
    n = xt.shape[0]
    I = np.eye(d)
    M = t[:, :, None] * W[None] + (1 - t)[:, :, None] * I[None]          # (n, d, d)
    rhs = xt - t * y
    e = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
    return y + e @ (W - I).T


class MLP:
    """Two-hidden-layer SiLU MLP with manual backprop and Adam."""

    def __init__(self, d_in, d_hidden, d_out, rng):
        def init(a, b):
            return rng.normal(0, np.sqrt(2.0 / a), (a, b))
        self.p = {"W1": init(d_in, d_hidden), "b1": np.zeros(d_hidden),
                  "W2": init(d_hidden, d_hidden), "b2": np.zeros(d_hidden),
                  "W3": init(d_hidden, d_out) * 0.1, "b3": np.zeros(d_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    @staticmethod
    def _silu(z):
        s = 1 / (1 + np.exp(-z))
        return z * s, s

    def forward(self, h0):
        p = self.p
        z1 = h0 @ p["W1"] + p["b1"]; h1, s1 = self._silu(z1)
        z2 = h1 @ p["W2"] + p["b2"]; h2, s2 = self._silu(z2)
        out = h2 @ p["W3"] + p["b3"]
        self.cache = (h0, z1, h1, s1, z2, h2, s2)
        return out

    def backward(self, g):
        p = self.p
        h0, z1, h1, s1, z2, h2, s2 = self.cache
        grads = {"W3": h2.T @ g, "b3": g.sum(0)}
        g2 = (g @ p["W3"].T) * (s2 * (1 + z2 * (1 - s2)))
        grads["W2"] = h1.T @ g2; grads["b2"] = g2.sum(0)
        g1 = (g2 @ p["W2"].T) * (s1 * (1 + z1 * (1 - s1)))
        grads["W1"] = h0.T @ g1; grads["b1"] = g1.sum(0)
        return grads

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.step)
            vh = self.v[k] / (1 - b2 ** self.step)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def features(xt, t):
    return np.concatenate([xt, t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)


class FlowModel:
    def __init__(self, d, W, hidden=128, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = W
        self.net = MLP(d + 3, hidden, d, self.rng)

    def predict_target(self, xt, t):
        return self.net.forward(features(xt, t))

    def velocity(self, xt, t):
        return pred_to_velocity(self.predict_target(xt, t), xt, t, self.W)

    def train(self, data, steps=3000, batch=256, lr=2e-3):
        n, d = data.shape
        losses = []
        for s in range(steps):
            x = data[self.rng.integers(0, n, batch)]
            e = self.rng.standard_normal((batch, d))
            t = self.rng.uniform(0, 1, (batch, 1))
            xt = t * x + (1 - t) * e
            y = target(x, e, self.W)
            pred = self.predict_target(xt, t)
            diff = pred - y
            losses.append(float((diff ** 2).mean()))
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * s / steps))
            self.net.adam(self.net.backward(2 * diff / diff.size), cur_lr)
        return losses

    def velocity_mse(self, data, n=4000, seed=123):
        """Held-out velocity-space error: comparable across targets."""
        rng = np.random.default_rng(seed)
        x = data[rng.integers(0, len(data), n)]
        e = rng.standard_normal(x.shape)
        t = rng.uniform(0.02, 0.98, (n, 1))
        xt = t * x + (1 - t) * e
        return float(((self.velocity(xt, t) - (x - e)) ** 2).mean())

    def sample(self, n, steps=50, seed=7):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.W.shape[0]))
        ts = np.linspace(0, 1, steps + 1)
        for i in range(steps):
            t = np.full((n, 1), ts[i])
            x = x + (ts[i + 1] - ts[i]) * self.velocity(x, t)
        return x


def sliced_wasserstein(a, b, n_proj=200, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m = min(len(a), len(b))
    pa = np.sort(a[:m] @ dirs.T, axis=0)
    pb = np.sort(b[:m] @ dirs.T, axis=0)
    return float(np.abs(pa - pb).mean())


def pca_basis(data):
    c = np.cov(data.T)
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    return vecs[:, order], vals[order]
