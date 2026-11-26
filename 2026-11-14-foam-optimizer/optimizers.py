"""Minimal NumPy optimizers: SGD-momentum, Adam, SOAP and FOAM.

All optimizers act on a dict of parameters {name: np.ndarray}. Matrix
parameters are assumed to be laid out as (out_features, in_features), so a
gradient G has shape (m, n) with m = outputs, n = inputs.

FOAM (Yu et al., 2026, arXiv:2611.16905) = SOAP with
  1. input-only (one-sided) preconditioning: only the input-side Kronecker
     factor R = EMA(G^T G) and its eigenbasis Q_R are kept;
  2. a factored second moment: instead of a dense (m, n) Adam second moment in
     the rotated basis, keep a row vector r (m) and column vector c (n) and
     reconstruct V ~= r c^T / sum(r) (Adafactor-style rank-1 estimate).
"""
import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.t = 0
        self.state = {k: {} for k in params}

    def state_floats(self):
        """Number of floats stored in optimizer state (excluding params)."""
        return sum(v.size for s in self.state.values() for v in s.values()
                   if isinstance(v, np.ndarray))

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self._update(k, params[k], grads[k], self.state[k])

    def _update(self, k, p, g, s):
        raise NotImplementedError


class SGDM(Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.9):
        super().__init__(params, lr)
        self.mu = momentum

    def _update(self, k, p, g, s):
        s.setdefault("m", np.zeros_like(p))
        s["m"] = self.mu * s["m"] + g
        p -= self.lr * s["m"]


def _adam_vector(p, g, s, lr, b1, b2, eps, t):
    s.setdefault("m", np.zeros_like(p))
    s.setdefault("v", np.zeros_like(p))
    s["m"] = b1 * s["m"] + (1 - b1) * g
    s["v"] = b2 * s["v"] + (1 - b2) * g * g
    mh = s["m"] / (1 - b1 ** t)
    vh = s["v"] / (1 - b2 ** t)
    p -= lr * mh / (np.sqrt(vh) + eps)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps

    def _update(self, k, p, g, s):
        _adam_vector(p, g, s, self.lr, self.b1, self.b2, self.eps, self.t)


def _eigvecs(A):
    # Symmetric PSD -> orthonormal eigenvectors (descending eigenvalues).
    w, Q = np.linalg.eigh(A)
    return Q[:, ::-1]


class SOAP(Optimizer):
    """Two-sided SOAP: Adam run in the eigenbasis of Shampoo's L and R."""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, precond_freq=10):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.sb, self.eps, self.freq = shampoo_beta, eps, precond_freq

    def _update(self, k, p, g, s):
        if p.ndim != 2:
            return _adam_vector(p, g, s, self.lr, self.b1, self.b2, self.eps, self.t)
        m, n = p.shape
        if "L" not in s:
            s["L"] = np.zeros((m, m)); s["R"] = np.zeros((n, n))
            s["QL"] = np.eye(m); s["QR"] = np.eye(n)
            s["m"] = np.zeros_like(p); s["v"] = np.zeros_like(p)
        s["L"] = self.sb * s["L"] + (1 - self.sb) * g @ g.T
        s["R"] = self.sb * s["R"] + (1 - self.sb) * g.T @ g
        if self.t == 1 or self.t % self.freq == 0:
            s["QL"], s["QR"] = _eigvecs(s["L"]), _eigvecs(s["R"])
        QL, QR = s["QL"], s["QR"]
        s["m"] = self.b1 * s["m"] + (1 - self.b1) * g          # original basis
        g_rot = QL.T @ g @ QR
        m_rot = QL.T @ s["m"] @ QR
        s["v"] = self.b2 * s["v"] + (1 - self.b2) * g_rot ** 2  # dense (m, n)
        mh = m_rot / (1 - self.b1 ** self.t)
        vh = s["v"] / (1 - self.b2 ** self.t)
        p -= self.lr * (QL @ (mh / (np.sqrt(vh) + self.eps)) @ QR.T)


class FOAM(Optimizer):
    """Factored One-sided Adam-Moment (simplified re-implementation).

    one_sided=False / factored=False flags let you ablate each change.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, precond_freq=10, factored=True):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.sb, self.eps, self.freq = shampoo_beta, eps, precond_freq
        self.factored = factored

    def _update(self, k, p, g, s):
        if p.ndim != 2:
            return _adam_vector(p, g, s, self.lr, self.b1, self.b2, self.eps, self.t)
        m, n = p.shape
        if "R" not in s:
            s["R"] = np.zeros((n, n)); s["QR"] = np.eye(n)
            s["m"] = np.zeros_like(p)
            if self.factored:
                s["vr"] = np.zeros(m); s["vc"] = np.zeros(n)
            else:
                s["v"] = np.zeros_like(p)
        # (1) input-side Kronecker factor only
        s["R"] = self.sb * s["R"] + (1 - self.sb) * g.T @ g
        if self.t == 1 or self.t % self.freq == 0:
            s["QR"] = _eigvecs(s["R"])
        QR = s["QR"]
        s["m"] = self.b1 * s["m"] + (1 - self.b1) * g
        g_rot = g @ QR                # rotate input side only
        m_rot = s["m"] @ QR
        g2 = g_rot ** 2
        bc2 = 1 - self.b2 ** self.t
        if self.factored:
            # (2) factored second moment: row / column statistics
            s["vr"] = self.b2 * s["vr"] + (1 - self.b2) * g2.mean(axis=1)
            s["vc"] = self.b2 * s["vc"] + (1 - self.b2) * g2.mean(axis=0)
            vh = np.outer(s["vr"], s["vc"]) / (s["vr"].mean() + 1e-30) / bc2
        else:
            s["v"] = self.b2 * s["v"] + (1 - self.b2) * g2
            vh = s["v"] / bc2
        mh = m_rot / (1 - self.b1 ** self.t)
        p -= self.lr * ((mh / (np.sqrt(vh) + self.eps)) @ QR.T)


def factored_second_moment(g2):
    """Rank-1 reconstruction used by FOAM (exposed for testing)."""
    r, c = g2.mean(axis=1), g2.mean(axis=0)
    return np.outer(r, c) / r.mean()
