"""Minimal NumPy implementation of Muon and AIMuon (Muon + spectral residual correction).

AIMuon idea (arXiv 2611.27077): Muon steps along msign(momentum). Momentum lags the
current gradient, so AIMuon also orthogonalizes the gradient-momentum residual and
mixes the two spectral directions, weighting each by its dual pairing with the
gradient (spectral/nuclear-norm duality), then rescales to Muon's update norm.

The exact weighting formula is not in the abstract; the version here is a faithful-
in-spirit reconstruction (see README).
"""
import numpy as np


def newton_schulz(G, steps=5, eps=1e-7):
    """Approximate msign(G) = U V^T via the quintic Newton-Schulz iteration used by Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + eps)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transpose else X


def exact_msign(G):
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


class Muon:
    def __init__(self, lr=0.02, beta=0.95, nesterov=True, ns_steps=5):
        self.lr, self.beta, self.nesterov, self.ns_steps = lr, beta, nesterov, ns_steps
        self.m = {}

    def _scale(self, shape):
        # match RMS of update across shapes (as in common Muon implementations)
        return max(1.0, shape[0] / shape[1]) ** 0.5

    def direction(self, key, G):
        m = self.m.get(key, np.zeros_like(G))
        m = self.beta * m + G
        self.m[key] = m
        v = G + self.beta * m if self.nesterov else m
        return newton_schulz(v, self.ns_steps)

    def step(self, key, W, G):
        W -= self.lr * self._scale(W.shape) * self.direction(key, G)


class AIMuon(Muon):
    """Muon + spectral residual correction.

    O1 = msign(v)            (Muon direction, v = Nesterov momentum)
    R  = G - m               (gradient-momentum residual: what momentum is missing)
    O2 = msign(R)
    weights via duality: <G, O_i> = how much each unit-spectral-norm direction
    decreases the loss to first order. Only positive pairings are used.
    U  = w1 O1 + gamma * w2 O2, rescaled to ||O1||_F (Muon's update norm).
    """

    def __init__(self, lr=0.02, beta=0.95, nesterov=True, ns_steps=5, gamma=0.25):
        super().__init__(lr, beta, nesterov, ns_steps)
        self.gamma = gamma

    def direction(self, key, G):
        m_prev = self.m.get(key, np.zeros_like(G))
        m = self.beta * m_prev + G
        self.m[key] = m
        v = G + self.beta * m if self.nesterov else m
        O1 = newton_schulz(v, self.ns_steps)
        R = G - (1 - self.beta) * m  # residual vs. the EMA-normalised momentum
        O2 = newton_schulz(R, self.ns_steps)
        w1 = max(np.sum(G * O1), 0.0)
        w2 = max(np.sum(G * O2), 0.0)
        if w1 + w2 == 0:
            return O1
        U = w1 * O1 + self.gamma * w2 * O2
        return U * (np.linalg.norm(O1) / (np.linalg.norm(U) + 1e-12))


class SGDM:
    def __init__(self, lr=0.05, beta=0.9):
        self.lr, self.beta, self.m = lr, beta, {}

    def step(self, key, W, G):
        m = self.beta * self.m.get(key, np.zeros_like(G)) + G
        self.m[key] = m
        W -= self.lr * m
