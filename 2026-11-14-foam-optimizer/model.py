"""Tiny 2-layer tanh MLP with manual backprop (NumPy)."""
import numpy as np


def init(d_in, d_h, d_out, rng):
    return {
        "W1": rng.standard_normal((d_h, d_in)) / np.sqrt(d_in),
        "b1": np.zeros(d_h),
        "W2": rng.standard_normal((d_out, d_h)) / np.sqrt(d_h),
        "b2": np.zeros(d_out),
    }


def forward_backward(p, X, Y):
    """MSE loss; X (B, d_in), Y (B, d_out). Returns loss, grads."""
    B = X.shape[0]
    H_pre = X @ p["W1"].T + p["b1"]
    H = np.tanh(H_pre)
    out = H @ p["W2"].T + p["b2"]
    diff = out - Y
    loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))
    d_out = diff / B
    g = {"W2": d_out.T @ H, "b2": d_out.sum(0)}
    dH = d_out @ p["W2"] * (1 - H ** 2)
    g["W1"] = dH.T @ X
    g["b1"] = dH.sum(0)
    return loss, g


def make_task(n, d_in, d_out, rng, cond=100.0):
    """Teacher-student regression with ill-conditioned, correlated inputs."""
    U, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    scales = np.logspace(0, -np.log10(cond) / 2, d_in)   # input std spread
    X = rng.standard_normal((n, d_in)) * scales @ U.T
    teacher = init(d_in, 32, d_out, rng)
    for k in ("W1",):
        teacher[k] = teacher[k] * 3.0
    H = np.tanh(X @ (teacher["W1"] @ U @ np.diag(1 / scales) @ U.T).T)
    Y = H @ teacher["W2"].T
    return X, Y
