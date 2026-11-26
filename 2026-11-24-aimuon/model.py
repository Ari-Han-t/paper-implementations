"""Tiny MLP with manual backprop, plus a synthetic teacher-student regression task."""
import numpy as np


def make_task(seed=0, n=2048, d_in=32, d_hidden_teacher=64, d_out=8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))
    W1 = rng.standard_normal((d_in, d_hidden_teacher)) / np.sqrt(d_in)
    W2 = rng.standard_normal((d_hidden_teacher, d_out)) / np.sqrt(d_hidden_teacher)
    Y = np.tanh(X @ W1) @ W2
    return X, Y


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed + 1000)
        self.params = {}
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.params[f"W{i}"] = rng.standard_normal((a, b)) / np.sqrt(a)
            self.params[f"b{i}"] = np.zeros(b)
        self.L = len(sizes) - 1

    def loss_and_grads(self, X, Y):
        acts, h = [X], X
        for i in range(self.L):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            h = np.tanh(z) if i < self.L - 1 else z
            acts.append(h)
        err = h - Y
        loss = 0.5 * np.mean(np.sum(err ** 2, axis=1))
        grads, d = {}, err / X.shape[0]
        for i in reversed(range(self.L)):
            grads[f"W{i}"] = acts[i].T @ d
            grads[f"b{i}"] = d.sum(0)
            if i > 0:
                d = (d @ self.params[f"W{i}"].T) * (1 - acts[i] ** 2)
        return loss, grads
