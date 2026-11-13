"""A small character-level MLP language model (Bengio-style) in pure NumPy.

    x = concat(E[c_{t-k}], ..., E[c_{t-1}])     (context of k chars)
    h = tanh(x @ W1 + b1)
    logits = h @ W2 + b2

Small enough to train on a laptop CPU in a couple of minutes, large enough
that 4-bit weights visibly change some of its predictions.
"""
import numpy as np

WEIGHT_NAMES = ("E", "W1", "W2")  # matrices that get quantized; biases stay fp32


def init_params(vocab, ctx=8, emb=24, hidden=384, seed=0):
    rng = np.random.default_rng(seed)
    d_in = ctx * emb
    return {
        "E": rng.normal(0, 1.0, (vocab, emb)).astype(np.float32),
        "W1": (rng.normal(0, 1, (d_in, hidden)) / np.sqrt(d_in)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "W2": (rng.normal(0, 1, (hidden, vocab)) / np.sqrt(hidden)).astype(np.float32),
        "b2": np.zeros(vocab, np.float32),
    }


def forward(params, X):
    """X: (B, ctx) int array -> logits (B, V) plus cache for backward."""
    emb = params["E"][X]                      # (B, ctx, emb)
    x = emb.reshape(X.shape[0], -1)           # (B, ctx*emb)
    h = np.tanh(x @ params["W1"] + params["b1"])
    logits = h @ params["W2"] + params["b2"]
    return logits, (X, x, h)


def log_softmax(logits):
    z = logits - logits.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def loss_and_grads(params, X, Y):
    logits, (Xc, x, h) = forward(params, X)
    lp = log_softmax(logits)
    B = X.shape[0]
    loss = -lp[np.arange(B), Y].mean()
    d = np.exp(lp)
    d[np.arange(B), Y] -= 1.0
    d /= B
    g = {}
    g["W2"] = h.T @ d
    g["b2"] = d.sum(0)
    dh = (d @ params["W2"].T) * (1 - h * h)
    g["W1"] = x.T @ dh
    g["b1"] = dh.sum(0)
    dx = (dh @ params["W1"].T).reshape(B, Xc.shape[1], -1)
    gE = np.zeros_like(params["E"])
    np.add.at(gE, Xc, dx)
    g["E"] = gE
    return float(loss), g


def make_windows(ids, ctx):
    ids = np.asarray(ids, dtype=np.int64)
    idx = np.arange(ctx)[None, :] + np.arange(len(ids) - ctx)[:, None]
    return ids[idx], ids[ctx:]


def train(params, X, Y, steps=6000, batch=256, lr=3e-3, seed=0, log_every=1000, verbose=True):
    """Plain Adam with cosine decay."""
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        bi = rng.integers(0, len(X), batch)
        loss, g = loss_and_grads(params, X[bi], Y[bi])
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / steps))
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** t)
            vh = v[k] / (1 - b2 ** t)
            params[k] -= (lr_t * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        if verbose and (t % log_every == 0 or t == 1):
            print(f"  step {t:5d}  loss {loss:.3f}")
    return params
