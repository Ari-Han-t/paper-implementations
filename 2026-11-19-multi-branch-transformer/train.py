"""Tiny training utilities: char corpus, Adam, train/eval loop."""
import glob, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from mbt import loss_fn, init_params


def load_corpus(max_chars=600_000):
    """Deterministic char-level corpus: Python stdlib sources (always available offline)."""
    root = os.path.dirname(os.__file__)
    text = []
    for f in sorted(glob.glob(os.path.join(root, "*.py"))):
        try:
            text.append(open(f, encoding="utf-8").read())
        except Exception:
            pass
        if sum(map(len, text)) > max_chars:
            break
    s = "".join(text)[:max_chars]
    chars = sorted(set(s))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in s], dtype=np.int32)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, rng, bs, T):
    ix = rng.integers(0, len(data) - T - 1, bs)
    return jnp.asarray(np.stack([data[i:i + T + 1] for i in ix]))


def train(cfg, train_data, val_data, steps=600, bs=32, lr=3e-3, seed=0, log_every=0):
    p = init_params(cfg, jax.random.PRNGKey(seed))
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    b1, b2 = 0.9, 0.98

    @jax.jit
    def step(p, m, v, batch, t):
        loss, g = jax.value_and_grad(lambda q: loss_fn(cfg, q, batch))(p)
        lr_t = lr * jnp.minimum(1.0, t / 50) * 0.5 * (1 + jnp.cos(jnp.pi * t / steps))
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        p = jax.tree_util.tree_map(
            lambda w, a, b: w - lr_t * (a / (1 - b1 ** t)) / (jnp.sqrt(b / (1 - b2 ** t)) + 1e-8),
            p, m, v)
        return p, m, v, loss

    eval_loss = jax.jit(lambda p, b: loss_fn(cfg, p, b))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for t in range(1, steps + 1):
        p, m, v, loss = step(p, m, v, get_batch(train_data, rng, bs, cfg.seq_len), t)
        if log_every and t % log_every == 0:
            print(f"  step {t} loss {float(loss):.3f}", flush=True)
    vrng = np.random.default_rng(1234)
    vl = float(np.mean([eval_loss(p, get_batch(val_data, vrng, 64, cfg.seq_len)) for _ in range(20)]))
    return p, vl, time.time() - t0
