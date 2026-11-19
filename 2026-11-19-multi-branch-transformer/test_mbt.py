import math
import jax, jax.numpy as jnp
import numpy as np
from mbt import Config, init_params, forward, count_params, loss_fn

V = 20
KEY = jax.random.PRNGKey(0)
toks = jnp.asarray(np.random.default_rng(0).integers(0, V, 16))


def cfgs():
    return [Config(vocab=V, seq_len=16, K=1, branch_depth=2),
            Config(vocab=V, seq_len=16, K=2, branch_depth=2),
            Config(vocab=V, seq_len=16, K=3, branch_depth=2, d_b=48),
            Config(vocab=V, seq_len=16, K=2, branch_depth=2, d_b=48, joint_kv=True),
            Config(vocab=V, seq_len=16, K=2, branch_depth=2, fusion="mean")]


def test_shapes_and_finite():
    for c in cfgs():
        out = forward(c, init_params(c, KEY), toks)
        assert out.shape == (16, V) and bool(jnp.all(jnp.isfinite(out)))


def test_causal_including_joint_kv():
    # changing a future token must not change earlier logits (Joint-KV shares K/V across branches only, not time)
    for c in cfgs():
        p = init_params(c, KEY)
        a = forward(c, p, toks)
        b = forward(c, p, toks.at[10].set((toks[10] + 1) % V))
        assert np.allclose(a[:10], b[:10], atol=1e-5)
        assert not np.allclose(a[10:], b[10:], atol=1e-5)


def test_joint_kv_same_param_count():
    ind = Config(vocab=V, K=3, branch_depth=2, d_b=48)
    jnt = Config(vocab=V, K=3, branch_depth=2, d_b=48, joint_kv=True)
    assert count_params(init_params(ind, KEY)) == count_params(init_params(jnt, KEY))


def test_joint_kv_cache_is_1_over_K_in_branch_region():
    for K in (2, 3, 4):
        ind = Config(vocab=V, K=K, branch_depth=3, d_b=48, prelude=0, coda=0)
        jnt = Config(vocab=V, K=K, branch_depth=3, d_b=48, prelude=0, coda=0, joint_kv=True)
        assert math.isclose(jnt.kv_cache_per_token() / ind.kv_cache_per_token(), 1 / K)


def test_branches_are_independent_params():
    c = Config(vocab=V, K=2, branch_depth=2)
    p = init_params(c, KEY)
    assert not np.allclose(p["branches"][0][0]["wq"], p["branches"][0][1]["wq"])


def test_depth_bookkeeping():
    c = Config(vocab=V, K=4, branch_depth=3, prelude=1, coda=1)
    assert c.physical_blocks() == 14 and c.sequential_depth() == 5


def test_gradients_reach_every_branch():
    c = Config(vocab=V, seq_len=16, K=2, branch_depth=2, joint_kv=True)
    p = init_params(c, KEY)
    g = jax.grad(lambda q: loss_fn(c, q, toks[None]))(p)
    for l, depth in enumerate(g["branches"]):
        assert float(jnp.abs(g["joint_wkv"][l]).sum()) > 0
        for blk in depth:
            assert float(jnp.abs(blk["wq"]).sum()) > 0
