"""Minimal Multi-Branch Transformer (MBT) in JAX, with Joint-KV.

Paper: "Multi-Branch Transformer" (Li et al., arXiv 2611.19800).

    H   = P(X)                       shared prelude (+ input projection d -> d_b if needed)
    Z_i = G_i(H),  i = 1..K          K independently parameterised branches (L_b blocks each)
    Y   = Dec(C(F(Z_1..Z_K)))        fusion F, shared coda C, decoder

Fusion (paper Eq. 4): RMSNorm each branch output, concat on hidden dim, bias-free
linear W_F -> d. A "mean" fusion is included for ablations.

Joint-KV (paper Eq. 6): at branch depth l, keys/values are produced once from the
concatenation of all branches' normalised inputs, scaled by 1/sqrt(K):
    KV_l = (1/sqrt K) [S_1,l ; ... ; S_K,l] W_l,   W_l in R^{(K d_b) x 2 d_kv}
Queries stay branch-local. Same param count / FLOPs as independent KV, but the
branch-region KV cache shrinks by a factor K.

Simplification vs. paper: dense FFN instead of MoE, no RoPE (learned positions),
tiny widths, char-level data.
"""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int
    seq_len: int = 64
    d: int = 64             # shared (prelude / coda) width
    n_heads: int = 4
    prelude: int = 1        # shared blocks before the branches
    coda: int = 1           # shared blocks after fusion
    K: int = 1              # number of branches; K=1 with branch_depth=L -> standard transformer
    branch_depth: int = 4   # blocks per branch
    d_b: int = 64           # branch width
    fusion: str = "concat"  # "concat" (paper) | "mean"
    joint_kv: bool = False

    @property
    def is_standard(self):
        return self.K == 1

    def sequential_depth(self):
        return self.prelude + self.branch_depth + self.coda

    def physical_blocks(self):
        return self.prelude + self.K * self.branch_depth + self.coda

    def kv_cache_per_token(self):
        """Number of cached K/V floats per token (keys + values)."""
        shared = (self.prelude + self.coda) * 2 * self.d
        if self.is_standard:
            return shared + self.branch_depth * 2 * self.d_b
        per_depth = 2 * self.d_b if self.joint_kv else self.K * 2 * self.d_b
        return shared + self.branch_depth * per_depth


# ---------------------------------------------------------------- params
def _dense(key, n_in, n_out):
    return jax.random.normal(key, (n_in, n_out)) * (1.0 / math.sqrt(n_in))


def init_block(key, d, own_kv=True):
    """Pre-norm block. own_kv=False for Joint-KV branch blocks (K/V come from a shared projection)."""
    ks = jax.random.split(key, 6)
    blk = {
        "n1": jnp.ones(d), "n2": jnp.ones(d),
        "wq": _dense(ks[0], d, d),
        "wo": _dense(ks[2], d, d) * 0.5,
        "w1": _dense(ks[3], d, 4 * d),
        "w2": _dense(ks[4], 4 * d, d) * 0.5,
    }
    if own_kv:
        blk["wkv"] = _dense(ks[1], d, 2 * d)
    return blk


def init_params(cfg: Config, key):
    keys = iter(jax.random.split(key, 64 + cfg.K * cfg.branch_depth))
    p = {
        "emb": jax.random.normal(next(keys), (cfg.vocab, cfg.d)) * 0.02,
        "pos": jax.random.normal(next(keys), (cfg.seq_len, cfg.d)) * 0.02,
        "prelude": [init_block(next(keys), cfg.d) for _ in range(cfg.prelude)],
        "coda": [init_block(next(keys), cfg.d) for _ in range(cfg.coda)],
        "nf": jnp.ones(cfg.d),
    }
    if cfg.is_standard:
        p["middle"] = [init_block(next(keys), cfg.d_b) for _ in range(cfg.branch_depth)]
        if cfg.d_b != cfg.d:
            raise ValueError("standard model must have d_b == d")
        return p
    # branches[l][i] = block of branch i at depth l
    p["branches"] = [[init_block(next(keys), cfg.d_b, own_kv=not cfg.joint_kv) for _ in range(cfg.K)]
                     for _ in range(cfg.branch_depth)]
    if cfg.joint_kv:
        # one shared K/V projection per depth: (K*d_b) -> 2*d_b  (same #params as K independent ones)
        p["joint_wkv"] = [_dense(next(keys), cfg.K * cfg.d_b, 2 * cfg.d_b)
                          for _ in range(cfg.branch_depth)]
    if cfg.d_b != cfg.d:
        p["in_proj"] = _dense(next(keys), cfg.d, cfg.d_b)       # shared input projection
    if cfg.fusion == "concat":
        p["fuse_norm"] = [jnp.ones(cfg.d_b) for _ in range(cfg.K)]
        p["W_F"] = _dense(next(keys), cfg.K * cfg.d_b, cfg.d)
    elif cfg.d_b != cfg.d:
        p["out_proj"] = _dense(next(keys), cfg.d_b, cfg.d)
    return p


def count_params(p):
    return sum(x.size for x in jax.tree_util.tree_leaves(p))


# ---------------------------------------------------------------- forward
def rmsnorm(x, g, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * g


def attention(blk, xn, n_heads, kv=None):
    """kv: optional precomputed (T, 2d) keys/values (Joint-KV); else computed from xn."""
    T, d = xn.shape
    hd = d // n_heads
    q = (xn @ blk["wq"]).reshape(T, n_heads, hd)
    if kv is None:
        kv = xn @ blk["wkv"]
    k, v = kv[:, :d].reshape(T, n_heads, hd), kv[:, d:].reshape(T, n_heads, hd)
    att = jnp.einsum("thd,shd->hts", q, k) / math.sqrt(hd)
    mask = jnp.tril(jnp.ones((T, T), bool))
    att = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
    out = jnp.einsum("hts,shd->thd", att, v).reshape(T, d)
    return out @ blk["wo"]


def block_fwd(blk, x, n_heads, kv=None):
    xn = rmsnorm(x, blk["n1"])
    x = x + attention(blk, xn, n_heads, kv)
    h = rmsnorm(x, blk["n2"])
    return x + jax.nn.gelu(h @ blk["w1"]) @ blk["w2"]


def forward(cfg: Config, p, tokens):
    """tokens: (T,) int -> logits (T, vocab). vmap for batches."""
    T = tokens.shape[0]
    x = p["emb"][tokens] + p["pos"][:T]
    for blk in p["prelude"]:
        x = block_fwd(blk, x, cfg.n_heads)
    if cfg.is_standard:
        for blk in p["middle"]:
            x = block_fwd(blk, x, cfg.n_heads)
    else:
        h = x @ p["in_proj"] if "in_proj" in p else x
        zs = [h] * cfg.K
        for l, depth in enumerate(p["branches"]):
            if cfg.joint_kv:
                # synchronise branches at this depth: ONE shared K/V (this is what gets cached)
                s = jnp.concatenate([rmsnorm(z, blk["n1"]) for z, blk in zip(zs, depth)], -1)
                kv = (s / math.sqrt(cfg.K)) @ p["joint_wkv"][l]
                zs = [block_fwd(blk, z, cfg.n_heads, kv=kv) for z, blk in zip(zs, depth)]
            else:
                zs = [block_fwd(blk, z, cfg.n_heads) for z, blk in zip(zs, depth)]
        if cfg.fusion == "concat":
            cat = jnp.concatenate([rmsnorm(z, g) for z, g in zip(zs, p["fuse_norm"])], -1)
            x = cat @ p["W_F"]
        else:
            x = sum(zs) / cfg.K
            if "out_proj" in p:
                x = x @ p["out_proj"]
    for blk in p["coda"]:
        x = block_fwd(blk, x, cfg.n_heads)
    x = rmsnorm(x, p["nf"])
    return x @ p["emb"].T          # tied decoder


def loss_fn(cfg, p, batch):
    logits = jax.vmap(lambda t: forward(cfg, p, t))(batch[:, :-1])
    tgt = batch[:, 1:]
    lp = jax.nn.log_softmax(logits, -1)
    return -jnp.mean(jnp.take_along_axis(lp, tgt[..., None], -1))


def flops_per_token(cfg: Config):
    """Rough matmul FLOPs/token (2*params of matmuls touched), excluding attention scores."""
    def blk(d):
        return 2 * (d * d + d * 2 * d + d * d + 8 * d * d)
    f = (cfg.prelude + cfg.coda) * blk(cfg.d)
    if cfg.is_standard:
        return f + cfg.branch_depth * blk(cfg.d)
    f += cfg.K * cfg.branch_depth * blk(cfg.d_b)   # Joint-KV: same matmul FLOPs (K d_b x 2 d_b once)
    if cfg.d_b != cfg.d:
        f += 2 * cfg.d * cfg.d_b
    if cfg.fusion == "concat":
        f += 2 * cfg.K * cfg.d_b * cfg.d
    return f
