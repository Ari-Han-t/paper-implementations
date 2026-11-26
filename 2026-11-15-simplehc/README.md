# SimpleHC: Simplified Hyper-Connections (minimal NumPy implementation)

**Paper:** *On the Design of Hyper-Connections: Simplification, Stability, and Scalability*
**Authors:** Xiangdong Zhang, Xiaohan Qin, Sunan Zou, Tuo Dai, Bizhao Shi, Zhuo Xia, Yebin Yang, Yizhou Chen, Yuliang Liu, Yu Cheng, Junchi Yan
**arXiv:** [2611.05984](https://arxiv.org/abs/2611.05984) (trending on Hugging Face Daily Papers, Nov 2026)

## Core idea

Hyper-Connections (HC / mHC) replace the single residual stream with `n` parallel streams
`X ∈ R^{n×C}`. Each sublayer reads `u = Xᵀ h_pre`, computes `o = F(u)`, and writes back
`X ← H_res X + h_post oᵀ`, with gates computed dynamically from `RMSNorm(vec X)`. mHC adds a
Sinkhorn-projected mixing matrix `H_res`. Quality improves, but every sublayer reads the wide
`n×C` state several times and writes it once, so training gets 7–27% slower even though extra
FLOPs are under 1%.

The paper finds that (a) learned read/write gates *without* mixing (`H_res = I`) already match
mHC, and (b) the real cost is memory traffic. **SimpleHC** therefore:

1. drops mixing (`H_res = I`), so the state is purely additive: `X_j = X_L + Σ_{k<j} h_post_k o_kᵀ`;
2. computes the gates for a group of `K` sublayers once, from the **group-entry state** `X_L`;
3. does one **batched read** `V = X_Lᵀ H_pre` (C×K) for the whole group;
4. recovers each sublayer's exact input with **scalar corrections**
   `u_a = V[:,a] + Σ_{b<a} D[a,b]·o_b`, where `D = tril(H_preᵀ H_post, −1)` is only K×K;
5. does a single **deferred write-back** `X_{L+K} = X_L + H_post O`.

With `n = K = 4`, full-state reads per group drop from 8 to 2 and writes from 4 to 1; the paper
reports only +2.4% end-to-end training time over a vanilla residual at 15B/30B MoE scale.

## What's implemented

`simplehc.py` (NumPy via `autograd.numpy`, so the same code trains):

- `stream_vanilla`, `stream_mhc` (dynamic gates + Sinkhorn `H_res`), `stream_identityhc`
  (`H_res = I`, gates recomputed every sublayer), `stream_simplehc` (grouped execution),
  and `stream_simplehc_sequential` (reference: sequential execution with group-entry gates).
- mHC-style gates: `ℓ = α·Wᵀ RMSNorm(vec X) + b`, `h_pre = σ(ℓ)`, `h_post = 2σ(ℓ)`.
- An `AccessCounter` that counts full residual-state reads/writes.
- A tiny next-character model (embedding → 8 MLP sublayers → unembed) for each variant.

**Simplified vs. the paper:** MLP-only sublayers (no attention/MoE), a toy dataset (synthetic
order-2 Markov source over 8 symbols, so the optimal loss is known), C=32, 400 Adam steps,
2 seeds. No fused kernels — NumPy cannot show the memory-bandwidth win in wall-clock, so the
access counts are the meaningful efficiency metric here. Gate initialisation (b=0, α=0.01) is my
choice, not the paper's.

## How to run

```bash
pip install numpy autograd pytest
python demo.py        # ~1-2 min on CPU
python -m pytest -q   # exactness + bookkeeping tests
```

## Results (real output)

```
source entropy floor (val): 1.2810 nats | unigram-uniform: 2.0794
model: 8 MLP sublayers, C=32, n=4 streams, K=4, 400 Adam steps

variant      params   train     val  max|g| g-jitter time(s)
vanilla       36352  1.3888  1.4257    3.78    0.103     4.4
mhc           61144  1.3791  1.3944    7.24    0.103    35.6
identityhc    44624  1.3774  1.3946    7.26    0.104    13.9
simplehc      44624  1.3723  1.3898    7.25    0.098    13.5

residual-stream overhead (F = identity, B=1024, C=512, n=4, 8 sublayers):
  identityhc (per-sublayer)     175.3 ms   full-state reads=16 writes= 8
  simplehc (grouped K=4)        173.7 ms   full-state reads= 4 writes= 2
```

Tests: `8 passed in 0.80s`.

**Takeaways (toy scale, 2 seeds — directional only):**

- All hyper-connection variants beat the vanilla residual (val 1.39 vs 1.43 nats; floor 1.28).
- SimpleHC matches mHC and identityHC (1.390 vs 1.394/1.395) despite computing gates only once per
  group of 4 and having no mixing — consistent with the paper's claim that mixing adds little.
- Grouped execution is **bit-for-bit equivalent** (≤1e-10, forward and gradients) to sequential
  execution with group-entry gates, and cuts full-state reads 4× and writes 4×, exactly the
  8→2 / 4→1 counts in the paper. In unfused NumPy the wall-clock is roughly equal; the speedup
  needs fused kernels where the n×C state traffic dominates.
- mHC's Sinkhorn mixing is the slowest variant here (2.6× SimpleHC's training time).
