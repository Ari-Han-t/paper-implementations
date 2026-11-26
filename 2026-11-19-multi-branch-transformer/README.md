# Multi-Branch Transformer (MBT) + Joint-KV

**Paper:** *Multi-Branch Transformer* — Gaotang Li, Lingjie Chen, Chunyuan Deng, Zhiwei Xu, Hongzheng Chen, Christopher De Sa, Jingrui He, Yingheng Wang, Hanghang Tong
**arXiv:** [2611.19800](https://arxiv.org/abs/2611.19800) (Nov 2026)

## Core idea

Transformers usually scale by getting **wider** or **deeper**. MBT adds a third axis, **branching**: a shared *prelude* of blocks feeds `K` independently parameterised *branches* (each a stack of Transformer blocks) that run **in parallel**. Their outputs are fused and passed through a shared *coda*:

```
H = P(X);   Z_i = G_i(H), i=1..K;   Y = Dec(C(F(Z_1..Z_K)))
F = W_F · concat(RMSNorm(Z_1), …, RMSNorm(Z_K))      (paper's fusion)
```

Under a matched compute budget you can split the middle blocks as **Wide-Shallow** (same width, fewer blocks per branch → fewer sequential steps), **Narrow-Deep** (same depth, narrower), or **Balanced** (in between).

**Joint-KV** synchronises branches at each depth and builds a *single* shared K/V from all branches' normalised inputs, `KV_l = (1/√K)[S_1;…;S_K] W_l`, while queries stay branch-local. Parameter count and FLOPs are identical to per-branch KV, but the branch-region KV cache shrinks by `K×`. The paper reports up to 1.26× compute-equivalent gains, up to 34% less KV cache and up to 51.4% higher decode throughput on an H200 (0.8B–2.1B MoE models).

## What's implemented vs. simplified

| Paper | Here |
|---|---|
| Prelude → K parallel branches → fusion → coda | ✅ same structure (`mbt.py`) |
| Concat + per-branch RMSNorm + bias-free `W_F` fusion | ✅ plus a `mean` fusion for ablation |
| Wide-Shallow / Narrow-Deep / Balanced at matched FLOPs | ✅ (`d=64`; widths 64/44/52 chosen so body FLOPs ≈ standard) |
| Shared input projection when branch width ≠ model width | ✅ |
| Joint-KV (one shared K/V projection per depth, queries local) | ✅ with param-count and `1/K` cache tests |
| MoE FFNs, RoPE, key-norm | ❌ dense GELU FFN, learned positions |
| 0.8–2.1B params, 200 tok/param, inference engine with branch grouping | ❌ ~0.3M params, char-level LM on Python stdlib source, 600 steps, CPU (JAX) |

## How to run

```bash
pip install jax numpy pytest      # CPU JAX is enough
python -m pytest -q test_mbt.py   # 7 tests
python demo.py 600                # ~25 min on 2 CPU cores (6 variants x 2 seeds)
```

## Results (real run, 2 CPU cores)

Tests: `7 passed in 11.26s` (shapes, causality incl. Joint-KV, Joint-KV param parity, branch-region cache = 1/K, independent branch weights, depth bookkeeping, gradients reach every branch).

Demo — char-level LM, 540K training chars, vocab 110, 600 Adam steps, batch 32, context 64, val loss over 20×64 sequences:

| Variant | Params | FLOPs/tok | Seq. depth | KV cache vs Std | Val loss (mean ± std, 2 seeds) |
|---|---|---|---|---|---|
| Standard (6 seq blocks) | 306,880 | 590K | 6 | 1.00x | 2.3755 ± 0.0054 |
| MBT Wide-Shallow K=2 (2x2@64) | 315,200 | 606K | 4 | 1.00x | 2.3973 ± 0.0050 |
| MBT Balanced K=2 (3x2@52) | 315,160 | 606K | 5 | 1.15x | 2.4693 ± 0.0078 |
| MBT Narrow-Deep K=2 (4x2@44) | 304,856 | 585K | 6 | 1.25x | 2.4735 ± 0.0100 |
| MBT Balanced K=2 + Joint-KV | 315,160 | 606K | 5 | **0.74x** | 2.4609 ± 0.0061 |
| MBT Wide-Shallow K=2, mean fusion | 306,880 | 590K | 4 | 1.00x | **2.3635 ± 0.0127** |

**Reading the numbers honestly:**

- At this toy scale the paper's headline (MBT beats a standard Transformer at matched compute) is **not** reproduced for the concat-fusion variants: Wide-Shallow is ~0.02 nats worse, Balanced/Narrow-Deep ~0.09 worse (the narrower 44–52-wide branches hurt a lot when the whole model is only 64 wide).
- Wide-Shallow with plain **mean fusion** roughly ties/edges the standard model (2.364 vs 2.376, within ~1σ) while needing only **4 sequential blocks instead of 6** — the latency benefit of branching survives even when quality is flat.
- **Joint-KV is a free win here:** it slightly *improves* Balanced (2.461 vs 2.469) while cutting the full-model KV cache to **0.74×** of the standard model (and to 0.64× of independent-KV Balanced), with identical parameters and FLOPs — consistent with the paper's 19–34% cache reductions.
- 600 steps / 0.3M params is far from the paper's regime; treat this as a correctness-checked reference implementation, not a scaling study.

## Files

- `mbt.py` — model (standard + MBT + Joint-KV), FLOP and KV-cache accounting
- `train.py` — corpus loader, Adam with warmup + cosine, train/eval loop
- `demo.py` — the experiment above (`demo_output.txt` holds the raw log)
- `test_mbt.py` — unit tests
