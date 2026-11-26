# RoCoKV: Uncertainty- and Redundancy-Aware KV Cache Compression (minimal NumPy version)

**Paper:** *RoCoKV: Accelerating LLM Inference Serving through Uncertainty and Redundancy Aware KV Cache Compression*
**Authors:** Payman Behnam, Elaheh Sobhani, Saad Amin, Yahya Hassan, Daniyal Khan, Biswadeep Chakraborty, Tanish Patwa, Souvik Kundu, Alexey Tumanov
**arXiv:** [2611.25433](https://arxiv.org/abs/2611.25433) (v1, 19 Nov 2026)

## Core idea

Reasoning models produce very long traces, so the KV cache dominates memory and decode cost. Those traces repeat themselves (loops, self-verification, corrections). The repeated tokens can still draw a lot of attention, so heavy-hitter methods like H2O waste their budget keeping several copies of the same thought. RoCoKV adds two signals:

1. **Redundancy-aware selection.** A token's importance score is down-weighted when its key closely matches other recent content, so one copy of a repeated thought is kept and the rest are evicted.
2. **Uncertainty-guided adaptive pruning.** The cache budget grows when the model is uncertain (high next-token entropy) and shrinks when it is confident.

The paper also ships fused CUDA decode kernels and a vLLM plugin. It reports 1.27× speedup and 38% higher throughput vs. full KV on AIME25 with DeepSeek-R1-Distill-Llama-8B.

## What is implemented vs. simplified

Only the abstract was accessible, so the formulas below are **our interpretation**, not the paper's exact equations.

| Component | This repo |
|---|---|
| Base importance | H2O-style accumulated attention, normalised by token age, plus a protected recent window |
| Redundancy | For each cached key: max cosine similarity to any *later* cached key. Score multiplied by `(1 − λ·redundancy)` |
| Uncertainty | Entropy of the next-token distribution from a (random) LM head. Smoothed with an EMA, z-scored online, mapped to a budget in `[0.5B, 1.5B]` |
| CUDA kernels / vLLM / paged KV | Not reproduced |
| Real LLM + reasoning benchmarks | Replaced by a synthetic single-head "reasoning trace" with injected loops |

Files:

- `rocokv.py` — `CompressedKVCache` with the `h2o` and `rocokv` policies
- `demo.py` — synthetic benchmark
- `test_rocokv.py` — unit tests

## How to run

```bash
pip install numpy pytest
python demo.py               # ~8 s on CPU
python -m pytest -q test_rocokv.py
```

## Results (real output, `demo_output.txt`)

Setup:

- An 800-token trace with d=32, built from 120 "facts"; about 35% of chunks are near-duplicate loops of recent content.
- Queries look up earlier facts, alternating between confident and uncertain phases.
- Metrics are KL(full ‖ compressed) and top-1 agreement of next-token distributions through a random 500-way LM head, averaged over 5 seeds.

```
--- average KV budget = 64 (8% of trace) ---
method                                      KL(full||comp)  top1 agree avg KV held
H2O (heavy hitters + recent)                        0.7030       0.412        64.0
+ redundancy-aware selection                        0.7248       0.384        64.0
+ uncertainty-adaptive budget (RoCoKV)              0.7911       0.386        64.7
  (redundancy-only, matched memory)                 0.7020       0.389        65.0

--- average KV budget = 128 (16% of trace) ---
H2O (heavy hitters + recent)                        0.3284       0.584       128.0
+ redundancy-aware selection                        0.2993       0.587       128.0
+ uncertainty-adaptive budget (RoCoKV)              0.4374       0.551       111.4
  (redundancy-only, matched memory)                 0.3397       0.543       111.0

--- average KV budget = 256 (32% of trace) ---
H2O (heavy hitters + recent)                        0.1471       0.754       256.0
+ redundancy-aware selection                        0.1117       0.770       256.0
+ uncertainty-adaptive budget (RoCoKV)              0.1833       0.705       190.8
  (redundancy-only, matched memory)                 0.1723       0.685       191.0
```

Tests: `5 passed in 0.10s`.

### Takeaways

- **Redundancy-aware selection helps when the budget is moderate.** At the same memory it cuts KL vs. H2O by about 9% at 16% budget and about 24% at 32% budget. At a tiny 8% budget it is slightly worse: almost everything is evicted anyway, and dropping duplicate copies also removes the attention mass those copies carried.
- **The adaptive budget is the subtle part.** Eviction is irreversible, so a per-token budget that flips up and down collapses to its minimum. (Our first unsmoothed version was clearly worse than a fixed budget.) Even with EMA smoothing, the cache actually holds less than the nominal average budget: 191 vs. 256 at the largest setting. At matched memory it lands roughly level with a fixed budget (slightly worse KL, slightly better top-1). In this toy setting we do **not** reproduce a gain from uncertainty adaptation. A real implementation probably needs lookahead or reversible offloading (keeping evicted KV in CPU memory) to make "spend more memory when uncertain" pay off.
- The synthetic task and random LM head are a stand-in, not real LLM evidence. The numbers show the mechanics, not the paper's claims.
