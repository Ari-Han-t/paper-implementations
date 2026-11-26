# Verify Before You Fetch (VerKV) — minimal NumPy implementation

**Paper:** *Verify Before You Fetch: Reliable KV Cache Compression for LLM Inference*
**Authors:** Devanshu Jha, Gaurav Kumar, Agniva Chowdhury, Kumar Krishna Agrawal, Aditya Desai
**arXiv:** [2611.11319](https://arxiv.org/abs/2611.11319) (cs.AI, Nov 2026)

## Core idea

Long-context decoding is limited by GPU memory for the KV cache. Most compression methods keep a
fixed subset of tokens on the GPU and hope it is enough; when it isn't, quality silently drops.
VerKV keeps the **full cache in CPU memory**, and on the GPU holds only

1. a small, **query-agnostic set of pivotal tokens** chosen by stochastic (Poisson) sampling, and
2. a cheap **token-level sketch** of the whole cache.

At every decode step, **KVFetch** uses the pivots and sketch to *estimate* the attention error the
GPU-only answer would have. If that estimate meets a user-set `(eps, delta)` tolerance, no host
memory is touched; otherwise it does a query-aware fetch of the extra tokens that matter.
The paper reports 90–100% of dense quality at 32K context with an 8–32x smaller GPU footprint,
fetching 2–23% of the cache per token.

## What is implemented here (and what is simplified)

`verkv.py` (~130 lines, NumPy only), single head, single decode step:

| Paper component | This implementation |
|---|---|
| PoissonComp pivotal-token selection | Independent Bernoulli (Poisson) sampling with inclusion prob ∝ ‖k‖², water-filled to the budget; last 16 tokens always kept |
| Token-level KV sketch | Per-token int4 quantised keys + one scalar ‖v_i‖ per token |
| Attention estimate from pivots | Horvitz–Thompson reweighting (1/p_i) of sampled tokens → ratio estimator of the dense output (tested to be ~unbiased) |
| KVFetch error check | Predicted variance of the HT estimator from sketch logits and ‖v_i‖², turned into an `(eps, delta)` bound via a CLT quantile (Chebyshev optional) |
| Query-aware retrieval | Mark tokens with the largest variance contribution as certain; fetch those not resident, in geometrically growing batches, until the bound holds |

Simplified / not done: no real LLM, GPU, or CPU↔GPU transfer (host reads are just counted);
synthetic attention data; exact PoissonComp probabilities and the paper's exact error estimator are
my own reconstruction from the abstract; error is measured as ‖ô − o‖ / rms‖v‖ (value-scale
normalised) because plain relative error is ill-posed when diffuse attention averages to ≈0.

## How to run

```bash
pip install numpy pytest
python demo.py
python -m pytest -q test_verkv.py
```

## Results (real output of `demo.py`)

Synthetic cache: 4096 tokens, d=64, 40 "hot" tokens grouped into 8 topics. 200 queries: 150 focused
on one topic (peaked attention), 50 diffuse. `viol>eps` = fraction of queries whose true error
exceeded the tolerance (target ≤ delta = 10%).

```
error metric: ||o_hat - o_dense|| / rms(||v||)

== cache: 4096 tokens, d=64; 200 queries; resident pivots ~128 (3.1% of cache), delta=0.1 ==
method                        mean err   p95 err   max err  viol>eps   fetch/q  %cache
pivots only (no verify)         0.6059    1.1500    1.2122      nan%       0.0   0.00%
VerKV eps=0.1                   0.0235    0.0658    0.0726     0.00%     506.6  12.37%
VerKV eps=0.05                  0.0136    0.0401    0.0575     1.50%     891.8  21.77%
VerKV eps=0.02                  0.0052    0.0135    0.0221     1.00%    1141.1  27.86%

== cache: 4096 tokens, d=64; 200 queries; resident pivots ~256 (6.2% of cache), delta=0.1 ==
method                        mean err   p95 err   max err  viol>eps   fetch/q  %cache
pivots only (no verify)         0.5258    1.1110    1.1876      nan%       0.0   0.00%
VerKV eps=0.1                   0.0241    0.0685    0.1370     2.00%      98.7   2.41%
VerKV eps=0.05                  0.0097    0.0248    0.0300     0.00%     774.3  18.90%
VerKV eps=0.02                  0.0037    0.0119    0.0142     0.00%    1017.1  24.83%

== cache: 4096 tokens, d=64; 200 queries; resident pivots ~512 (12.5% of cache), delta=0.1 ==
method                        mean err   p95 err   max err  viol>eps   fetch/q  %cache
pivots only (no verify)         0.5166    0.9446    1.0090      nan%       0.0   0.00%
VerKV eps=0.1                   0.0179    0.0509    0.0545     0.00%       4.7   0.11%
VerKV eps=0.05                  0.0112    0.0331    0.0450     0.00%     413.9  10.10%
VerKV eps=0.02                  0.0031    0.0106    0.0158     0.00%     903.8  22.06%

budget 512, eps=0.05: focused queries fetched median 6 rows (0% needed zero fetches); diffuse queries median 1414 rows
```

Tests: `6 passed` (exactness at full budget, Poisson budget/recency, sketch accuracy, HT
unbiasedness, tolerance respected at the delta level, fetch accounting).

**Takeaways**

- Static pivots alone are badly wrong for focused queries (mean error 0.5–0.6): query-agnostic
  selection misses the few tokens each query actually needs.
- Verification fixes this at every budget: the observed violation rate stays well under delta.
- Cost adapts to the query: with 512 pivots and eps=0.05, focused queries fetch a median of
  **6 rows**, while diffuse queries fetch ~1.4K rows (their error is sampling noise over the whole
  cache). With eps=0.1, fetches fall to 0.1% of the cache.
- The bound is conservative (actual p95 error is ~1.5–3x below eps), so fetch volume could be
  reduced with a tighter estimator, which is where the paper's PoissonComp/KVFetch details matter.
