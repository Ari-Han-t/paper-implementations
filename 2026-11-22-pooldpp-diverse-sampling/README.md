# A Model of Diverse Sampling from Language Models (PoolDPP)

**Authors:** Manuel Prada-Corral, Yahya Emara, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira (NeurIPS 2026)
**arXiv:** [2611.25003](https://arxiv.org/abs/2611.25003)

## Core idea
When you ask an LM for K answers, i.i.d. sampling keeps returning near-duplicates of the mode.
The paper treats "give me K diverse outputs" as sampling a size-K **subset** of complete strings from a
**determinantal point process (k-DPP)**: the LM likelihood supplies *quality*, an embedding similarity
supplies *repulsion*, and the subset probability is `det(L_Y)` with `L(x,y) = p(x)^{b/2} S(x,y) p(y)^{b/2}`.

The string space is far too large to build L, so **PoolDPP**:
1. draws a pool of N candidates from a cheap proposal `q` (e.g. temperature sampling),
2. importance-weights each draw, `w_i = p(x_i)^b / (N q(x_i))`,
3. builds the pool kernel `L~_ij = sqrt(w_i) S(x_i,x_j) sqrt(w_j)`,
4. runs an exact k-DPP sampler on `L~` (or greedy MAP for a deterministic set).

The paper shows the KL to the target k-DPP shrinks as O(1/N) and reports better quality-diversity
trade-offs on open-ended generation, text-to-SQL (Ambrosia) and GSM8K at roughly equal pass@K.

## What's implemented
- `kdpp.py` — exact k-DPP sampling (elementary symmetric polynomials + projection-DPP sampling,
  Kulesza & Taskar 2012) and greedy MAP, NumPy only.
- `pooldpp.py` — the PoolDPP estimator: importance-weighted pool kernel + k-DPP / MAP selection.
- `toy_lm.py` — a toy "LM" over 40 strings with 2-D embeddings in 5 semantic clusters; the target `p` is
  mode-collapsed on one cluster, the proposal `q` is `p` at temperature 2.
- `demo.py` — the two experiments below. `test_pooldpp.py` — 6 tests.

**Simplified:** no real LM or sentence encoder (toy universe instead, so the exact target k-DPP can be
enumerated and compared against); RBF similarity on embeddings; b = 1; no paper-specific tuning.

## How to run
```bash
pip install numpy pytest
python -m pytest -q      # 6 passed
python demo.py           # ~75 s on CPU
```

## Results (actual output)
```
Experiment 1: convergence to target 3-DPP (universe M=40, 3000 draws per N)
  Monte-Carlo noise floor (exact k-DPP samples): TV = 0.0197
  N     TV(marginals)
  10    0.0384
  25    0.0285
  50    0.0260
  100   0.0191
  200   0.0219
  400   0.0196

Experiment 2: quality vs diversity of K=3 outputs (1000 trials, N=100)
  method         mean log p  distinct  clusters  pair dist
  iid from p         -2.912      2.78      1.79      2.158
  PoolDPP            -3.299      3.00      2.38      3.349
  PoolDPP-MAP        -3.085      3.00      2.99      4.336
```
- PoolDPP's inclusion marginals converge to the exact target k-DPP's; by N ≈ 100 the gap is at the
  Monte-Carlo noise floor of 3000 draws, so larger pools can't be told apart from exact sampling here.
- i.i.d. sampling returns duplicates (2.78 distinct of 3) and covers only 1.79 clusters. PoolDPP-MAP
  covers nearly all 3 clusters with 2× the pairwise spread, losing ~0.17 nats of mean log-likelihood.
