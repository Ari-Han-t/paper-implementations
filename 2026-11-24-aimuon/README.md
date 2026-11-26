# Accelerating Muon with Spectral Residual Correction (AIMuon)

- **Paper:** *Accelerating Muon with Spectral Residual Correction* (2026)
- **arXiv:** [2611.27077](https://arxiv.org/abs/2611.27077)
- **Authors:** see the arXiv page

## Core idea

Muon updates each weight matrix along `msign(momentum)`: the momentum's orthogonal polar factor `U Vᵀ`, approximated with a few Newton–Schulz matmuls. Momentum lags behind the current gradient. AIMuon also orthogonalizes the **gradient–momentum residual** (the part of the fresh gradient that momentum has not absorbed yet) and mixes the two spectral directions. Each direction is weighted by its dual pairing with the gradient (spectral/nuclear-norm duality: `<G, msign(X)>` is the first-order loss decrease from a unit-spectral-norm step). The mix is then rescaled to Muon's update norm. The paper reports 6.5–10% fewer steps on GPT-2 and Qwen3-0.6B pretraining.

## What I implemented vs. simplified

| Component | Status |
|---|---|
| Quintic Newton–Schulz `msign` (Muon coefficients 3.4445, −4.7750, 2.0315) | implemented |
| Muon (Nesterov and plain momentum, shape-aware LR scale) | implemented |
| AIMuon: `O1 = msign(v)`, `R = G − (1−β)m`, `O2 = msign(R)`, weights `w_i = max(<G,O_i>,0)`, `U = w1·O1 + γ·w2·O2`, rescaled to `‖O1‖_F` | **my reconstruction.** The abstract does not give the exact weighting formula, so this follows its description (residual, own matrix-sign transform, duality weighting, rescale to Muon norm) |
| Transformer pretraining | replaced by a teacher–student tanh MLP regression (NumPy, manual backprop, CPU) |
| Biases | SGD-momentum (Muon is applied only to 2-D weights, as usual) |

## How to run

```bash
pip install numpy pytest
python demo.py      # ~45 s on CPU
python -m pytest -q
```

## Results (real output, `results.txt`)

```
5 seeds, 600 steps, batch 128, MLP [32, 128, 128, 8]; full-data MSE loss
target = Muon(Nesterov) median final loss = 0.02144

optimizer                    loss@200  loss@400   loss@600 (mean+-std)  seeds hitting target
SGD-momentum                  0.10078   0.07593      0.05942 +- 0.00114  0/5
Muon (Nesterov)               0.05859   0.03106      0.02249 +- 0.00194  3/5
AIMuon (Nesterov, g=0.25)     0.05168   0.03081      0.02414 +- 0.00145  2/5
Muon (plain momentum)         0.14837   0.09825      0.07573 +- 0.00721  0/5
AIMuon (plain, g=0.5)         0.06670   0.03991      0.02838 +- 0.00246  0/5
```

Tests: `4 passed`. They check that NS approximates msign (cosine > 0.98 with the exact SVD polar factor), that the AIMuon update has Muon's norm and is a descent direction, MLP gradients against finite differences, and that AIMuon trains.

**What this shows:**
- With plain (heavy-ball) momentum, the residual correction is a large win: final loss 0.076 → 0.028, and 2.2× lower loss at step 200. This matches the paper's motivation that correcting momentum lag speeds Muon up.
- With Nesterov momentum, which already folds the current gradient back in, AIMuon is ~12% better early (step 200) but ends level with or slightly behind Muon at this scale. My weighting is a guess, so this toy does **not** reproduce the paper's 6.5–10% end-to-end step savings.
- A small γ sweep (3 seeds) gave final losses of 0.0224 / 0.0236 / 0.0262 for γ = 0.1 / 0.25 / 0.5, against Muon's 0.0233. A smaller correction is safer once Nesterov is on.
