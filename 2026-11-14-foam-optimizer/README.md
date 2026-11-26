# FOAM: Factored One-sided Adam-Moment for Practical and Scalable SOAP

**Authors:** Jaemyung Yu, Byeongho Heo, Sangdoo Yun, Dongyoon Han (NAVER AI Lab) — NeurIPS 2026
**arXiv:** [2611.16905](https://arxiv.org/abs/2611.16905) (submitted 12 Nov 2026)

## Core idea

SOAP runs Adam in the eigenbasis of Shampoo's two Kronecker factors: for a weight gradient `G` (out × in) it tracks `L = EMA(G Gᵀ)` and `R = EMA(Gᵀ G)`, rotates `G → Q_Lᵀ G Q_R`, and keeps a dense Adam second moment there. It trains in fewer steps than AdamW, but it costs a lot of memory and two eigendecompositions per layer.

FOAM makes two changes:

1. **One-sided (input-only) preconditioning.** Drop `L`/`Q_L` and rotate only on the input side: `G → G Q_R`. This is exact when the output-side curvature is diagonal, and the paper finds it is close to diagonal in practice.
2. **Factored second moment.** Instead of a dense `(m, n)` second moment, keep a row vector `r` and a column vector `c` (Adafactor-style) and rebuild `V ≈ r cᵀ / mean(r)`. Under the K-FAC Fisher assumption with diagonal output curvature, the rotated second moment is exactly rank 1, so nothing is lost.

The paper reports that FOAM matches SOAP and Muon on validation loss for 124M–760M LLM pretraining, with half of SOAP's optimizer state and Muon-class wall-clock time.

## What's implemented vs. simplified

| Implemented | Simplified / omitted |
|---|---|
| SOAP (two-sided, dense V) baseline, in NumPy | No PyTorch (not installable in the sandbox), so this is a small MLP rather than a transformer |
| FOAM: input-side factor `R`, eigenbasis `Q_R` refreshed every 10 steps, first moment in the original basis then rotated, factored `r, c` second moment, bias correction | The exact update equations weren't readable (arXiv HTML was rate-limited and the official code isn't released yet), so this follows the abstract's description. Details such as how the moments are handled when the basis changes, or where epsilon goes, may differ from the paper |
| Ablation flag `factored=False` (one-sided, dense V) | No weight decay, grad clipping, or LR schedule |
| Adam/SGD-M baselines; plain Adam for bias vectors | Eigendecomposition with `eigh` instead of SOAP's power-iteration/QR refresh |

## Files

- `optimizers.py`: SGD-M, Adam, SOAP and FOAM
- `model.py`: a 2-layer tanh MLP with manual backprop, plus an ill-conditioned teacher-student regression task
- `demo.py`: learning-rate sweep × 3 seeds for each optimizer
- `test_foam.py`: unit tests

## How to run

```bash
pip install numpy pytest
python demo.py        # ~40 s on a laptop CPU
python -m pytest -q
```

## Results (real output)

MLP 64-128-16, 600 steps, batch 128. Inputs are correlated and ill-conditioned (condition number 100). Eval loss is the mean ± sd over 3 seeds at the best LR in each sweep.

```
Task: teacher-student regression, MLP 64-128-16, 600 steps, batch 128, seeds (0, 1, 2)

optimizer                         best lr    eval loss (mean±sd)  state floats  ms/step  per-lr mean loss
SGD-momentum                          0.3       0.4887 ± 0.0302         10384     0.38  0.1:1.1072  0.3:0.4887  1:inf
Adam                                  0.1       0.1419 ± 0.0147         20768     0.47  0.01:1.1121  0.03:0.3302  0.1:0.1419  0.3:8.1910
SOAP (two-sided, dense V)            0.03       0.0346 ± 0.0011         95008     1.39  0.01:0.1502  0.03:0.0346  0.1:0.8426
One-sided, dense V (ablation)        0.03       0.0651 ± 0.0035         61728     1.01  0.01:0.5422  0.03:0.0651  0.1:0.9874
FOAM (one-sided, factored V)         0.03       0.0596 ± 0.0039         51824     1.28  0.01:0.5386  0.03:0.0596  0.1:0.8849
```

Optimizer state (floats, excluding params): SOAP 95,008 → FOAM 51,824 (**−45%**). The paper reports about −50% at LLM scale, where `m·n` dominates.

Tests:

```
......                                                                   [100%]
6 passed in 0.16s
```

### Takeaways

- Both preconditioned optimizers beat Adam by a wide margin on this ill-conditioned task: Adam reached 0.142, FOAM 0.060 and SOAP 0.035.
- Factoring the second moment was free here. FOAM's factored V (0.0596) did as well as the one-sided dense-V ablation (0.0651), and the difference is within noise. This matches the paper's claim about rank 1 under K-FAC, and `test_kfac_diag_output_makes_rotated_second_moment_rank1` checks it numerically.
- Dropping the output-side factor was **not** free in this toy: SOAP stayed about 1.7× better. The paper's argument relies on near-diagonal output curvature, as found in transformer layers. In a small dense MLP where the hidden units are strongly coupled, that assumption is weaker. Checking this at transformer scale is the obvious next step.
- At these tiny sizes, NumPy's `eigh` and Python overhead dominate wall-clock time, so the ms/step figures shouldn't be read as the paper's speed claims.
