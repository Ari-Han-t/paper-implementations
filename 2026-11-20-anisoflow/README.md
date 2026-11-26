# Learning Anisotropic Prediction Targets for Flow Models

**Authors:** Rayhan Zirvi, Hansheng Chen, Gordon Wetzstein, Leonidas Guibas
**arXiv:** [2611.17767](https://arxiv.org/abs/2611.17767)

## Core idea

Flow-matching models are usually trained to predict one fixed target: the velocity (`v = x - e`), the noise, or the clean data `x`. This paper makes the target *learnable* instead. It scales the noise part of the target with a separate weight per direction, in a learned basis. A closed-form map turns any such prediction back into a velocity, so the network, loss and sampler stay the same. The authors report that the preferred noise weight grows roughly linearly with the data variance in each direction. On ImageNet 256x256 they report FID going from 1.90 to 1.55 over the JiT baseline.

## What this implements

The abstract does not give the paper's equations, so this uses one natural parameterisation that matches the description:

```
x_t = t x + (1 - t) e                 (data x, noise e ~ N(0, I))
W   = Q diag(w) Q^T,  w >= 0           (per-direction noise weights)
y   = x - W e                          (network target; w=0 -> x-pred, w=1 -> v-pred)
e   = (tW + (1-t)I)^{-1} (x_t - t y)   (closed-form inversion)
v   = y + (W - I) e                    (velocity used by the ODE sampler)
```

- `anisoflow.py`: the target, the closed-form target-to-velocity map, a NumPy SiLU MLP with manual backprop and Adam, training, an Euler sampler, and a sliced-Wasserstein metric.
- `demo.py`: 2D toy data (a 4-mode mixture along a wide major axis and a thin Gaussian along the minor axis, rotated 30°). It sweeps `(w_major, w_minor)` with 2 seeds each, then runs a variance-scaling check.
- `test_anisoflow.py`: checks exact velocity recovery, the x-pred and v-pred special cases, MLP gradients against finite differences, and that training plus sampling work end to end.

**Simplified:**

- The basis `Q` is the data's PCA basis, not learned jointly.
- The weights `w` are picked by grid search on held-out velocity-space MSE, not learned by gradient descent.
- The model is a 2D toy MLP instead of an ImageNet transformer.

## How to run

```bash
pip install numpy pytest
python -m pytest -q        # 4 tests, under 1 s
python demo.py             # about 4.5 min on CPU
```

## Results (actual output, `demo_output.txt`)

Data PCA variances: major = 4.017, minor = 0.0101. Scores are the mean of 2 seeds; lower is better.

| Target | w (major, minor) | Val velocity MSE | Sliced-Wasserstein (samples vs data) |
|---|---|---|---|
| x-prediction | (0, 0) | 1.6706 | 0.1760 |
| v-prediction | (1, 1) | 1.5264 | 0.1287 |
| **Anisotropic (best)** | **(1.5, 0.5)** | **1.5052** | **0.0992** |

- The anisotropic target improved sample quality (sliced-Wasserstein) by about 23% over v-prediction and 44% over x-prediction.
- The best setting gives the high-variance direction a higher noise weight (1.5) than the low-variance one (0.5). That matches the paper's qualitative claim that the preferred weight grows with variance. Setting `w_minor = 0` (pure x-pred on the thin axis) was clearly harmful.
- Variance-scaling check (isotropic sweep per data scale, 1 seed): the best `w` went 0.5, 0.5, 2.0, 2.0 for std = 0.1, 0.5, 1.0, 2.0. The trend is in the predicted direction, but the MSE differences among `w >= 0.25` are tiny at this scale, so treat it as weak evidence. The one robust finding is that `w = 0` is worst at every scale.

```
std= 0.1  best w=0.50  0.0:0.150 0.25:0.138 0.5:0.138 1.0:0.138 1.5:0.139 2.0:0.139
std= 0.5  best w=0.50  0.0:0.848 0.25:0.763 0.5:0.762 1.0:0.762 1.5:0.763 2.0:0.764
std= 1.0  best w=2.00  0.0:1.674 0.25:1.542 0.5:1.540 1.0:1.539 1.5:1.539 2.0:1.539
std= 2.0  best w=2.00  0.0:3.178 0.25:3.108 0.5:3.105 1.0:3.102 1.5:3.101 2.0:3.099
```

Tests: `4 passed in 0.71s`.
