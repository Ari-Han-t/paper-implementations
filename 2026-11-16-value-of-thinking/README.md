# The Value of Thinking: Learning When to Reason in Hybrid Decision Models

**Authors:** Zhengxu Yu, Ke Xu, Meng Fang, Haitham Ammar, Weilin Luo, Jun Wang
**arXiv:** [2611.16961](https://arxiv.org/abs/2611.16961) (submitted 12 Nov 2026)

## Core idea

Adaptive-reasoning methods usually shorten chain-of-thought or switch modes, but the answer still comes out one token at a time. This paper proposes **zero-or-T thinking**: a single model either

- answers from the **implicit computation of one forward pass**, emitting no reasoning tokens (cost 0), or
- runs **full autoregressive reasoning** (cost T tokens) when that is worth it.

A lightweight **self-routing block** sits on a frozen backbone and estimates the **value of thinking**, the expected extra utility of slow reasoning over the fast answer, from what the implicit pass has already produced. Slow mode runs only when `v_hat(x) > λ`, where λ is a per-request price. When there are no per-decision labels (for example in episodic agent tasks), the router is trained with **CDPO (Calibrated Decision Policy Optimisation)**, a reinforcement-learning recipe that uses only episode-level returns.

The paper reports, on JevBench, HDM-4B at 0.857 vs. 0.851 for the baseline with 72.5% lower median latency, and on MiniWoB, HDM-27B with 80.1% lower token cost.

## What's implemented and what's simplified

All of this is NumPy and runs on a CPU in about 3 seconds.

| Paper | This repo |
|---|---|
| LLM backbone, frozen | Frozen random-feature network `h = tanh(Wx+b)` |
| Fast mode (single forward pass, 0 tokens) | Linear softmax head on `h` |
| Slow mode (T-token reasoning) | A better-informed decoder that sees 8 extra noisy re-observations of the input and costs T=64 "tokens". It stands in for the information that explicit reasoning uncovers. |
| Tasks such as BANKING77 | Synthetic 8-intent classification. Each request has a hidden difficulty (noise scale) that the router never sees. |
| Self-routing block that estimates value of thinking | 2-layer MLP over `[h, top-3 probs, margin, entropy]` that predicts `v = 1[slow correct] − 1[fast correct]` |
| Decision rule `v_hat > λ` with a per-request price λ | Same |
| Supervised router (per-decision labels) | MSE regression on `v` |
| CDPO (episode-level returns) | **Our simplified reading of it.** Episodes have 16 requests and one price λ sampled per episode. The policy is `π(slow) = σ((v_hat − λ)/τ)`. We use REINFORCE with a group baseline (8 rollouts per episode) on the episode return `R = Σ correct − λ·#slow`. A squared-hinge **calibration term** keeps `v_hat` inside the price range, because the policy gradient vanishes outside it and `v_hat` drifts without bound otherwise. The paper's exact CDPO objective was not available to us. |

Not reproduced: real LLMs, the paper's benchmarks, latency measurements, and the agentic MiniWoB setting.

## How to run

```bash
pip install numpy pytest
python demo.py        # experiment
python -m pytest -q   # tests
```

## Results (real output of `python demo.py`, seed 0, 6k held-out requests)

```
fast-mode acc 0.875 (0 tokens) | slow-mode acc 0.998 (64 tokens)
fraction of requests where thinking helps: 0.124, hurts: 0.001

Threshold-free comparison: mean accuracy over all slow-call budgets
  random                                   0.9382
  confidence (1 - max p)                   0.9792
  value router (supervised)                0.9775
  value router (CDPO, episode returns)     0.9745

Price-conditioned routing (slow iff v_hat > lambda), utility = acc - lambda * slow_frac
  lambda | policy                acc  slow%  tokens  utility
    0.02 | always-slow         0.998 100.0%    64.0   0.9780
    0.02 | supervised HDM      0.993  53.3%    34.1   0.9818
    0.02 | CDPO HDM            0.988  44.2%    28.3   0.9788
    0.10 | always-slow         0.998 100.0%    64.0   0.8980
    0.10 | supervised HDM      0.987  35.8%    22.9   0.9516
    0.10 | CDPO HDM            0.980  33.4%    21.3   0.9470
    0.40 | always-fast         0.875   0.0%     0.0   0.8747
    0.40 | supervised HDM      0.947  14.4%     9.2   0.8896
    0.40 | CDPO HDM            0.931  11.9%     7.6   0.8835
  (full table printed by demo.py)

Calibration (test set, binned by predicted v_hat):
  supervised pred -0.094 / real +0.015 | pred -0.006 / real +0.009 | pred +0.101 / real +0.072 | pred +0.483 / real +0.398
  CDPO       pred -0.177 / real +0.017 | pred -0.062 / real +0.016 | pred +0.072 / real +0.086 | pred +0.424 / real +0.375

Headline: supervised HDM reaches acc 0.988 (always-slow 0.998) using 23.9 tokens/request vs 64 -> 62.6% fewer tokens (lambda=0.090)
```

The CDPO router, trained only on episode returns, reaches 0.988 accuracy at 29.4 tokens per request (54% fewer tokens than always-slow).

Tests: `7 passed in 1.42s`. They cover a gradient check on the router's backprop, the oracle beating random routing, the supervised router beating random and saving tokens, and the CDPO router getting fewer slow calls as λ rises.

**Takeaways:**

- Routing on the value of thinking recovers almost all of the slow-mode accuracy with about 40–60% fewer reasoning tokens. Because `v_hat` is in utility units, one router serves every price λ at test time without retraining.
- CDPO learns from episode returns alone and comes close to the supervised router. The calibration term is essential: without it, `v_hat` for low-value requests drifted to about −9 in our runs.
- **Honest negative:** in this toy world, a plain confidence threshold (`1 − max p`) ranks requests as well as the learned router (0.979 vs. 0.978). Here the fast head's confidence already captures difficulty. The learned router's advantage is that its outputs are calibrated values that can be compared directly with a price, while a confidence threshold has to be tuned separately for every λ.

## Files

- `hdm.py`: world, frozen backbone, fast head, router (supervised and CDPO training), evaluation helpers
- `demo.py`: the experiment above
- `test_hdm.py`: unit tests
