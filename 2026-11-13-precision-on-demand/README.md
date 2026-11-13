# Precision on Demand: Escalating Weight Bits at Uncertain Decoding Steps

**Paper:** *Precision on Demand: Escalating Weight Bits at Uncertain Decoding Steps in a Small Local Model*
**Authors:** Mohammed Nayeem Khan, Logesh L, Aliyas Ahmad Misri
**arXiv:** [2611.14418](https://arxiv.org/abs/2611.14418) (submitted 10 Nov 2026, cs.AI)

## Core idea

Local LLM decoding is usually memory-bandwidth bound: each token means reading every weight once. The paper stores weights in **int8** and, at every decoding step, first runs a cheap pass that reads only the **top 4 bits** of each weight. If that low-precision output distribution looks **uncertain** (high entropy, or a small gap between the top-2 tokens), the step is **escalated** and re-run with the full 8 bits. Most steps are easy, so you pay for high precision only where low precision would probably change the chosen token.

The paper (Qwen3-0.6B, GSM8K, simulated quantization) reports that low-pass entropy predicts argmax "flips" with AUC 0.93 offline, and that at a 15% escalation budget the entropy and margin gates catch 78% and 94% of flips, against about 15% for a random gate. The live accuracy results were described by the authors as inconclusive.

## What I implemented

| Paper | Here |
|---|---|
| Qwen3-0.6B | A 2-layer char-level MLP LM (~110K weights) in pure NumPy, trained in about 20 s on Python stdlib docstrings (no downloads) |
| Simulated int8 storage + top-4-bit low pass | Same: per-channel symmetric int8; the low pass uses `q >> 4` with midpoint reconstruction, so both passes share one int8 tensor |
| Entropy gate, margin gate | Both, with thresholds calibrated on a held-out split to hit a target escalation budget |
| Random-gate control (computed, not run, in the paper) | Actually run |
| Offline replay: flip AUC, flips caught at a budget | Yes |
| Live evaluation on GSM8K | Teacher-forced next-char accuracy + free-running greedy generation measured against the all-high continuation |
| Cost = weight bytes read per token | Two models: `nibble` (escalation reads only the extra 4 bits: 0.5 + 0.5r) and `rerun` (re-reads all 8 bits: 0.5 + r) |
| Verified speculation baseline | Not implemented |

The main simplification is the tiny model and character-level task. The aim is to reproduce the mechanism and the shape of the results, not the paper's numbers.

## Files

- `data.py`: builds an offline corpus from stdlib docstrings, plus a char vocab
- `model.py`: MLP LM with manual backprop and Adam
- `quant.py`: int8 quantization, high/low dequant views, bytes-per-token cost model
- `gates.py`: entropy/margin scores, budget thresholds, AUC, gated decoding, greedy generation
- `demo.py`: the full experiment
- `tests/test_pod.py`: 11 unit tests (gradient check, nibble semantics, AUC vs brute force, budget calibration, policy consistency)

## How to run

```bash
pip install numpy pytest
python -m pytest -q tests      # 11 passed in 0.17s
python demo.py                 # ~20-30 s on a laptop CPU
```

## Results (real output of `python demo.py`)

```
corpus 539,185 chars | vocab 96 | train 431,340 | cal 53,910 | test 53,911
next-char acc  fp32 0.5451 | int8 (high) 0.5457 | top-4-bit (low) 0.5227
argmax flips (low != high) on test: 0.1996 of 53,911 steps

[offline replay] flip-prediction AUC
  entropy  AUC = 0.800
  margin   AUC = 0.866

[gated policies on test, thresholds calibrated on held-out cal split]
  policy      budget  esc.rate  flips caught  acc     mean KL(hi||pol)  bytes(nibble)  bytes(rerun)
  all-low      0.00     0.000         0.000   0.5227          0.10844         0.500        0.500
  entropy      0.05     0.055         0.140   0.5244          0.09931         0.528        0.555
  margin       0.05     0.053         0.165   0.5268          0.09973         0.527        0.553
  random       0.05     0.050         0.050   0.5239          0.10290         0.525        0.550
  entropy      0.10     0.111         0.270   0.5256          0.09010         0.556        0.611
  margin       0.10     0.105         0.316   0.5288          0.09142         0.553        0.605
  random       0.10     0.102         0.098   0.5253          0.09757         0.551        0.602
  entropy      0.15     0.163         0.377   0.5276          0.08187         0.582        0.663
  margin       0.15     0.157         0.449   0.5320          0.08373         0.578        0.657
  random       0.15     0.149         0.152   0.5265          0.09210         0.575        0.649
  entropy      0.25     0.263         0.568   0.5312          0.06578         0.632        0.763
  margin       0.25     0.261         0.667   0.5371          0.06796         0.631        0.761
  random       0.25     0.250         0.254   0.5291          0.08121         0.625        0.750
  all-high     1.00     1.000         1.000   0.5457          0.00000         1.000        1.500

[free-running greedy generation: 60 prompts x 80 tokens, budget 0.15]
  policy     esc.rate  token-agree w/ high  mean prefix-match  exact-match seqs
  all-low      0.000                0.193                9.4        4/60
  entropy      0.119                0.254               14.1        4/60
  margin       0.142                0.287               18.6        4/60
  random       0.148                0.220               10.8        4/60

sample prompt: 'iles.\n\nR'
  high   : 'eturn a list of all the string containing a new string containing a new string c'
  low    : 'eturn a list of all the specified with the corresponding to the optional keyword'
  entropy: 'eturn a list of all the specified to the server object.\n\nThe function to the ser'
```

### Takeaways

- **The paper's main offline claim holds at small scale.** Low-pass uncertainty predicts flips well (AUC 0.80 entropy, 0.87 margin). At a ~15% budget, the margin gate catches **45%** of flips and the entropy gate **38%**, against **15%** for random: about 3x better than chance.
- **The margin gate beats the entropy gate**, which matches the paper's ordering (94% vs 78%).
- **The accuracy gains are real but small**, which is in line with the paper calling its live results inconclusive. At a 15% budget the margin gate recovers about 41% of the low-to-high accuracy gap (0.5227 → 0.5320 vs 0.5457) while reading 0.58x the all-high weight bytes. Random recovers only about 17%.
- **Free-running decoding is fragile.** One flip changes the context, and the generations then diverge. Gating roughly doubles the matching prefix length (9.4 → 18.6 tokens) but rarely keeps whole sequences identical to the high pass.
- Flip rates are higher here (20%) than one would expect for a large model, because a 110K-weight model has much less redundancy to absorb 4-bit noise.
