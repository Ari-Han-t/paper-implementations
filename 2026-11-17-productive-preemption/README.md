# Making Preemption Productive in Long-Output LLM Serving

- **Paper:** *Making Preemption Productive in Long-Output LLM Serving*
- **arXiv:** [2611.20106](https://arxiv.org/abs/2611.20106) (see the arXiv page for the author list)
- **Area:** systems / LLM inference serving (cs.DC)

> Note: the arXiv page could not be fetched when this was built, so this is a minimal
> reproduction of the paper's central idea as described in its abstract-level summary
> ("KV thrashing" in long-output serving, fixed by proactive preemption). The specific
> policies here are my own simplified versions, not the paper's exact algorithm.

## Core idea

LLM servers batch many requests and keep each one's KV cache in GPU memory. In
**long-output** workloads (reasoning traces, agents, code) every running request's KV
cache grows by one token per step. A **reactive** scheduler (vLLM-style default)
admits requests whenever the prompt fits *right now*, and only preempts when memory is
already full. It evicts the newest request, drops its KV cache, and recomputes it on
readmission. The evicted request is readmitted as soon as some memory frees, grows, and
gets evicted again. The paper calls this cycle **KV thrashing**: memory and compute go
into rebuilding KV caches that are thrown away again.

The fix is to make preemption **proactive and productive**: use a prediction of each
request's output length to decide admission and preemption *before* overflow happens,
so every preemption is rare, early (little sunk work lost) and actually resolves the
memory pressure.

## What's implemented

`sim.py` is a discrete-time simulator of a continuous-batching server with a bounded
KV cache (in tokens), Poisson arrivals and lognormal (heavy-tailed) output lengths.
Step time = `base + per_seq * batch + prefill_per_tok * prefilled_tokens`.

| Policy | Admission | Preemption |
|---|---|---|
| `reactive` | FCFS if the prompt fits now | on overflow, evict newest (LIFO), recompute later |
| `watermark` | FCFS only if free memory stays above 10% | same as reactive |
| `proactive` | simulate the batch's future KV trajectory with *predicted* lengths (each request grows 1 tok/step until predicted finish, then frees); admit only if the projected **peak** fits | early, if overflow is within 8 steps *and* the projection confirms it; victim = least generated tokens (cheapest to redo). Requests that outlive their prediction double it. |

**Simplified / omitted:** no real model or GPU; KV cache counted in tokens, not pages;
recompute-only (no swap-to-CPU); the length predictor is the true length times
lognormal noise (stands in for a learned predictor); no fairness/SLO-aware scheduling.

## How to run

```bash
pip install numpy pytest
python demo.py          # prints the tables below
python -m pytest -q     # 10 tests
```

## Results (real output of `python demo.py`)

Time is in simulator "decode-step units"; throughput counts useful decoded tokens only.

### KV capacity 40,000 tokens, 200 long-output requests, predictor noise 0.25

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive | 46959.6 | 11.3 | 679 | 516183 | 20977.6 | 45385.3 |
| watermark | 48179.8 | 11.02 | 236 | 199652 | 21396.6 | 45957.8 |
| proactive | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |

### KV capacity 60,000 tokens, 200 long-output requests, predictor noise 0.25

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive | 36692.8 | 14.46 | 631 | 486356 | 15863.0 | 34734.9 |
| watermark | 37060.1 | 14.32 | 178 | 139891 | 16037.1 | 35293.3 |
| proactive | 38168.9 | 13.9 | 45 | 36392 | 16683.0 | 36626.5 |

### Sensitivity of proactive policy to length-prediction error (capacity 40,000)

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| proactive, noise=0.0 | 50476.4 | 10.51 | 20 | 20723 | 22736.0 | 49344.8 |
| proactive, noise=0.25 | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |
| proactive, noise=0.6 | 66326.3 | 8.0 | 42 | 45344 | 30883.4 | 63721.2 |
| proactive, noise=1.0 | 69834.4 | 7.6 | 22 | 23444 | 31818.5 | 66337.1 |

### Sensitivity to recompute cost (capacity 40,000)

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive, recompute cost=0.0015/tok | 46959.6 | 11.3 | 679 | 516183 | 20977.6 | 45385.3 |
| watermark, recompute cost=0.0015/tok | 48179.8 | 11.02 | 236 | 199652 | 21396.6 | 45957.8 |
| proactive, recompute cost=0.0015/tok | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |
| reactive, recompute cost=0.005/tok | 49216.1 | 10.78 | 690 | 534541 | 22096.4 | 47557.4 |
| watermark, recompute cost=0.005/tok | 48949.7 | 10.84 | 227 | 187673 | 21871.8 | 46569.9 |
| proactive, recompute cost=0.005/tok | 51215.7 | 10.36 | 72 | 71659 | 23188.9 | 49656.7 |
| reactive, recompute cost=0.01/tok | 51959.4 | 10.21 | 671 | 525252 | 23662.0 | 50279.0 |
| watermark, recompute cost=0.01/tok | 49966.2 | 10.62 | 228 | 185729 | 22508.5 | 48099.0 |
| proactive, recompute cost=0.01/tok | 52053.6 | 10.2 | 54 | 57469 | 23234.5 | 49513.8 |
| reactive, recompute cost=0.02/tok | 57818.6 | 9.18 | 677 | 527932 | 26824.6 | 56127.3 |
| watermark, recompute cost=0.02/tok | 52518.8 | 10.11 | 226 | 181840 | 23800.7 | 50441.5 |
| proactive, recompute cost=0.02/tok | 53609.4 | 9.9 | 68 | 69873 | 24357.3 | 52121.3 |

### Takeaways

- **Thrashing is real in this model:** under memory pressure the reactive policy
  preempts ~680 times for 200 requests and recomputes ~516k tokens of KV cache, with
  some requests evicted many times.
- **Proactive admission cuts wasted recompute by ~89–92%** (516k → 56k tokens) and
  preemptions by ~91–93% with a moderately noisy (σ=0.25) length predictor.
- **It's a trade-off, not a free win:** when recompute is cheap (0.0015 steps/token),
  packing memory aggressively and paying for thrashing still gives ~10% higher
  throughput than the conservative proactive policy. As recompute gets more expensive
  (0.01–0.02 steps/token, e.g. longer contexts or slower prefill) the gap closes and then
  reverses against the reactive policy (reactive 9.18 vs proactive 9.9 tok/t at 0.02).
  The simple watermark heuristic is a strong middle ground.
- **Prediction quality matters:** with oracle lengths the proactive policy needs only
  20 preemptions (vs 679 reactive) at ~7% throughput cost; at σ=1.0 noise it
  over-reserves and loses ~33% throughput.

## Files

- `sim.py`: simulator, workload generator, three scheduling policies
- `demo.py`: experiments (capacity, predictor-noise and recompute-cost sweeps)
- `demo_output.md`: saved demo output
- `test_sim.py`: tests (completion, memory bound, thrashing detected, proactive reduces recompute, determinism, oversized request)
