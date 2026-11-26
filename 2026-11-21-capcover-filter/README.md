# CapCover Filter: Semantic Approximate Membership Queries over Embeddings

- **Authors:** Dvir David Biton, Shay Hahiashvili, Nadav Keren, et al.
- **arXiv:** [2611.20132](https://arxiv.org/abs/2611.20132) (cs.DB, November 2026)

## Core idea

A Bloom filter answers "have I seen this exact key?". CapCover asks the semantic version of that question: "is there any indexed embedding with cosine similarity ≥ τ to this query?" It says "no" without touching the vector index whenever it can, and it never gives a false negative.

It summarises the indexed vectors as a set of **spherical caps** (a centre direction plus an angular radius). If a query is farther from every cap than `radius + arccos(τ)`, the triangle inequality on the sphere means no indexed vector can reach similarity τ, so the expensive nearest-neighbour lookup is skipped. The summary does not depend on τ, so a single filter serves every threshold. The paper reports skipping 20–100% of index accesses at τ = 0.95 using about 2% of the FP32 index memory, and up to 382% higher throughput in front of a FAISS flat index.

## What is implemented here (and what is simplified)

The abstract does not spell out how the caps are built or quantised, so this is a from-scratch reconstruction of the method it describes:

| Piece | This implementation |
|---|---|
| Cap construction | Spherical k-means, then recursive 2-means splitting of any cap wider than `max_theta` |
| Cap radius | Largest angle from the centre to any member, so every member is covered |
| Compression | Centres stored as int8 with a per-cap scale. Radii are measured against the **dequantised** centre, so quantisation cannot cause false negatives |
| Query test | Reject if `angle(q, c_k) > θ_k + arccos(τ)` for every cap k |
| Bonus | Surviving caps are the only places a match can be, which gives an exact *pruned* existence search over contiguous per-cap slices |
| Not done | The paper's exact cap-cover optimisation and its quantisation scheme, real embedding datasets, and a FAISS integration |

The guarantee: if `cos(q, x) ≥ τ`, then `angle(q, x) ≤ φ = arccos τ`. Since x lies inside its cap, `angle(q, c) ≤ angle(q, x) + angle(x, c) ≤ φ + θ`. So any query the filter rejects has no match.

## How to run

```bash
pip install numpy pytest
python demo.py        # benchmark (≈1 min on CPU)
python -m pytest -v   # tests
```

## Results (real output, CPU, NumPy only)

The data is 50,000 synthetic 128-d embeddings in 200 topics. Each topic has low intrinsic dimension plus small noise. There are 1,000 queries, one at a time: 50% are paraphrase-like (near an indexed vector) and 50% are off-topic (random directions). In the table, "scanned" is the share of all (query, vector) dot products actually computed.

```
== max cap radius: unbounded deg | build 1.0s | caps=256 | filter=34 KiB = 0.14% of FP32 index
  tau  true+  rejected  FN  FP rate  scanned  brute us/q  capcover us/q  speedup
 0.80  50.0%      0.0%   0   100.0%   73.04%         971           1535    0.63x
 0.90  50.0%      0.0%   0   100.0%   48.07%        1023            989    1.03x
 0.95  50.0%      0.0%   0   100.0%   15.95%         984            366    2.69x
 0.98  50.0%      0.1%   0    99.8%    2.85%        1002            120    8.37x

== max cap radius: 35 deg | build 3.3s | caps=9015 | filter=1197 KiB = 4.79% of FP32 index
  tau  true+  rejected  FN  FP rate  scanned  brute us/q  capcover us/q  speedup
 0.80  50.0%      0.0%   0   100.0%    0.80%         958            708    1.35x
 0.90  50.0%      0.2%   0    99.6%    0.12%         989            508    1.94x
 0.95  50.0%     24.3%   0    51.4%    0.05%         954            452    2.11x
 0.98  50.0%     47.4%   0     5.2%    0.02%        1003            377    2.66x
```

(The full output, including a 25° run that is nearly identical to the 35° run, is in `results.txt`.)

Tests (`test_output.txt`):

```
test_capcover.py::test_no_false_negatives_across_tau PASSED
test_capcover.py::test_indexed_vectors_always_found PASSED
test_capcover.py::test_exact_search_matches_bruteforce PASSED
test_capcover.py::test_rejects_far_queries_and_is_small PASSED
4 passed
```

**Takeaways**

- There were **zero false negatives** in every configuration. The demo asserts this, and also checks that the pruned exact answers equal brute force.
- With tight caps (≤ 35°) and τ = 0.95, about **24% of all queries (≈49% of off-topic ones) are rejected outright** using a filter about 5% the size of the index. At τ = 0.98 nearly every off-topic query is rejected, and the false-positive rate is 5%. This matches the paper's claim that the filter works best at high thresholds.
- Coarse caps (256, 0.14% of memory) almost never reject a query outright, but they still cut the vectors scanned to 3–16% at high τ, for up to an **8.4× per-query speedup**.
- Rejection is weak at τ ≤ 0.9, because `arccos(0.9) ≈ 26°` is already as wide as a cap.
- There is a radius/memory trade-off. Tighter caps reject more queries but need more of them, and in this pure-Python loop the per-query cap check becomes the bottleneck. A real system would vectorise or tree-index the caps.
