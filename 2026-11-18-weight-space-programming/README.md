# Weight-Space Programming Doesn't Need Transformers

**Authors:** Hyunjong Ok, Jaeho Lee
**arXiv:** [2611.22055](https://arxiv.org/abs/2611.22055) (cs.LG, submitted 17 Nov 2026)

## Core idea

*Weight-space programming* means an LLM writes a model's weights directly instead of training them with gradient descent. Transformers became the default because they train well with gradients and run fast on GPUs. Neither advantage applies once nobody is running gradient descent. The authors gave three frontier LLMs ten labeled examples and a sandbox, mostly on multiple-choice tasks. When the LLMs were also free to design the architecture, all three compiled plain-text facts into a **sparse lookup table** that a fixed procedure queries. That table beat the best Transformer the same LLM wrote while using about 1,800× fewer parameters. It is also interpretable: each wrong answer traces to a single row, and deleting a row changes about one other answer. From this the authors propose an associative-memory model that runs up to 17× faster than a Transformer of the same size.

## What is implemented here (NumPy only, CPU)

| Paper | This repo |
|---|---|
| An LLM reads plain-text facts and writes the weights | A deterministic **compiler** (`SparseTableMemory.compile`). Each sentence becomes a row with key (subject tokens, predicate tokens) and value (object). No gradients are used. |
| Ten labeled examples | `fit_cues` uses the 10 examples to learn question-word → predicate cue weights. These are needed because "lives in" and "was born in" share the same answer pool. |
| Fixed query procedure | Inverted index on the subject, a full-name match, a filter to rows whose value is one of the options, and a ranking by cue score. If no row matches, the model abstains and returns option 0. |
| LLM-written Transformer baseline | `DenseAttentionMemory`: one softmax-attention layer with hand-written dense weights (random token embeddings, keys from summed embeddings, one-hot values) |
| Interpretability: error → row, delete-row locality, rewrite | `explain()` returns the row used for an answer. `delete_row`, `restore_row` and `edit_row` support the ablation and edit experiments. |
| Real benchmarks (MMLU-style MC, generative tasks) | **Simplified:** a synthetic fictional world (`wsp/data.py`) with 4 relations and 4-option questions. 8% of facts are deliberately corrupted and 8% are missing. |

What this does *not* reproduce: no LLM writes the weights (the compiler is rule-based and stands in for the LLM's output), there are no real benchmarks, and there is no GPU kernel. The dense baseline is a single attention layer, not a full Transformer, so the parameter ratio below is not comparable to the paper's 1,800×.

## How to run

```bash
pip install numpy pytest
python demo.py            # all experiments, ~10 s
python -m pytest -q tests
```

## Results (actual output of `python demo.py`)

```
world: 442 fact sentences, 10 labeled examples, 470 test questions (43 corrupted facts, 38 missing facts)
ceiling (answerable & correct in text): 0.828

== 1. accuracy / size ==
sparse table (+10-example cues)    acc=0.862  params=   2397
sparse table (no cues)             acc=0.834  params=   2385
dense attention, d=128             acc=0.743  params=  77248
param ratio dense/sparse: 32.2x

== 2. error attribution (sparse) ==
  no row (fact missing from text)    38
  picked wrong row (cue error)       15
  used a corrupted row               12

== 3. delete-one-row locality ==
  answers changed per deleted row: mean=0.69  max=2  histogram={0: 144, 1: 289, 2: 9}

== 4. rewrite corrupted rows ==
  edited 43 rows: acc 0.862 -> 0.919

== 5. query latency (single query, CPU) ==
  rows=  1000  sparse=     5.0 us  dense=     36.9 us  speedup=   7.3x
  rows= 10000  sparse=     4.8 us  dense=    497.6 us  speedup= 103.3x
  rows= 50000  sparse=     4.9 us  dense=   2297.1 us  speedup= 473.4x
```

Tests: `7 passed in 0.10s`.

### Reading the numbers

- **Accuracy and size:** the sparse table beats the hand-written dense attention layer (0.862 vs 0.743) with 32× fewer parameters. The 10 labeled examples add about 3 points by teaching which question words point at "born in" rather than "lives in". Accuracy can exceed the "ceiling" because, when a corrupted value is not among the 4 options, the procedure falls back to another row or abstains, and those fallbacks are sometimes right by luck.
- **Every error has an address:** each wrong answer maps to a missing fact, a specific corrupted row, or a wrong-row choice. This matches the paper's "each error traces to a single row".
- **Locality:** deleting a row changes 0.69 answers on average and never more than 2, in line with the paper's "about one other answer".
- **Editing:** rewriting the 43 corrupted rows raises accuracy from 0.862 to 0.919 with no retraining. The errors left come from missing facts and cue errors.
- **Speed:** sparse lookup through an inverted index stays at about 5 µs per query regardless of table size, while the dense layer grows linearly with the number of rows. The CPU speedups here are larger than the paper's 17× GPU figure because a GPU parallelises the dense matmul.

## Files

- `wsp/data.py`: synthetic world, corrupted and missing facts, 10 clean labeled examples
- `wsp/memory.py`: `SparseTableMemory` and `DenseAttentionMemory`
- `demo.py`: experiments 1–5
- `tests/test_memory.py`: unit tests
