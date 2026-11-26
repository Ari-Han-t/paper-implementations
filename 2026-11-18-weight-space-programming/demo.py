"""Run all experiments and print the numbers quoted in README.md."""
import time
from collections import Counter

import numpy as np

from wsp.data import make_world
from wsp.memory import DenseAttentionMemory, SparseTableMemory, Row


def accuracy(model, qs):
    return float(np.mean([model.predict(q.text, q.options) == q.answer for q in qs]))


def main(seed: int = 0, n_people: int = 120):
    w = make_world(n_people=n_people, seed=seed)
    print(f"world: {len(w.facts_text)} fact sentences, {len(w.train)} labeled examples, "
          f"{len(w.test)} test questions "
          f"({len(w.corrupted)} corrupted facts, {len(w.missing)} missing facts)")
    ceiling = np.mean([(q.subject, q.relation) not in w.corrupted | w.missing for q in w.test])
    print(f"ceiling (answerable & correct in text): {ceiling:.3f}\n")

    sparse = SparseTableMemory().compile(w.facts_text)
    sparse_nocue = SparseTableMemory().compile(w.facts_text)
    sparse.fit_cues(w.train)
    dense = DenseAttentionMemory(d=128).compile(w.facts_text)

    # ---- 1. accuracy & size ------------------------------------------------
    print("== 1. accuracy / size ==")
    rows = [("sparse table (+10-example cues)", sparse),
            ("sparse table (no cues)", sparse_nocue),
            ("dense attention, d=128", dense)]
    for name, m in rows:
        print(f"{name:34s} acc={accuracy(m, w.test):.3f}  params={m.n_params():>7d}")
    print(f"param ratio dense/sparse: {dense.n_params() / sparse.n_params():.1f}x\n")

    # ---- 2. every error traces to one row (or to no row) -------------------
    print("== 2. error attribution (sparse) ==")
    causes = Counter()
    for q in w.test:
        pred, rid = sparse.explain(q.text, q.options)
        if pred == q.answer:
            continue
        key = (q.subject, q.relation)
        if rid is None:
            causes["no row (fact missing from text)"] += 1
        elif key in w.corrupted and sparse.rows[rid].source.startswith(q.subject) \
                and _row_relation(sparse.rows[rid]) == q.relation:
            causes["used a corrupted row"] += 1
        else:
            causes["picked wrong row (cue error)"] += 1
    for k, v in causes.most_common():
        print(f"  {k:34s} {v}")
    print()

    # ---- 3. ablation locality ----------------------------------------------
    print("== 3. delete-one-row locality ==")
    base = [sparse.predict(q.text, q.options) for q in w.test]
    changed = []
    for rid in range(len(sparse.rows)):
        row = sparse.delete_row(rid)
        now = [sparse.predict(q.text, q.options) for q in w.test]
        changed.append(sum(a != b for a, b in zip(base, now)))
        sparse.restore_row(rid, row)
    changed = np.array(changed)
    print(f"  answers changed per deleted row: mean={changed.mean():.2f}  "
          f"max={changed.max()}  histogram={dict(sorted(Counter(changed.tolist()).items()))}\n")

    # ---- 4. locate-and-rewrite edits ---------------------------------------
    print("== 4. rewrite corrupted rows ==")
    before = accuracy(sparse, w.test)
    n_edit = 0
    for rid, row in enumerate(sparse.rows):
        key = (" ".join(row.subject), _row_relation(row))
        if key in w.corrupted:
            sparse.edit_row(rid, w.truth[key])
            n_edit += 1
    after = accuracy(sparse, w.test)
    print(f"  edited {n_edit} rows: acc {before:.3f} -> {after:.3f}\n")

    # ---- 5. query latency vs table size ------------------------------------
    print("== 5. query latency (single query, CPU) ==")
    rng = np.random.default_rng(1)
    for n_rows in [1_000, 10_000, 50_000]:
        sents = [f"P{i} Q{i} lives in {rng.choice(['Pune', 'Oslo', 'Lyon', 'Perth'])}."
                 for i in range(n_rows)]
        qs = [(f"Where does P{i} Q{i} live?", ["Pune", "Oslo", "Lyon", "Perth"])
              for i in rng.integers(0, n_rows, 200)]
        s = SparseTableMemory().compile(sents)
        d = DenseAttentionMemory(d=128).compile(sents)
        ts = _time(s, qs)
        td = _time(d, qs)
        print(f"  rows={n_rows:>6d}  sparse={ts*1e6:8.1f} us  dense={td*1e6:9.1f} us  "
              f"speedup={td/ts:6.1f}x")


def _row_relation(row: Row) -> str:
    p = " ".join(row.predicate)
    if p in ("lives in", "has home in"):
        return "city"
    if p in ("was born in", "is native of"):
        return "born"
    if p in ("works as", "is employed as"):
        return "job"
    return "pet"


def _time(model, qs, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for text, opts in qs:
            model.predict(text, opts)
        best = min(best, (time.perf_counter() - t) / len(qs))
    return best


if __name__ == "__main__":
    main()
