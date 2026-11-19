"""Compare a standard Transformer with MBT variants at ~matched body FLOPs."""
import sys
import numpy as np
from mbt import Config, count_params, flops_per_token
from train import load_corpus, train

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SEEDS = [0, 1]
tr, va, V = load_corpus()
base = dict(vocab=V, d=64, n_heads=4, prelude=1, coda=1)
variants = {
    "Standard (6 seq blocks)":       Config(**base, K=1, branch_depth=4, d_b=64),
    "MBT Wide-Shallow K=2 (2x2@64)": Config(**base, K=2, branch_depth=2, d_b=64),
    "MBT Balanced K=2 (3x2@52)":     Config(**base, K=2, branch_depth=3, d_b=52),
    "MBT Narrow-Deep K=2 (4x2@44)":  Config(**base, K=2, branch_depth=4, d_b=44),
    "MBT Balanced K=2 + Joint-KV":   Config(**base, K=2, branch_depth=3, d_b=52, joint_kv=True),
    "MBT Wide-Shallow K=2, mean fusion": Config(**base, K=2, branch_depth=2, d_b=64, fusion="mean"),
}
std_kv = variants["Standard (6 seq blocks)"].kv_cache_per_token()
print(f"corpus: {len(tr)} train chars, vocab {V}, steps {STEPS}, seeds {SEEDS}\n")
rows = []
for name, cfg in variants.items():
    losses, secs = [], []
    for s in SEEDS:
        p, vl, dt = train(cfg, tr, va, steps=STEPS, seed=s)
        losses.append(vl); secs.append(dt)
    rows.append((name, count_params(p), flops_per_token(cfg), cfg.sequential_depth(),
                 cfg.kv_cache_per_token() / std_kv, np.mean(losses), np.std(losses), np.mean(secs)))
    print(f"done {name}: {np.mean(losses):.4f}", flush=True)

print("\n| Variant | Params | FLOPs/tok | Seq. depth | KV cache vs Std | Val loss (mean ± std, 2 seeds) | Train s |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]:,} | {r[2]/1e3:.0f}K | {r[3]} | {r[4]:.2f}x | {r[5]:.4f} ± {r[6]:.4f} | {r[7]:.0f} |")
