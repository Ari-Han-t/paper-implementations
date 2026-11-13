"""End-to-end experiment for Precision on Demand (arXiv:2611.14418), CPU only.

1. Train a small char-level MLP LM on Python stdlib docstrings.
2. Quantize to int8; derive a 4-bit "top nibble" low pass from the same storage.
3. Offline replay on held-out text: do low-pass entropy / top-2 margin predict
   steps where the low pass's argmax differs from the high pass ("flips")?
4. Gated policies at several escalation budgets: accuracy vs. ground truth,
   KL to the high pass, and weight bytes read per token.
5. Free-running greedy generation: agreement with the all-high continuation.

Run:  python demo.py            (about 2-4 minutes on a laptop CPU)
"""
import time

import numpy as np

from data import CharVocab, build_corpus
from gates import (auc, generate, gated_logp, kl, replay, threshold_for_budget,
                   uncertainty_scores)
from model import init_params, make_windows, train
from quant import bytes_per_token, make_views

CTX = 8
SEED = 0


def main():
    t0 = time.time()
    text = build_corpus()
    vocab = CharVocab(text)
    ids = vocab.encode(text)
    n = len(ids)
    tr, cal, te = ids[: int(0.8 * n)], ids[int(0.8 * n): int(0.9 * n)], ids[int(0.9 * n):]
    Xtr, Ytr = make_windows(tr, CTX)
    Xcal, _ = make_windows(cal, CTX)
    Xte, Yte = make_windows(te, CTX)
    print(f"corpus {n:,} chars | vocab {len(vocab)} | train {len(Xtr):,} | cal {len(Xcal):,} | test {len(Xte):,}")

    params = init_params(len(vocab), ctx=CTX, seed=SEED)
    print("training fp32 model ...")
    train(params, Xtr, Ytr, steps=6000, seed=SEED)

    hi, lo = make_views(params)
    lhi, llo = replay(hi, lo, Xte)
    lhi_c, llo_c = replay(hi, lo, Xcal)

    acc_fp = (replay(params, params, Xte)[0].argmax(-1) == Yte).mean()
    acc_hi = (lhi.argmax(-1) == Yte).mean()
    acc_lo = (llo.argmax(-1) == Yte).mean()
    flips = lhi.argmax(-1) != llo.argmax(-1)
    print(f"\nnext-char acc  fp32 {acc_fp:.4f} | int8 (high) {acc_hi:.4f} | top-4-bit (low) {acc_lo:.4f}")
    print(f"argmax flips (low != high) on test: {flips.mean():.4f} of {len(flips):,} steps")

    # ---- 1. offline replay: can the low pass predict its own flips? ----
    sc = uncertainty_scores(llo)
    sc_cal = uncertainty_scores(llo_c)
    print("\n[offline replay] flip-prediction AUC")
    for g in ("entropy", "margin"):
        print(f"  {g:8s} AUC = {auc(sc[g], flips):.3f}")

    rng = np.random.default_rng(SEED)
    budgets = [0.05, 0.10, 0.15, 0.25]
    print("\n[gated policies on test, thresholds calibrated on held-out cal split]")
    print("  policy      budget  esc.rate  flips caught  acc     mean KL(hi||pol)  bytes(nibble)  bytes(rerun)")
    rows = []

    def report(name, budget, esc):
        lp = gated_logp(lhi, llo, esc)
        caught = (esc & flips).sum() / max(flips.sum(), 1)
        acc = (lp.argmax(-1) == Yte).mean()
        k = kl(lhi, lp).mean()
        r = esc.mean()
        rows.append((name, budget, r, caught, acc, k))
        print(f"  {name:10s}  {budget:5.2f}   {r:7.3f}   {caught:11.3f}   {acc:.4f}  {k:15.5f}  "
              f"{bytes_per_token(r, 'nibble'):12.3f}  {bytes_per_token(r, 'rerun'):11.3f}")

    report("all-low", 0.0, np.zeros(len(Yte), bool))
    for b in budgets:
        for g in ("entropy", "margin"):
            thr = threshold_for_budget(sc_cal[g], b)
            report(g, b, sc[g] > thr)
        report("random", b, rng.random(len(Yte)) < b)
    report("all-high", 1.0, np.ones(len(Yte), bool))

    # ---- 2. free-running greedy generation ----
    print("\n[free-running greedy generation: 60 prompts x 80 tokens, budget 0.15]")
    n_prompts, n_tok, b = 60, 80, 0.15
    starts = np.random.default_rng(1).integers(0, len(te) - CTX - 1, n_prompts)
    thr = {g: threshold_for_budget(sc_cal[g], b) for g in ("entropy", "margin")}
    refs = [generate(hi, lo, te[s:s + CTX], n_tok, CTX, policy="high")[0] for s in starts]
    policies = [("all-low", dict(policy="low")),
                ("entropy", dict(policy="gate", gate="entropy", threshold=thr["entropy"])),
                ("margin", dict(policy="gate", gate="margin", threshold=thr["margin"])),
                ("random", dict(policy="random", rate=b))]
    print("  policy     esc.rate  token-agree w/ high  mean prefix-match  exact-match seqs")
    for name, kw in policies:
        agree, prefix, exact, esc_total = [], [], 0, 0
        prng = np.random.default_rng(7)
        for s, ref in zip(starts, refs):
            out, esc = generate(hi, lo, te[s:s + CTX], n_tok, CTX, rng=prng, **kw)
            esc_total += esc
            a = np.array(out) == np.array(ref)
            agree.append(a.mean())
            prefix.append(n_tok if a.all() else int(np.argmin(a)))
            exact += int(a.all())
        print(f"  {name:9s}  {esc_total / (n_prompts * n_tok):7.3f}   {np.mean(agree):18.3f}  "
              f"{np.mean(prefix):17.1f}  {exact:7d}/{n_prompts}")

    s0 = starts[0]
    print("\nsample prompt:", repr(vocab.decode(te[s0:s0 + CTX])))
    print("  high   :", repr(vocab.decode(refs[0])))
    print("  low    :", repr(vocab.decode(generate(hi, lo, te[s0:s0 + CTX], n_tok, CTX, policy="low")[0])))
    print("  entropy:", repr(vocab.decode(generate(hi, lo, te[s0:s0 + CTX], n_tok, CTX, policy="gate",
                                                     gate="entropy", threshold=thr["entropy"])[0])))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
