"""Demo: zero-or-T thinking with a learned value-of-thinking router.

Compares, on held-out requests:
  always-fast, always-slow, random routing, confidence-threshold baseline,
  supervised value router, and CDPO-style router (episode returns only).
"""
import time

import numpy as np

from hdm import Router, area_under_tradeoff, build, evaluate


def main(seed=0):
    t0 = time.time()
    world, route, test = build(seed=seed)
    T = world.T
    fo, so = test["fast_ok"], test["slow_ok"]
    v_route = route["slow_ok"] - route["fast_ok"]

    print(f"fast-mode acc {fo.mean():.3f} (0 tokens) | slow-mode acc {so.mean():.3f} ({T} tokens)")
    print(f"fraction of requests where thinking helps: {(so > fo).mean():.3f}, hurts: {(so < fo).mean():.3f}\n")

    sup = Router(route["f"].shape[1], seed=seed + 2).fit_supervised(route["f"], v_route, seed=seed)
    cdpo = Router(route["f"].shape[1], seed=seed + 3).fit_cdpo(
        route["f"], route["fast_ok"], route["slow_ok"], episodes=60000, seed=seed)

    rng = np.random.default_rng(seed + 7)
    conf = 1 - test["p"].max(1)  # baseline: think when fast head is unsure
    scores = {
        "random": rng.random(len(fo)),
        "confidence (1 - max p)": conf,
        "value router (supervised)": sup.value(test["f"]),
        "value router (CDPO, episode returns)": cdpo.value(test["f"]),
    }

    print("Threshold-free comparison: mean accuracy over all slow-call budgets")
    for k, s in scores.items():
        print(f"  {k:<40s} {area_under_tradeoff(s, fo, so):.4f}")

    print("\nPrice-conditioned routing (slow iff v_hat > lambda), utility = acc - lambda * slow_frac")
    print(f"  {'lambda':>6} | {'policy':<18} {'acc':>6} {'slow%':>6} {'tokens':>7} {'utility':>8}")
    for lam in [0.02, 0.05, 0.1, 0.2, 0.4]:
        rows = {
            "always-fast": np.zeros(len(fo), bool),
            "always-slow": np.ones(len(fo), bool),
            "supervised HDM": sup.value(test["f"]) > lam,
            "CDPO HDM": cdpo.value(test["f"]) > lam,
        }
        for name, dec in rows.items():
            m = evaluate(dec, fo, so, T)
            u = m["acc"] - lam * m["slow_frac"]
            print(f"  {lam:>6.2f} | {name:<18} {m['acc']:>6.3f} {100*m['slow_frac']:>5.1f}% {m['tokens']:>7.1f} {u:>8.4f}")

    # Calibration check: does v_hat match realised value of thinking?
    print("\nCalibration (test set, binned by predicted v_hat):")
    v_true = so - fo
    for name, r in [("supervised", sup), ("CDPO", cdpo)]:
        vh = r.value(test["f"])
        qs = np.quantile(vh, [0, .25, .5, .75, 1.0])
        parts = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (vh >= lo) & (vh <= hi)
            parts.append(f"pred {vh[m].mean():+.3f} / real {v_true[m].mean():+.3f}")
        print(f"  {name:<10s} " + " | ".join(parts))

    # Headline: match always-slow accuracy within 1pt at minimum tokens
    target = so.mean() - 0.01
    vh = sup.value(test["f"])
    best = None
    for lam in np.linspace(-0.2, 0.6, 161):
        m = evaluate(vh > lam, fo, so, T)
        if m["acc"] >= target:
            best = (lam, m)
    lam, m = best
    print(f"\nHeadline: supervised HDM reaches acc {m['acc']:.3f} (always-slow {so.mean():.3f}) "
          f"using {m['tokens']:.1f} tokens/request vs {T} -> {100*(1-m['tokens']/T):.1f}% fewer tokens (lambda={lam:.3f})")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
