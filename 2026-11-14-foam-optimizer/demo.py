"""Compare SGD-M, Adam, SOAP, FOAM (and FOAM ablations) on a small MLP.

Each optimizer gets a small learning-rate sweep; best final eval loss is
reported along with optimizer-state size and wall-clock per step.
"""
import time
import numpy as np
np.seterr(all="ignore")
from model import init, forward_backward, make_task
from optimizers import SGDM, Adam, SOAP, FOAM

STEPS, BATCH, D_IN, D_H, D_OUT = 600, 128, 64, 128, 16
SEEDS = (0, 1, 2)


def run(make_opt, lr, seed):
    rng = np.random.default_rng(seed)
    X, Y = make_task(6000, D_IN, D_OUT, rng)
    Xtr, Ytr, Xte, Yte = X[:5000], Y[:5000], X[5000:], Y[5000:]
    p = init(D_IN, D_H, D_OUT, np.random.default_rng(100 + seed))
    opt = make_opt(p, lr)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        idx = rng.integers(0, len(Xtr), BATCH)
        loss, g = forward_backward(p, Xtr[idx], Ytr[idx])
        if not np.isfinite(loss):
            return np.inf, 0, 0
        opt.step(p, g)
    dt = (time.perf_counter() - t0) / STEPS * 1e3
    ev, _ = forward_backward(p, Xte, Yte)
    return ev, opt.state_floats(), dt


CONFIGS = {
    "SGD-momentum": (lambda p, lr: SGDM(p, lr), [0.1, 0.3, 1.0]),
    "Adam": (lambda p, lr: Adam(p, lr), [1e-2, 3e-2, 1e-1, 3e-1]),
    "SOAP (two-sided, dense V)": (lambda p, lr: SOAP(p, lr), [1e-2, 3e-2, 1e-1]),
    "One-sided, dense V (ablation)": (lambda p, lr: FOAM(p, lr, factored=False), [1e-2, 3e-2, 1e-1]),
    "FOAM (one-sided, factored V)": (lambda p, lr: FOAM(p, lr), [1e-2, 3e-2, 1e-1]),
}

if __name__ == "__main__":
    print(f"Task: teacher-student regression, MLP {D_IN}-{D_H}-{D_OUT}, "
          f"{STEPS} steps, batch {BATCH}, seeds {SEEDS}\n")
    base = None
    print(f"{'optimizer':32s} {'best lr':>8s} {'eval loss (mean±sd)':>22s} "
          f"{'state floats':>13s} {'ms/step':>8s}  per-lr mean loss")
    for name, (mk, lrs) in CONFIGS.items():
        per_lr = {}
        for lr in lrs:
            res = [run(mk, lr, s) for s in SEEDS]
            per_lr[lr] = res
        best = min(lrs, key=lambda lr: np.mean([r[0] for r in per_lr[lr]]))
        losses = [r[0] for r in per_lr[best]]
        floats, ms = per_lr[best][0][1], np.mean([r[2] for r in per_lr[best]])
        sweep = "  ".join(f"{lr:g}:{np.mean([r[0] for r in per_lr[lr]]):.4f}" for lr in lrs)
        print(f"{name:32s} {best:8g} {np.mean(losses):12.4f} ± {np.std(losses):.4f} "
              f"{floats:13d} {ms:8.2f}  {sweep}")
