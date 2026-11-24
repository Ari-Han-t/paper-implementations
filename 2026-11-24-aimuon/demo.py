"""Compare SGD-momentum, Muon and AIMuon on a teacher-student MLP regression (CPU, NumPy)."""
import numpy as np
from aimuon import Muon, AIMuon, SGDM
from model import MLP, make_task

STEPS, BATCH, SEEDS = 600, 128, 5
SIZES = [32, 128, 128, 8]


def train(opt_factory, seed):
    X, Y = make_task(seed)
    net = MLP(SIZES, seed)
    opt, bias_opt = opt_factory(), SGDM(lr=0.05)
    rng = np.random.default_rng(seed + 7)
    curve = []
    for t in range(STEPS):
        idx = rng.integers(0, len(X), BATCH)
        _, g = net.loss_and_grads(X[idx], Y[idx])
        for k, G in g.items():
            (opt if k.startswith("W") else bias_opt).step(k, net.params[k], G)
        if t % 10 == 9:
            curve.append(net.loss_and_grads(X, Y)[0])
    return np.array(curve)


def steps_to(curve, target):
    hit = np.nonzero(curve <= target)[0]
    return (hit[0] + 1) * 10 if len(hit) else None


if __name__ == "__main__":
    opts = {
        "SGD-momentum": lambda: SGDM(lr=0.05),
        "Muon (Nesterov)": lambda: Muon(lr=0.02),
        "AIMuon (Nesterov, g=0.25)": lambda: AIMuon(lr=0.02, gamma=0.25),
        "Muon (plain momentum)": lambda: Muon(lr=0.02, nesterov=False),
        "AIMuon (plain, g=0.5)": lambda: AIMuon(lr=0.02, gamma=0.5, nesterov=False),
    }
    curves = {n: np.stack([train(f, s) for s in range(SEEDS)]) for n, f in opts.items()}
    target = float(np.median(curves["Muon (Nesterov)"][:, -1]))
    print(f"{SEEDS} seeds, {STEPS} steps, batch {BATCH}, MLP {SIZES}; full-data MSE loss")
    print(f"target = Muon(Nesterov) median final loss = {target:.5f}\n")
    print(f"{'optimizer':27s} {'loss@200':>9s} {'loss@400':>9s} {'loss@600 (mean+-std)':>22s}  seeds hitting target")
    for n, c in curves.items():
        hits = sum(steps_to(c[s], target) is not None for s in range(SEEDS))
        print(f"{n:27s} {c[:, 19].mean():9.5f} {c[:, 39].mean():9.5f} {c[:, -1].mean():12.5f} +- {c[:, -1].std():.5f}  {hits}/{SEEDS}")
