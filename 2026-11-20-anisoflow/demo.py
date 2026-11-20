"""Experiments: fixed targets (x / v) vs anisotropic per-direction weights on 2D toy data."""
import itertools
import numpy as np
from anisoflow import FlowModel, make_W, pca_basis, sliced_wasserstein


def anisotropic_data(n, major_std, minor_std, angle=np.pi / 6, seed=0):
    """Rotated 2D data: 4-mode mixture along the major axis, narrow Gaussian along the minor axis."""
    rng = np.random.default_rng(seed)
    centers = np.array([-1.5, -0.5, 0.5, 1.5]) * major_std / 1.12
    a = centers[rng.integers(0, 4, n)] + 0.08 * major_std * rng.standard_normal(n)
    b = minor_std * rng.standard_normal(n)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.stack([a, b], 1) @ R.T


def run(data, Q, w, steps, seed=0):
    m = FlowModel(2, make_W(Q, np.asarray(w, float)), seed=seed)
    m.train(data, steps=steps)
    samples = m.sample(2000)
    return m.velocity_mse(data), sliced_wasserstein(samples, data[:2000])


def main(steps=2500):
    data = anisotropic_data(20000, major_std=2.0, minor_std=0.1)
    Q, var = pca_basis(data)
    print(f"PCA variances: major={var[0]:.3f} minor={var[1]:.4f}")
    grid = [0.0, 0.25, 0.5, 1.0, 1.5]
    results = {}
    for w in itertools.product(grid, grid):
        mses, sws = zip(*[run(data, Q, w, steps, seed=s) for s in (0, 1)])
        results[w] = (np.mean(mses), np.mean(sws))
    print("\nw_major w_minor | val vel-MSE | sliced-W (samples vs data)")
    for w, (mse, sw) in sorted(results.items(), key=lambda kv: kv[1][0]):
        tag = {(0.0, 0.0): " <- x-pred", (1.0, 1.0): " <- v-pred"}.get(w, "")
        print(f"  {w[0]:4.2f}   {w[1]:4.2f}   |   {mse:.4f}    |  {sw:.4f}{tag}")
    best = min(results, key=lambda k: results[k][0])
    print(f"\nBest anisotropic w (selected by val vel-MSE): {best}")
    for name, w in [("x-pred", (0.0, 0.0)), ("v-pred", (1.0, 1.0)), ("anisotropic best", best)]:
        print(f"  {name:17s} vel-MSE={results[w][0]:.4f}  SW={results[w][1]:.4f}")

    # Variance-scaling check: best isotropic w for 1D-like data at different scales
    print("\nPreferred w vs data std (isotropic sweep per scale, 1 seed):")
    for std in (0.1, 0.5, 1.0, 2.0):
        d = anisotropic_data(20000, major_std=std, minor_std=std * 0.999, angle=0.0, seed=1)
        Qs = np.eye(2)
        scores = {w: run(d, Qs, (w, w), steps // 2)[0] for w in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)}
        bw = min(scores, key=scores.get)
        print(f"  std={std:4.1f}  best w={bw:4.2f}  " + " ".join(f"{k}:{v:.3f}" for k, v in scores.items()))


if __name__ == "__main__":
    main()
