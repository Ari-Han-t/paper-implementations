import numpy as np
from anisoflow import make_W, target, pred_to_velocity, FlowModel, pca_basis, MLP, features


def _setup(w, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    W = make_W(Q, np.array(w, float))
    x, e = rng.standard_normal((50, 3)) * 2, rng.standard_normal((50, 3))
    t = rng.uniform(0.01, 0.99, (50, 1))
    return W, x, e, t, t * x + (1 - t) * e


def test_closed_form_recovers_velocity():
    W, x, e, t, xt = _setup([0.0, 0.7, 2.0])
    v = pred_to_velocity(target(x, e, W), xt, t, W)
    assert np.allclose(v, x - e, atol=1e-8)


def test_special_cases():
    W0, x, e, *_ = _setup([0, 0, 0])
    assert np.allclose(target(x, e, W0), x)          # x-prediction
    W1, x, e, *_ = _setup([1, 1, 1])
    assert np.allclose(target(x, e, W1), x - e)      # v-prediction


def test_mlp_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    net = MLP(4, 8, 2, rng)
    h = rng.standard_normal((5, 4)); g = rng.standard_normal((5, 2))
    net.forward(h); grads = net.backward(g)
    k, idx, eps = "W1", (1, 2), 1e-6
    net.p[k][idx] += eps; fp = (net.forward(h) * g).sum()
    net.p[k][idx] -= 2 * eps; fm = (net.forward(h) * g).sum()
    net.p[k][idx] += eps
    assert abs((fp - fm) / (2 * eps) - grads[k][idx]) < 1e-5


def test_training_reduces_loss_and_samples_reasonable():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4000, 2)) * [1.0, 0.2] + [2.0, 0.0]
    Q, var = pca_basis(data)
    assert var[0] > var[1]
    m = FlowModel(2, make_W(Q, np.array([0.5, 0.0])), hidden=64)
    losses = m.train(data, steps=800)
    assert np.mean(losses[-100:]) < np.mean(losses[:100])
    s = m.sample(1000, steps=30)
    assert abs(s[:, 0].mean() - 2.0) < 0.3
