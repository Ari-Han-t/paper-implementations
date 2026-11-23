import numpy as np
from verkv import VerKV, dense_attention, softmax

def make_cache(n=4096, d=64, n_hot=40, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.normal(size=(n, d)) * 0.6
    V = rng.normal(size=(n, d))
    topics = rng.normal(size=(8, d))
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    hot = rng.choice(n, n_hot, replace=False)
    owner = rng.integers(0, 8, n_hot)
    K[hot] += topics[owner] * 8  # tokens that some queries attend to strongly
    return K, V, topics, rng

def queries(topics, rng, m=200, d=64):
    t = rng.integers(0, 8, m)
    Q = topics[t] * 10.0 + rng.normal(size=(m, d)) * 0.3
    # 25% "diffuse" queries that look at nothing in particular
    Q[: m // 4] = rng.normal(size=(m // 4, d)) * 0.3
    return Q

VRMS = None
def rel(a, b):
    return np.linalg.norm(a - b) / VRMS

K, V, topics, rng = make_cache()
Q = queries(topics, rng)
n = K.shape[0]
VRMS = np.sqrt((np.linalg.norm(V, axis=1)**2).mean())
print('error metric: ||o_hat - o_dense|| / rms(||v||)')
for B in [128, 256, 512]:
  print(f"\n== cache: {n} tokens, d=64; {len(Q)} queries; resident pivots ~{B} ({B/n:.1%} of cache), delta=0.1 ==")
  print(f"{'method':<28}{'mean err':>10}{'p95 err':>10}{'max err':>10}{'viol>eps':>10}{'fetch/q':>10}{'%cache':>8}")
  for eps in [None, 0.10, 0.05, 0.02]:
      m = VerKV(K, V, budget=B, eps=eps if eps else 1e9, delta=0.1)
      errs, fetched = [], []
      for q in Q:
          o, info = m.step(q)
          errs.append(rel(o, dense_attention(q, K, V)))
          fetched.append(info["fetched"])
      errs = np.array(errs)
      name = "pivots only (no verify)" if eps is None else f"VerKV eps={eps}"
      viol = (errs > eps).mean() if eps else float("nan")
      print(f"{name:<28}{errs.mean():>10.4f}{np.percentile(errs,95):>10.4f}{errs.max():>10.4f}"
            f"{viol:>10.2%}{np.mean(fetched):>10.1f}{np.mean(fetched)/n:>8.2%}")
m = VerKV(K, V, budget=512, eps=0.05)
info = [m.step(q)[1] for q in Q]
foc = [x["fetched"] for x in info[50:]]
dif = [x["fetched"] for x in info[:50]]
print(f"\nbudget 512, eps=0.05: focused queries fetched median {np.median(foc):.0f} rows"
      f" ({np.mean(np.array(foc) == 0):.0%} needed zero fetches); diffuse queries median {np.median(dif):.0f} rows")
