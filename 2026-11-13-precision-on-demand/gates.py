"""Uncertainty gates and evaluation metrics for precision-on-demand decoding."""
import numpy as np

from model import forward, log_softmax


def entropy(logp):
    return -(np.exp(logp) * logp).sum(-1)


def top2_margin(logp):
    """Probability gap between top-1 and top-2 tokens (small gap = uncertain)."""
    p = np.sort(np.exp(logp), axis=-1)
    return p[..., -1] - p[..., -2]


def uncertainty_scores(logp_low):
    """Higher score = more uncertain = more deserving of escalation."""
    return {"entropy": entropy(logp_low), "margin": -top2_margin(logp_low)}


def threshold_for_budget(scores, budget):
    """Threshold such that ~`budget` fraction of steps have score > threshold."""
    if budget <= 0:
        return np.inf
    if budget >= 1:
        return -np.inf
    return float(np.quantile(scores, 1.0 - budget))


def auc(scores, labels):
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def replay(params_hi, params_lo, X):
    """Teacher-forced replay: low/high log-probs for every step."""
    lhi = log_softmax(forward(params_hi, X)[0])
    llo = log_softmax(forward(params_lo, X)[0])
    return lhi, llo


def gated_logp(lhi, llo, escalate):
    """Policy output: high-pass distribution where escalated, low-pass elsewhere."""
    return np.where(escalate[:, None], lhi, llo)


def kl(lp, lq):
    """KL(p || q) per row, inputs are log-probs."""
    return (np.exp(lp) * (lp - lq)).sum(-1)


def generate(params_hi, params_lo, prompt_ids, n_tokens, ctx, policy="high",
             gate="entropy", threshold=None, rate=None, rng=None):
    """Free-running greedy decoding under a precision policy.

    policy: "high" | "low" | "gate" | "random"
    Returns (generated ids, number of escalated steps).
    """
    ids = list(prompt_ids)
    escalated = 0
    rng = rng or np.random.default_rng(0)
    for _ in range(n_tokens):
        x = np.asarray(ids[-ctx:])[None, :]
        if policy == "high":
            lp = log_softmax(forward(params_hi, x)[0])[0]
            escalated += 1
        else:
            lp = log_softmax(forward(params_lo, x)[0])[0]
            if policy == "gate":
                score = uncertainty_scores(lp[None])[gate][0]
                esc = score > threshold
            elif policy == "random":
                esc = rng.random() < rate
            else:
                esc = False
            if esc:
                lp = log_softmax(forward(params_hi, x)[0])[0]
                escalated += 1
        ids.append(int(lp.argmax()))
    return ids[len(prompt_ids):], escalated
