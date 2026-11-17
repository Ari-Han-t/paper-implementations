"""Discrete-time simulator of a continuous-batching LLM server with a bounded KV cache.

Models the failure mode the paper calls "KV thrashing": in long-output workloads the
KV cache of every running request keeps growing, so a reactive scheduler that only
preempts when memory is already full keeps evicting and re-admitting the same
requests, paying a full recompute (prefill of prompt + generated tokens) each time.

Policies:
  reactive   - vLLM-style default. Admit FCFS whenever the prompt fits right now;
               when decode growth overflows, evict the most recently admitted
               request and recompute it later.
  watermark  - reactive, plus hysteresis: only (re)admit when free memory after
               admission stays above a watermark fraction of capacity.
  proactive  - length-aware: each request carries a (noisy) predicted output length.
               Admission reserves the request's projected peak KV footprint, so the
               batch is sized to finish without overflow. If the projection is
               violated anyway, it preempts early (before an OOM step) and picks
               the victim with the least sunk work, so less recompute is wasted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Request:
    rid: int
    arrival: float
    prompt: int
    output: int            # true output length (unknown to reactive policies)
    predicted: int         # predicted output length (used by proactive)
    generated: int = 0
    kv: int = 0            # KV tokens currently resident
    preemptions: int = 0
    finish: float | None = None
    admit_seq: int = 0     # order of most recent admission

    @property
    def done(self) -> bool:
        return self.generated >= self.output


@dataclass
class CostModel:
    step_base: float = 1.0          # fixed per decode step
    step_per_seq: float = 0.02      # per running sequence
    prefill_per_tok: float = 0.0015 # per prefilled (or recomputed) token


@dataclass
class Stats:
    time: float = 0.0
    steps: int = 0
    preemptions: int = 0
    recompute_tokens: int = 0
    useful_tokens: int = 0
    peak_kv: int = 0
    latencies: list = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {
            "makespan": round(self.time, 1),
            "throughput_tok_per_t": round(self.useful_tokens / max(self.time, 1e-9), 2),
            "preemptions": self.preemptions,
            "recompute_tokens": self.recompute_tokens,
            "mean_latency": round(float(lat.mean()), 1),
            "p99_latency": round(float(np.percentile(lat, 99)), 1),
            "peak_kv": self.peak_kv,
        }


def make_workload(n=200, seed=0, prompt_mean=300, out_median=2000, sigma=0.8,
                  rate=0.5, pred_noise=0.25):
    """Long-output workload: lognormal output lengths, Poisson arrivals."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / rate, n))
    prompts = rng.integers(prompt_mean // 2, prompt_mean * 3 // 2, n)
    outputs = np.clip(rng.lognormal(np.log(out_median), sigma, n), 50, 16000).astype(int)
    noise = rng.lognormal(0.0, pred_noise, n)
    preds = np.maximum((outputs * noise).astype(int), 1)
    return [Request(i, float(a), int(p), int(o), int(q))
            for i, (a, p, o, q) in enumerate(zip(arrivals, prompts, outputs, preds))]


def simulate(requests, capacity, policy="reactive", cost=CostModel(),
             watermark=0.1, horizon=8, max_steps=2_000_000):
    assert policy in ("reactive", "watermark", "proactive")
    reqs = [Request(**{k: getattr(r, k) for k in ("rid", "arrival", "prompt", "output", "predicted")})
            for r in requests]
    pending = sorted(reqs, key=lambda r: r.arrival)  # not yet arrived
    waiting: list[Request] = []                       # arrived / preempted, FCFS
    running: list[Request] = []
    st = Stats()
    seq = 0

    def used():
        return sum(r.kv for r in running)

    def projected_peak(batch, extra=None):
        extra = extra or {}
        kv = np.array([extra.get(x.rid, x.kv) for x in batch], dtype=np.int64)
        rem = np.array([max(x.predicted - x.generated, 1) for x in batch], dtype=np.int64)
        # usage(t) = sum_{i: rem_i > t} (kv_i + t); max is reached at t = rem_j - 1
        ts = rem - 1
        alive = rem[None, :] > ts[:, None]
        usage = (alive * (kv[None, :] + ts[:, None])).sum(axis=1)
        return int(usage.max())

    def can_admit(r):
        need = r.prompt + r.generated + 1  # recompute context + first new token
        if policy == "reactive":
            return used() + need <= capacity
        if policy == "watermark":
            return used() + need <= capacity * (1 - watermark)
        # proactive: simulate the batch's KV trajectory using predicted lengths
        # and admit only if its projected peak (incl. the newcomer) fits.
        if not running:
            return need <= capacity
        return projected_peak(running + [r], extra={r.rid: need}) <= capacity

    def preempt(victim):
        nonlocal st
        running.remove(victim)
        victim.kv = 0
        victim.preemptions += 1
        st.preemptions += 1
        waiting.insert(0, victim)  # recompute-style: back to the head of the queue

    while (pending or waiting or running) and st.steps < max_steps:
        while pending and pending[0].arrival <= st.time:
            waiting.append(pending.pop(0))
        if not running and not waiting:
            st.time = pending[0].arrival
            continue

        # --- admission (FCFS, stop at first that doesn't fit: no reordering) ---
        prefill = 0
        while waiting and can_admit(waiting[0]):
            r = waiting.pop(0)
            ctx = r.prompt + r.generated
            prefill += ctx
            if r.preemptions > 0:
                st.recompute_tokens += ctx  # wasted: rebuilding KV that was dropped
            r.kv = ctx
            seq += 1
            r.admit_seq = seq
            running.append(r)

        # --- proactive early preemption: overflow is imminent (within `horizon`
        # steps) AND the projection says it is real (not just a request about to
        # finish and free its memory) ---
        if policy == "proactive":
            while (len(running) > 1 and used() + horizon * len(running) > capacity
                   and projected_peak(running) > capacity):
                # least sunk work => cheapest to recompute
                preempt(min(running, key=lambda x: x.generated))

        # --- reactive preemption: this step's growth must fit ---
        while running and used() + len(running) > capacity:
            if len(running) == 1:
                raise RuntimeError("single request exceeds KV capacity")
            preempt(max(running, key=lambda x: x.admit_seq))  # LIFO, like vLLM

        # --- one decode step ---
        for r in running:
            r.kv += 1
            r.generated += 1
            # a request that outlives its prediction re-estimates (doubles) it
            if r.generated >= r.predicted:
                r.predicted = r.predicted * 2
        st.useful_tokens += len(running)
        st.peak_kv = max(st.peak_kv, used())
        assert used() <= capacity
        st.time += cost.step_base + cost.step_per_seq * len(running) + cost.prefill_per_tok * prefill
        st.steps += 1

        for r in [r for r in running if r.done]:
            running.remove(r)
            r.kv = 0
            r.finish = st.time
            st.latencies.append(r.finish - r.arrival)

    return st, reqs
