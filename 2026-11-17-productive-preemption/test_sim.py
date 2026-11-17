import pytest

from sim import Request, make_workload, simulate

POLICIES = ["reactive", "watermark", "proactive"]


@pytest.mark.parametrize("policy", POLICIES)
def test_all_requests_complete_and_memory_bounded(policy):
    w = make_workload(n=60, seed=1)
    st, reqs = simulate(w, 30_000, policy)
    assert all(r.finish is not None and r.generated == r.output for r in reqs)
    assert st.peak_kv <= 30_000
    assert st.useful_tokens == sum(r.output for r in w)


@pytest.mark.parametrize("policy", POLICIES)
def test_no_preemption_when_memory_is_ample(policy):
    w = make_workload(n=40, seed=2)
    st, _ = simulate(w, 10_000_000, policy)
    assert st.preemptions == 0 and st.recompute_tokens == 0


def test_reactive_thrashes_under_pressure():
    st, reqs = simulate(make_workload(n=100, seed=3), 30_000, "reactive")
    assert st.preemptions > 50
    assert max(r.preemptions for r in reqs) >= 3  # same request evicted repeatedly


def test_proactive_reduces_wasted_recompute():
    w = make_workload(n=100, seed=3)
    r = simulate(w, 30_000, "reactive")[0]
    p = simulate(w, 30_000, "proactive")[0]
    assert p.recompute_tokens < 0.3 * r.recompute_tokens
    assert p.preemptions < 0.3 * r.preemptions


def test_deterministic():
    w = make_workload(n=50, seed=4)
    assert simulate(w, 25_000, "proactive")[0].summary() == simulate(w, 25_000, "proactive")[0].summary()


def test_oversized_request_raises():
    big = [Request(0, 0.0, prompt=100, output=500, predicted=500)]
    with pytest.raises(RuntimeError):
        simulate(big, 300, "reactive")
