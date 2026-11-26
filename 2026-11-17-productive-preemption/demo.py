"""Run the KV-thrashing experiment and print markdown tables."""
from sim import CostModel, make_workload, simulate

COLS = ["makespan", "throughput_tok_per_t", "preemptions", "recompute_tokens",
        "mean_latency", "p99_latency"]


def table(title, rows):
    print(f"\n### {title}\n")
    print("| policy | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name, s in rows:
        print(f"| {name} | " + " | ".join(str(s[c]) for c in COLS) + " |")


def main():
    for cap in (40_000, 60_000):
        w = make_workload(n=200, seed=0, pred_noise=0.25)
        rows = [(p, simulate(w, cap, p)[0].summary())
                for p in ("reactive", "watermark", "proactive")]
        table(f"KV capacity {cap:,} tokens, 200 long-output requests, predictor noise 0.25", rows)

    w_rows = []
    for noise in (0.0, 0.25, 0.6, 1.0):
        w = make_workload(n=200, seed=0, pred_noise=noise)
        w_rows.append((f"proactive, noise={noise}", simulate(w, 40_000, "proactive")[0].summary()))
    table("Sensitivity of proactive policy to length-prediction error (capacity 40,000)", w_rows)

    w = make_workload(n=200, seed=0, pred_noise=0.25)
    c_rows = []
    for c in (0.0015, 0.005, 0.01, 0.02):
        for p in ("reactive", "watermark", "proactive"):
            c_rows.append((f"{p}, recompute cost={c}/tok",
                           simulate(w, 40_000, p, cost=CostModel(prefill_per_tok=c))[0].summary()))
    table("Sensitivity to recompute cost (capacity 40,000)", c_rows)


if __name__ == "__main__":
    main()
