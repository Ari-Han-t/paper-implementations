
### KV capacity 40,000 tokens, 200 long-output requests, predictor noise 0.25

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive | 46959.6 | 11.3 | 679 | 516183 | 20977.6 | 45385.3 |
| watermark | 48179.8 | 11.02 | 236 | 199652 | 21396.6 | 45957.8 |
| proactive | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |

### KV capacity 60,000 tokens, 200 long-output requests, predictor noise 0.25

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive | 36692.8 | 14.46 | 631 | 486356 | 15863.0 | 34734.9 |
| watermark | 37060.1 | 14.32 | 178 | 139891 | 16037.1 | 35293.3 |
| proactive | 38168.9 | 13.9 | 45 | 36392 | 16683.0 | 36626.5 |

### Sensitivity of proactive policy to length-prediction error (capacity 40,000)

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| proactive, noise=0.0 | 50476.4 | 10.51 | 20 | 20723 | 22736.0 | 49344.8 |
| proactive, noise=0.25 | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |
| proactive, noise=0.6 | 66326.3 | 8.0 | 42 | 45344 | 30883.4 | 63721.2 |
| proactive, noise=1.0 | 69834.4 | 7.6 | 22 | 23444 | 31818.5 | 66337.1 |

### Sensitivity to recompute cost (capacity 40,000)

| policy | makespan | throughput_tok_per_t | preemptions | recompute_tokens | mean_latency | p99_latency |
|---|---|---|---|---|---|---|
| reactive, recompute cost=0.0015/tok | 46959.6 | 11.3 | 679 | 516183 | 20977.6 | 45385.3 |
| watermark, recompute cost=0.0015/tok | 48179.8 | 11.02 | 236 | 199652 | 21396.6 | 45957.8 |
| proactive, recompute cost=0.0015/tok | 52086.0 | 10.19 | 58 | 56491 | 23329.6 | 50201.8 |
| reactive, recompute cost=0.005/tok | 49216.1 | 10.78 | 690 | 534541 | 22096.4 | 47557.4 |
| watermark, recompute cost=0.005/tok | 48949.7 | 10.84 | 227 | 187673 | 21871.8 | 46569.9 |
| proactive, recompute cost=0.005/tok | 51215.7 | 10.36 | 72 | 71659 | 23188.9 | 49656.7 |
| reactive, recompute cost=0.01/tok | 51959.4 | 10.21 | 671 | 525252 | 23662.0 | 50279.0 |
| watermark, recompute cost=0.01/tok | 49966.2 | 10.62 | 228 | 185729 | 22508.5 | 48099.0 |
| proactive, recompute cost=0.01/tok | 52053.6 | 10.2 | 54 | 57469 | 23234.5 | 49513.8 |
| reactive, recompute cost=0.02/tok | 57818.6 | 9.18 | 677 | 527932 | 26824.6 | 56127.3 |
| watermark, recompute cost=0.02/tok | 52518.8 | 10.11 | 226 | 181840 | 23800.7 | 50441.5 |
| proactive, recompute cost=0.02/tok | 53609.4 | 9.9 | 68 | 69873 | 24357.3 | 52121.3 |
