# Variant B: Scalability Diagnostics

## Experiment A: Task Granularity

Host: Intel Core i5-11400H, 6 physical cores and 12 logical threads (Windows). The same 1,000,000 integers were processed in every run with `multiprocessing.Pool.map`; timing includes pool startup, task dispatch, computation, and result collection. The checksum was **131,434,424** in all eight runs. Raw nanosecond timings are in `exp_a_results.json`.

| Chunk size | Processes P | Time (s) | Speedup vs P=1 | Efficiency | Karp-Flatt e(P) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 67.696 | 1.00x | 100.0% | n/a |
| 1 | 2 | 74.671 | 0.91x | 45.3% | 1.206 |
| 1 | 4 | 88.724 | 0.76x | 19.1% | 1.414 |
| 1 | 8 | 83.674 | 0.81x | 10.1% | 1.270 |
| 10,000 | 1 | 11.196 | 1.00x | 100.0% | n/a |
| 10,000 | 2 | 6.747 | 1.66x | 83.0% | 0.205 |
| 10,000 | 4 | 4.250 | 2.63x | 65.9% | 0.173 |
| 10,000 | 8 | 3.119 | 3.59x | 44.9% | 0.176 |

For each chunk size, `S(P) = T(1)/T(P)`, `E(P) = 100*S(P)/P`, and `e(P) = (P*T(P)/T(1) - 1)/(P - 1)`. With chunk size 1, the one-million individual task envelopes overwhelm the useful Collatz work: every parallel configuration is slower than one process. Chunking by 10,000 reduces the number of task envelopes to about 100, and the eight-process run is **26.83x faster** than the eight-process micro-task run. Its Karp-Flatt values are roughly flat near 0.18-0.21, while the micro-task values are above 1 and fluctuate. The eight-process case also exceeds the host's six physical cores, so scheduler and SMT effects contribute to the measured ceiling.

## Experiment B: Queue Choke-Point

The student ran `py exp_b_queue.py` on the same host and supplied the terminal output. Each configuration sent and consumed 400,000 individual values through one `multiprocessing.Queue`. The times below are the displayed wall-clock values; speedups and Karp-Flatt metrics are from that output.

| Processes P | Time (s) | Speedup vs P=1 | Efficiency | Karp-Flatt e(P) |
|---:|---:|---:|---:|---:|
| 1 | 4.016 | 1.00x | 100.0% | n/a |
| 2 | 6.620 | 0.61x | 30.3% | 2.297 |
| 4 | 21.003 | 0.19x | 4.8% | 6.639 |
| 8 | 44.306 | 0.09x | 1.1% | 12.465 |

The runtime rises monotonically as workers are added, while throughput falls from about 99,600 to 9,030 items/s between P=1 and P=8. The sharply rising Karp-Flatt values show escalating overhead consistent with contention on the shared IPC queue, its pipe and synchronization mechanisms, and scheduling. Timing alone cannot isolate the contribution of each mechanism.

## Experiment C: Buffered Aggregation

The student ran `py -u exp_c_buffered.py` and supplied the terminal output. Each configuration processed 10,000,000 values, locally aggregated groups of 10,000, and sent only group sums through the queue. The independently calculated expected checksum was **29,999,994**; every configuration returned that value and the script reported `CHECKSUM: PASSED`.

| Processes P | Time (s) | Speedup vs P=1 | Efficiency | Karp-Flatt e(P) |
|---:|---:|---:|---:|---:|
| 1 | 0.885 | 1.00x | 100.0% | n/a |
| 2 | 0.552 | 1.60x | 80.1% | 0.248 |
| 4 | 0.419 | 2.11x | 52.8% | 0.298 |
| 8 | 0.478 | 1.85x | 23.2% | 0.474 |

At P=8, buffering handled 10 million items in 0.478 s, whereas the unbuffered queue needed 44.306 s for 400,000 items. The corresponding throughput is about 20.9 million versus 9,030 items/s. The buffered run is fastest at P=4; P=8 is slightly slower, consistent with overhead and the host's six physical cores. The measured S8 of 1.85 and e(8) of 0.474 do **not** reach the handout's example targets of S8 > 5 and e(8) <= 0.05; these are the observed measurements, not substituted target values.

## Comparative Summary

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(8) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A: Micro (k=1) | 67.696 | 74.671 | 88.724 | 83.674 | 0.81x | 1.206 | 1.270 | Dispatch overhead overwhelms work |
| A: Coarse (k=10,000) | 11.196 | 6.747 | 4.250 | 3.119 | 3.59x | 0.205 | 0.176 | Chunking amortizes dispatch cost |
| B: Queue choke | 4.016 | 6.620 | 21.003 | 44.306 | 0.09x | 2.297 | 12.465 | Shared queue becomes the bottleneck |
| C: Buffered batch | 0.885 | 0.552 | 0.419 | 0.478 | 1.85x | 0.248 | 0.474 | Local aggregation greatly improves throughput |
