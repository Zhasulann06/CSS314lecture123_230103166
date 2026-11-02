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
