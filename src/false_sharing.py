"""Compare shared counters, cache-line padding, and local privatization."""

import argparse
import csv
import ctypes
import multiprocessing as mp
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker(counters, base, worker_id, iterations, mode):
    index = base + (worker_id * 8 if mode == "padded" else worker_id)
    if mode == "privatized":
        local = 0
        for _ in range(iterations):
            local += 1
        counters[index] = local
    else:
        for _ in range(iterations):
            counters[index] += 1


def measure(processes, iterations, mode):
    # Eight 64-bit counters fit exactly on one aligned cache line.
    counters = mp.RawArray(ctypes.c_int64, 72)
    address = ctypes.addressof(counters)
    base = (-address % 64) // ctypes.sizeof(ctypes.c_int64)
    jobs = [
        mp.Process(target=worker, args=(counters, base, p, iterations, mode))
        for p in range(processes)
    ]
    start = time.perf_counter_ns()
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    elapsed = time.perf_counter_ns() - start
    if any(job.exitcode != 0 for job in jobs):
        raise RuntimeError(f"{mode}: a worker exited unsuccessfully")
    indices = [base + (p * 8 if mode == "padded" else p) for p in range(processes)]
    total = sum(counters[index] for index in indices)
    if total != processes * iterations:
        raise AssertionError(f"{mode}: {total} != {processes * iterations}")
    return elapsed, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    rows = []
    baselines = {}
    for mode in ("unpadded", "padded", "privatized"):
        for p in (1, 2, 4, 8):
            elapsed, total = measure(p, args.iterations, mode)
            if p == 1:
                baselines[mode] = elapsed
            speedup = baselines[mode] / elapsed
            karp_flatt = ((p * elapsed / baselines[mode]) - 1) / (p - 1) if p > 1 else ""
            row = {
                "mode": mode,
                "processes": p,
                "iterations_per_process": args.iterations,
                "total_count": total,
                "time_ns": elapsed,
                "speedup_vs_single_process": speedup,
                "karp_flatt_as_specified": karp_flatt,
                "increments_per_second": total * 1e9 / elapsed,
            }
            rows.append(row)
            print(f"{mode:11s} P={p} time={elapsed / 1e9:.3f}s e={karp_flatt}", flush=True)
    with (ROOT / "results" / "false_sharing_karp_flatt.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
