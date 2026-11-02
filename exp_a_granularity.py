"""Variant B, experiment A: multiprocessing task granularity."""

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path


def collatz_steps(n: int) -> int:
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers: int, total_items: int, chunk_size: int) -> tuple[int, int]:
    started = time.perf_counter_ns()
    with mp.Pool(processes=workers) as pool:
        values = pool.map(collatz_steps, range(1, total_items + 1), chunksize=chunk_size)
    return time.perf_counter_ns() - started, sum(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-items", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 10_000])
    parser.add_argument("--output", type=Path, default=Path("exp_a_results.json"))
    args = parser.parse_args()
    if args.total_items < 1 or any(p < 1 for p in args.workers) or any(k < 1 for k in args.chunks):
        parser.error("total items, worker counts, and chunk sizes must be positive")
    if 1 not in args.workers:
        parser.error("include one worker to calculate speedup and Karp-Flatt values")

    rows = []
    reference_sum = None
    for chunk_size in args.chunks:
        baseline_ns = None
        for workers in args.workers:
            elapsed_ns, checksum = run_granularity(workers, args.total_items, chunk_size)
            if reference_sum is None:
                reference_sum = checksum
            elif checksum != reference_sum:
                raise AssertionError(f"checksum differs: {checksum} != {reference_sum}")
            if workers == 1:
                baseline_ns = elapsed_ns
            speedup = baseline_ns / elapsed_ns
            karp_flatt = (
                (workers * elapsed_ns / baseline_ns - 1) / (workers - 1)
                if workers > 1 else None
            )
            row = {
                "chunk_size": chunk_size,
                "workers": workers,
                "elapsed_ns": elapsed_ns,
                "elapsed_s": elapsed_ns / 1e9,
                "speedup": speedup,
                "efficiency_percent": 100 * speedup / workers,
                "karp_flatt": karp_flatt,
                "checksum": checksum,
            }
            rows.append(row)
            e_text = "n/a" if karp_flatt is None else f"{karp_flatt:.3f}"
            print(
                f"chunk={chunk_size:5d} P={workers} time={row['elapsed_s']:.3f}s "
                f"speedup={speedup:.2f}x e={e_text} checksum={checksum}",
                flush=True,
            )

    payload = {
        "experiment": "A: task granularity",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_items": args.total_items,
        "workers": args.workers,
        "chunk_sizes": args.chunks,
        "checksum": reference_sum,
        "measurements": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
