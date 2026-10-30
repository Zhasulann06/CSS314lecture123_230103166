"""Measure per-element read time for increasingly sparse memory access."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRIDES = (1, 2, 4, 8, 16, 32, 64, 128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("warmups must be nonnegative and runs must be positive")
    params = json.loads((ROOT / "results" / "salted_params.json").read_text(encoding="utf-8"))
    data = np.ones(params["buffer_elements"], dtype=np.float64)
    rows = []
    for stride in STRIDES:
        view = data[::stride]
        readings = []
        for repetition in range(args.warmups + args.runs):
            start = time.perf_counter_ns()
            total = float(np.sum(view))  # NumPy executes the reduction in native code.
            elapsed = time.perf_counter_ns() - start
            if total != len(view):
                raise AssertionError("stride reduction checksum mismatch")
            if repetition >= args.warmups:
                readings.append(elapsed)
        median_ns = statistics.median(readings)
        rows.append({
            "stride": stride,
            "elements_touched": len(view),
            "median_time_ns": median_ns,
            "ns_per_element": median_ns / len(view),
            "latency_ns": median_ns / len(view),
            "checksum": total,
        })
        print(f"stride={stride:3d}  {rows[-1]['ns_per_element']:.3f} ns/element")
    with (ROOT / "results" / "stride_results.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
