"""Compare i-j-k and i-k-j dense matrix multiplication loop order."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def naive_ijk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer i,j loops; NumPy performs each strided k reduction in native code."""
    m = a.shape[0]
    c = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            c[i, j] = np.sum(a[i, :] * b[:, j])
    return c


def interchanged_ikj(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer i,k loops; each j update walks a contiguous row of B and C."""
    m = a.shape[0]
    c = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for k in range(m):
            c[i, :] += a[i, k] * b[k, :]
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    params = json.loads((ROOT / "results" / "salted_params.json").read_text(encoding="utf-8"))
    m = params["N"] // 2
    rng = np.random.default_rng(params["seed"])
    a = rng.random((m, m), dtype=np.float64)
    b = rng.random((m, m), dtype=np.float64)
    timings = {"naive_ijk": [], "interchanged_ikj": []}
    outputs = {}
    for name, function in (("naive_ijk", naive_ijk), ("interchanged_ikj", interchanged_ikj)):
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            result = function(a, b)
            timings[name].append(time.perf_counter_ns() - start)
        outputs[name] = result
        print(f"{name}: median {statistics.median(timings[name]) / 1e9:.3f} s")
    maximum_error = float(np.max(np.abs(outputs["naive_ijk"] - outputs["interchanged_ikj"])))
    if maximum_error > 1e-12:
        raise AssertionError(f"matrix results disagree by {maximum_error}")
    naive_median = statistics.median(timings["naive_ijk"])
    interchanged_median = statistics.median(timings["interchanged_ikj"])
    summary = {
        "student_id": params["student_id"],
        "salted_N": params["N"],
        "measured_block_M": m,
        "runs": args.runs,
        "time_ns": timings,
        "median_time_ns": {"naive_ijk": naive_median, "interchanged_ikj": interchanged_median},
        "speedup": naive_median / interchanged_median,
        "max_absolute_error": maximum_error,
        "method": "NumPy inner-loop operations; outer loop order retained",
    }
    (ROOT / "results" / "loop_interchange.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"speedup={summary['speedup']:.2f}x; max error={maximum_error:.3g}")


if __name__ == "__main__":
    main()
