"""Strong/weak scaling study and Roofline estimate for a five-point stencil."""

import argparse
import csv
import ctypes
import json
import multiprocessing as mp
import statistics
import time
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def stencil_rows(source_raw, output_raw, dimension, first_row, last_row, repetitions):
    """Update a disjoint set of interior rows from a fixed input matrix."""
    source = np.frombuffer(source_raw, dtype=np.float64).reshape(dimension, dimension)
    output = np.frombuffer(output_raw, dtype=np.float64).reshape(dimension, dimension)
    for _ in range(repetitions):
        output[first_row:last_row, 1:-1] = 0.2 * (
            source[first_row - 1:last_row - 1, 1:-1]
            + source[first_row + 1:last_row + 1, 1:-1]
            + source[first_row:last_row, :-2]
            + source[first_row:last_row, 2:]
            + source[first_row:last_row, 1:-1]
        )


def partition_rows(dimension, processes):
    interior = dimension - 2
    for worker_id in range(processes):
        first = 1 + worker_id * interior // processes
        last = 1 + (worker_id + 1) * interior // processes
        yield first, last


def measure(source_raw, output_raw, dimension, processes, repetitions):
    jobs = [
        mp.Process(
            target=stencil_rows,
            args=(source_raw, output_raw, dimension, first, last, repetitions),
        )
        for first, last in partition_rows(dimension, processes)
    ]
    started = time.perf_counter_ns()
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    elapsed = time.perf_counter_ns() - started
    if any(job.exitcode != 0 for job in jobs):
        raise RuntimeError("a stencil worker failed")
    return elapsed


def reference_stencil(source):
    return 0.2 * (
        source[:-2, 1:-1] + source[2:, 1:-1] + source[1:-1, :-2]
        + source[1:-1, 2:] + source[1:-1, 1:-1]
    )


def weak_study(params, profile, repetitions, runs):
    output_path = ROOT / "results" / "scaling_metrics.tsv"
    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        strong_rows = [row for row in reader if row["scaling"] == "strong"]
    if not strong_rows:
        raise ValueError("Run the strong-scaling phase first")
    if any(int(row["repetitions"]) != repetitions for row in strong_rows):
        raise ValueError("Weak and strong phases must use the same repetition count")
    baseline = next(row for row in strong_rows if int(row["processes"]) == 1)
    base_time_ns = float(baseline["median_time_ns"])
    base_work = int(baseline["interior_cells"]) * repetitions
    fitted_s = float(baseline["amdahl_fitted_serial_fraction"])
    worker_counts = [int(row["processes"]) for row in strong_rows]
    weak_rows = []

    for processes in worker_counts:
        dimension = math.ceil(params["N"] * math.sqrt(processes))
        work = (dimension - 2) ** 2 * repetitions
        if processes == 1:
            timings = [int(value) for value in baseline["run_times_ns"].split(",")]
            median_ns = base_time_ns
            max_error = float(baseline["max_absolute_error"])
        else:
            source_raw = mp.RawArray(ctypes.c_double, dimension * dimension)
            output_raw = mp.RawArray(ctypes.c_double, dimension * dimension)
            source = np.frombuffer(source_raw, dtype=np.float64).reshape(dimension, dimension)
            output = np.frombuffer(output_raw, dtype=np.float64).reshape(dimension, dimension)
            source[:] = np.random.default_rng(params["seed"] + processes).random((dimension, dimension))
            expected = reference_stencil(source)
            timings = [
                measure(source_raw, output_raw, dimension, processes, repetitions)
                for _ in range(runs)
            ]
            max_error = float(np.max(np.abs(output[1:-1, 1:-1] - expected)))
            if max_error > 1e-12:
                raise AssertionError(f"weak stencil differs from reference by {max_error}")
            median_ns = statistics.median(timings)
        weak_rows.append({
            "scaling": "weak",
            "processes": processes,
            "dimension": dimension,
            "interior_cells": (dimension - 2) ** 2,
            "repetitions": repetitions,
            "run_times_ns": ",".join(str(value) for value in timings),
            "median_time_ns": median_ns,
            "speedup": "",
            "efficiency_percent": "",
            "amdahl_serial_fraction": "",
            "max_absolute_error": max_error,
            "amdahl_fitted_serial_fraction": "",
            "amdahl_asymptotic_speedup": "",
            "work_cells": work,
            "work_ratio": work / base_work,
            "weak_scaled_speedup_as_specified": processes * base_work * base_time_ns / (work * median_ns),
            "scaled_throughput_speedup": (work / base_work) * (base_time_ns / median_ns),
            "gustafson_barsis_prediction": processes - fitted_s * (processes - 1),
            "time_ratio_to_p1": median_ns / base_time_ns,
        })
        print(
            f"weak P={processes} N={dimension} median={median_ns / 1e9:.3f}s "
            f"work={work / base_work:.2f}x scaled-throughput={weak_rows[-1]['scaled_throughput_speedup']:.2f}x",
            flush=True,
        )

    for row in strong_rows:
        row["work_cells"] = int(row["interior_cells"]) * repetitions
        row["work_ratio"] = 1
        row["weak_scaled_speedup_as_specified"] = ""
        row["scaled_throughput_speedup"] = ""
        row["gustafson_barsis_prediction"] = ""
        row["time_ratio_to_p1"] = ""
    rows = strong_rows + weak_rows
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_roofline(params, profile, rows)


def write_roofline(params, profile, rows):
    """Document an idealized bandwidth roof and its assumptions."""
    caches = profile["topology"]["cache_bytes_per_instance"]
    stride_path = ROOT / "results" / "stride_results.tsv"
    with stride_path.open(newline="", encoding="utf-8") as file:
        stride_one = next(row for row in csv.DictReader(file, delimiter="\t") if int(row["stride"]) == 1)
    observed_read_gb_s = (
        int(stride_one["elements_touched"]) * 8 / float(stride_one["median_time_ns"])
    )
    # DDR4-3200, two 64-bit channels: 3200e6 transfers/s * 8 B * 2 = 51.2 GB/s.
    theoretical_gb_s = 51.2
    flops_per_cell = 5  # Four additions and one multiplication.
    naive_bytes_per_cell = 6 * 8  # Five input loads plus one output store.
    reuse_bytes_per_cell = 2 * 8  # One amortized load plus one store.
    strong_p1 = next(row for row in rows if row["scaling"] == "strong" and int(row["processes"]) == 1)
    observed_gflops = (
        flops_per_cell * int(strong_p1["work_cells"]) / float(strong_p1["median_time_ns"])
    )
    summary = {
        "student_id": params["student_id"],
        "kernel": "five-point stencil: output = 0.2 * (center + north + south + east + west)",
        "floating_point_operations_per_cell": flops_per_cell,
        "bytes_per_cell_no_reuse": naive_bytes_per_cell,
        "bytes_per_cell_ideal_reuse": reuse_bytes_per_cell,
        "arithmetic_intensity_flop_per_byte_no_reuse": flops_per_cell / naive_bytes_per_cell,
        "arithmetic_intensity_flop_per_byte_ideal_reuse": flops_per_cell / reuse_bytes_per_cell,
        "cache_bytes_per_instance": caches,
        "strong_p1_input_and_output_bytes": 2 * params["N"] ** 2 * 8,
        "theoretical_memory_bandwidth_gb_s": theoretical_gb_s,
        "theoretical_bandwidth_assumption": "CPU supports up to DDR4-3200 and two memory channels; assumes both channels populated at 3200 MT/s, which is unverified for this host",
        "cpu_memory_spec_source": "https://www.intel.com/content/www/us/en/products/sku/213805/intel-core-i511400h-processor-12m-cache-up-to-4-50-ghz/specifications.html",
        "measured_stride_one_read_bandwidth_gb_s": observed_read_gb_s,
        "theoretical_bandwidth_roof_gflops_no_reuse": theoretical_gb_s * flops_per_cell / naive_bytes_per_cell,
        "theoretical_bandwidth_roof_gflops_ideal_reuse": theoretical_gb_s * flops_per_cell / reuse_bytes_per_cell,
        "observed_strong_p1_gflops": observed_gflops,
        "classification": "low-arithmetic-intensity; bandwidth-limited in an optimized Roofline model",
        "interpretation_limit": "The measured NumPy/multiprocessing implementation also includes array temporaries, Python dispatch, and process startup; these measurements do not isolate a DRAM bandwidth limit.",
    }
    (ROOT / "results" / "roofline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("strong", "weak"), default="strong")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or args.runs < 1:
        parser.error("repetitions and runs must be positive")

    params = json.loads((ROOT / "results" / "salted_params.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "results" / "host_profile.json").read_text(encoding="utf-8"))
    if args.phase == "weak":
        weak_study(params, profile, args.repetitions, args.runs)
        return
    dimension = params["N"]
    cores = profile["topology"]["physical_cores"] or 1
    worker_counts = sorted({1, *[p for p in (2, 4, 8, 16, 32) if p <= cores], cores})

    source_raw = mp.RawArray(ctypes.c_double, dimension * dimension)
    output_raw = mp.RawArray(ctypes.c_double, dimension * dimension)
    source = np.frombuffer(source_raw, dtype=np.float64).reshape(dimension, dimension)
    output = np.frombuffer(output_raw, dtype=np.float64).reshape(dimension, dimension)
    source[:] = np.random.default_rng(params["seed"]).random((dimension, dimension))
    expected = 0.2 * (
        source[:-2, 1:-1] + source[2:, 1:-1] + source[1:-1, :-2]
        + source[1:-1, 2:] + source[1:-1, 1:-1]
    )

    rows = []
    baseline = None
    for processes in worker_counts:
        timings = [
            measure(source_raw, output_raw, dimension, processes, args.repetitions)
            for _ in range(args.runs)
        ]
        max_error = float(np.max(np.abs(output[1:-1, 1:-1] - expected)))
        if max_error > 1e-12:
            raise AssertionError(f"stencil result differs from reference by {max_error}")
        median_ns = statistics.median(timings)
        if processes == 1:
            baseline = median_ns
        speedup = baseline / median_ns
        serial_fraction = (
            (1 / speedup - 1 / processes) / (1 - 1 / processes)
            if processes > 1 else ""
        )
        rows.append({
            "scaling": "strong",
            "processes": processes,
            "dimension": dimension,
            "interior_cells": (dimension - 2) ** 2,
            "repetitions": args.repetitions,
            "run_times_ns": ",".join(str(value) for value in timings),
            "median_time_ns": median_ns,
            "speedup": speedup,
            "efficiency_percent": 100 * speedup / processes,
            "amdahl_serial_fraction": serial_fraction,
            "max_absolute_error": max_error,
        })
        print(
            f"P={processes} median={median_ns / 1e9:.3f}s "
            f"speedup={speedup:.2f} efficiency={100 * speedup / processes:.1f}% "
            f"s={serial_fraction}",
            flush=True,
        )

    # Least-squares fit of 1/S = 1/P + s(1 - 1/P), constrained to a physical fraction.
    xs = [1 - 1 / row["processes"] for row in rows if row["processes"] > 1]
    ys = [1 / row["speedup"] - 1 / row["processes"] for row in rows if row["processes"] > 1]
    fitted_s = min(1.0, max(0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)))
    for row in rows:
        row["amdahl_fitted_serial_fraction"] = fitted_s
        row["amdahl_asymptotic_speedup"] = 1 / fitted_s if fitted_s > 0 else "unbounded"

    output_path = ROOT / "results" / "scaling_metrics.tsv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Amdahl fit: s={fitted_s:.4f}, limit={1 / fitted_s if fitted_s else 'unbounded'}")


if __name__ == "__main__":
    main()
