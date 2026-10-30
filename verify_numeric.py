"""Recompute numerical outputs and emit a reproducible SHA-256 verification receipt."""

import csv
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

from src.dense_kernel import interchanged_ikj, naive_ijk
from src.false_sharing import measure as measure_counters
from src.salt import get_student_params
from src.scaling_study import measure as measure_stencil


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TOLERANCE = 1e-12


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_digest(array):
    """Hash rounded values as little-endian float64 bytes."""
    rounded = np.round(array, 6).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes(order="C")).hexdigest()


def load_tsv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def verify_m0():
    params = json.loads((RESULTS / "salted_params.json").read_text(encoding="utf-8"))
    profile = json.loads((RESULTS / "host_profile.json").read_text(encoding="utf-8"))
    assert params == get_student_params(params["student_id"])
    assert profile["cpu"]["model"]
    assert profile["topology"]["physical_cores"] > 0
    assert profile["topology"]["logical_threads"] >= profile["topology"]["physical_cores"]
    assert profile["topology"]["cache_line_bytes"] > 0
    assert all(profile["topology"]["cache_bytes_per_instance"][key] for key in ("L1_data", "L2", "L3"))
    assert profile["os"]["kernel_version"] and profile["python"]["version"]
    print("[PASS] Milestone 0: Host topology and salted parameters validated.", flush=True)
    return params


def verify_m1(params):
    stride_rows = load_tsv("stride_results.tsv")
    assert [int(row["stride"]) for row in stride_rows] == [1, 2, 4, 8, 16, 32, 64, 128]
    for row in stride_rows:
        stride = int(row["stride"])
        count = (params["buffer_elements"] + stride - 1) // stride
        assert int(row["elements_touched"]) == count
        assert float(row["checksum"]) == count
        assert float(row["median_time_ns"]) > 0
        assert math.isclose(float(row["ns_per_element"]), float(row["median_time_ns"]) / count)

    stored = json.loads((RESULTS / "loop_interchange.json").read_text(encoding="utf-8"))
    size = params["N"] // 2
    assert int(stored["measured_block_M"]) == size
    rng = np.random.default_rng(params["seed"])
    a = rng.random((size, size), dtype=np.float64)
    b = rng.random((size, size), dtype=np.float64)
    baseline = naive_ijk(a, b)
    optimized = interchanged_ikj(a, b)
    error = float(np.max(np.abs(baseline - optimized)))
    assert error <= TOLERANCE, f"matrix error {error} exceeds {TOLERANCE}"
    baseline_digest = numerical_digest(baseline)
    optimized_digest = numerical_digest(optimized)
    assert baseline_digest == optimized_digest, "rounded matrix digests disagree"
    assert math.isclose(
        float(stored["speedup"]),
        float(stored["median_time_ns"]["naive_ijk"]) / float(stored["median_time_ns"]["interchanged_ikj"]),
    )
    print(f"[PASS] Milestone 1: Matrix numerical digest matches reference: {optimized_digest[:12]}...", flush=True)
    return optimized_digest


def verify_m2():
    rows = load_tsv("false_sharing_karp_flatt.tsv")
    assert len(rows) == 12
    checks = {}
    for row in rows:
        mode = row["mode"]
        processes = int(row["processes"])
        iterations = int(row["iterations_per_process"])
        assert mode in ("unpadded", "padded", "privatized")
        assert processes in (1, 2, 4, 8)
        assert int(row["total_count"]) == processes * iterations
        if processes == 1:
            checks[mode] = float(row["time_ns"])
        else:
            expected_e = (processes * float(row["time_ns"]) / checks[mode] - 1) / (processes - 1)
            assert math.isclose(float(row["karp_flatt_as_specified"]), expected_e)
        _, actual_count = measure_counters(processes, iterations, mode)
        assert actual_count == processes * iterations
    assert {(row["mode"], int(row["processes"])) for row in rows} == {
        (mode, p) for mode in ("unpadded", "padded", "privatized") for p in (1, 2, 4, 8)
    }
    print("[PASS] Milestone 2: All shared counter totals exactly match P x K.", flush=True)


def verify_m3(params):
    rows = load_tsv("scaling_metrics.tsv")
    assert len(rows) == 8
    assert {(row["scaling"], int(row["processes"])) for row in rows} == {
        (scaling, p) for scaling in ("strong", "weak") for p in (1, 2, 4, 6)
    }
    digests = {}
    for row in rows:
        scaling, processes = row["scaling"], int(row["processes"])
        dimension = int(row["dimension"])
        assert dimension == (params["N"] if scaling == "strong" else math.ceil(params["N"] * math.sqrt(processes)))
        assert int(row["work_cells"]) == (dimension - 2) ** 2 * int(row["repetitions"])
        assert float(row["max_absolute_error"]) <= TOLERANCE
        source_raw = mp.RawArray("d", dimension * dimension)
        output_raw = mp.RawArray("d", dimension * dimension)
        source = np.frombuffer(source_raw, dtype=np.float64).reshape(dimension, dimension)
        output = np.frombuffer(output_raw, dtype=np.float64).reshape(dimension, dimension)
        seed = params["seed"] if scaling == "strong" or processes == 1 else params["seed"] + processes
        source[:] = np.random.default_rng(seed).random((dimension, dimension))
        reference = 0.2 * (
            source[:-2, 1:-1] + source[2:, 1:-1] + source[1:-1, :-2]
            + source[1:-1, 2:] + source[1:-1, 1:-1]
        )
        measure_stencil(source_raw, output_raw, dimension, processes, 1)
        actual = output[1:-1, 1:-1]
        error = float(np.max(np.abs(reference - actual)))
        assert error <= TOLERANCE, f"{scaling} P={processes} error {error}"
        actual_digest = numerical_digest(actual)
        assert actual_digest == numerical_digest(reference)
        digests[f"{scaling}_P{processes}"] = actual_digest
        print(f"[PASS] Milestone 3: {scaling} P={processes}, digest {actual_digest[:12]}...", flush=True)
    roofline = json.loads((RESULTS / "roofline_summary.json").read_text(encoding="utf-8"))
    assert math.isclose(roofline["arithmetic_intensity_flop_per_byte_no_reuse"], 5 / 48)
    assert math.isclose(roofline["arithmetic_intensity_flop_per_byte_ideal_reuse"], 5 / 16)
    return digests


def write_receipt(params, matrix_digest, stencil_digests):
    files = [
        *sorted((ROOT / "src").glob("*.py")),
        ROOT / "verify_outputs.py",
        ROOT / "README.md",
        *sorted(path for path in RESULTS.iterdir() if path.name != "verification_receipt.sig"),
    ]
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in files}
    receipt = {
        "format": "local_sha256_verification_receipt_v1",
        "authentication": "unsigned; no course signing key or provided verifier was available",
        "student_id": params["student_id"],
        "matrix_rounded_6_sha256": matrix_digest,
        "stencil_rounded_6_sha256": stencil_digests,
        "file_sha256": manifest,
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    (RESULTS / "verification_receipt.sig").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("All local empirical datasets verified. Emitting verification_receipt.sig.", flush=True)


def main():
    params = verify_m0()
    matrix_digest = verify_m1(params)
    verify_m2()
    stencil_digests = verify_m3(params)
    write_receipt(params, matrix_digest, stencil_digests)


if __name__ == "__main__":
    main()
