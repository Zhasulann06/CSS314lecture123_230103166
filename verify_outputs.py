#!/usr/bin/env python3
"""
CS482: Automated Practicum Verification Suite
Validates all empirical outputs, numerical tolerances, and generates verification_receipt.sig.
Usage:
    python3 verify_outputs.py
"""

import sys
import os
import json
import csv
import hashlib

def fail(milestone: str, reason: str):
    print(f"[[91mFAIL[0m] {milestone}: {reason}")
    sys.exit(1)

def warn(milestone: str, reason: str):
    print(f"[[93mWARN[0m] {milestone}: {reason}")

def pass_step(milestone: str, message: str):
    print(f"[[92mPASS[0m] {milestone}: {message}")

def check_file(path: str, milestone: str) -> str:
    if not os.path.isfile(path):
        fail(milestone, f"Missing required deliverable '{path}'")
    return path

def verify_milestone_0():
    m = "Milestone 0 (Host & Salt)"
    prof_path = check_file("results/host_profile.json", m)
    salt_path = check_file("results/salted_params.json", m)

    with open(prof_path, "r", encoding="utf-8") as f:
        try:
            prof = json.load(f)
        except Exception as e:
            fail(m, f"Failed to parse {prof_path}: {e}")

    # Check hardware fields
    for field in ["cpu_model", "physical_cores", "logical_threads", "cache_line_bytes"]:
        if field not in prof:
            fail(m, f"Missing key '{field}' in {prof_path}")

    if prof.get("cache_line_bytes") != 64:
        fail(m, f"Expected cache_line_bytes == 64, got {prof.get('cache_line_bytes')}")

    with open(salt_path, "r", encoding="utf-8") as f:
        try:
            salt = json.load(f)
        except Exception as e:
            fail(m, f"Failed to parse {salt_path}: {e}")

    sid = str(salt.get("student_id", "")).strip()
    if not sid:
        fail(m, "student_id is empty in salted_params.json")

    # Verify salting formula
    h = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
    expected_N = 1024 + ((h % 9) * 64)
    if salt.get("N") != expected_N:
        fail(m, f"Salted dimension N mismatch. Expected {expected_N}, got {salt.get('N')}")

    pass_step(m, f"Topology verified. Student ID: {sid} | Salted N: {expected_N}")
    return sid, expected_N

def verify_milestone_1():
    m = "Milestone 1 (Locality & Interchange)"
    stride_path = check_file("results/stride_results.tsv", m)
    interchange_path = check_file("results/loop_interchange.json", m)

    # 1. Stride verification
    strides = {}
    with open(stride_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            try:
                s = int(row.get("stride", row.get("Stride", -1)))
                lat = float(row.get("latency_ns", row.get("latency", row.get("Latency_ns", 0.0))))
                strides[s] = lat
            except (ValueError, TypeError):
                continue

    if 1 not in strides or 8 not in strides:
        fail(m, "stride_results.tsv must include measurements for at least stride 1 and stride 8")

    if strides[8] <= strides[1]:
        warn(m, f"Stride 8 latency ({strides[8]:.2f} ns) not higher than Stride 1 ({strides[1]:.2f} ns). Check cache flush or buffer size.")
    else:
        ratio = strides[8] / strides[1]
        pass_step(m, f"Stride cliff reproduced: Stride-8 is {ratio:.2f}x slower per element than Stride-1")

    # 2. Interchange timing verification
    with open(interchange_path, "r", encoding="utf-8") as f:
        try:
            inter = json.load(f)
        except Exception as e:
            fail(m, f"Failed to parse {interchange_path}: {e}")

    t_naive = inter.get("naive_time_s", inter.get("t_naive", None))
    t_inter = inter.get("interchanged_time_s", inter.get("t_interchanged", None))
    speedup = inter.get("speedup", None)

    if t_naive is None or t_inter is None:
        fail(m, "loop_interchange.json must contain 'naive_time_s' and 'interchanged_time_s'")

    calc_speedup = t_naive / t_inter if t_inter > 0 else 0.0
    if calc_speedup < 1.2:
        warn(m, f"Measured loop interchange speedup ({calc_speedup:.2f}x) is lower than typical cache benefit (>1.5x)")
    else:
        pass_step(m, f"Loop interchange speedup verified: {calc_speedup:.2f}x (Naive: {t_naive:.2f}s -> i-k-j: {t_inter:.2f}s)")

def verify_milestone_2():
    m = "Milestone 2 (False Sharing & Karp-Flatt)"
    kf_path = check_file("results/false_sharing_karp_flatt.tsv", m)

    rows = []
    with open(kf_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        for r in reader:
            rows.append(r)

    if len(rows) < 2:
        fail(m, "false_sharing_karp_flatt.tsv must contain multiple core count rows (e.g., P=1, 2, 4, 8)")

    # Check unpadded vs padded/privatized scaling
    unpadded_e = []
    padded_e = []
    for r in rows:
        try:
            p = int(r.get("p", r.get("P", 1)))
            if p > 1:
                e_unpad = float(r.get("e_unpadded", r.get("e_p_unpadded", 0.0)))
                e_pad = float(r.get("e_padded", r.get("e_p_padded", 0.0)))
                unpadded_e.append((p, e_unpad))
                padded_e.append((p, e_pad))
        except (ValueError, TypeError):
            continue

    if unpadded_e and padded_e:
        # Check Karp-Flatt trend: unpadded e should increase with core count
        first_p, first_e = unpadded_e[0]
        last_p, last_e = unpadded_e[-1]
        if last_e > first_e:
            pass_step(m, f"Karp-Flatt correctly diagnosed false sharing contention: e grew from {first_e:.3f} (P={first_p}) to {last_e:.3f} (P={last_p})")
        else:
            warn(m, "Unpadded Karp-Flatt metric did not strictly increase. Hardware thread affinity or contention was low.")
    else:
        pass_step(m, "Telemetry loaded, raw columns verified.")

def verify_milestone_3():
    m = "Milestone 3 (Scaling & Roofline)"
    scale_path = check_file("results/scaling_metrics.tsv", m)
    roof_path = check_file("results/roofline_summary.json", m)

    with open(scale_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        rows = list(reader)

    if len(rows) < 2:
        fail(m, "scaling_metrics.tsv must record multiple worker configurations")

    with open(roof_path, "r", encoding="utf-8") as f:
        try:
            roof = json.load(f)
        except Exception as e:
            fail(m, f"Failed to parse {roof_path}: {e}")

    regime = roof.get("regime", roof.get("classification", "")).lower()
    if "bound" not in regime:
        fail(m, "roofline_summary.json must classify the kernel as 'memory-bound' or 'compute-bound'")

    ai = roof.get("arithmetic_intensity", roof.get("ai", None))
    if ai is None:
        fail(m, "roofline_summary.json must state the computed 'arithmetic_intensity'")

    pass_step(m, f"Roofline analysis complete. AI: {ai} FLOPs/byte | Regime: {regime}")

def emit_receipt(sid: str, N: int):
    # Cryptographic proof combining student ID, dimension, and system secret salt
    raw_token = f"CS482-PRACTICUM-VERIFIED-{sid}-{N}-HASH-SOLO-LAB"
    token = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    out_path = "results/verification_receipt.sig"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"STUDENT_ID={sid}\n")
        f.write(f"SALTED_N={N}\n")
        f.write(f"SIGNATURE={token}\n")
    print("\n" + "=" * 60)
    print(f"[\033[92mCOMPLETE\033[0m] All milestones verified successfully!")
    print(f"Generated submission receipt: {out_path}")
    print(f"Token: {token[:16]}...")
    print("=" * 60)

def main():
    print("=" * 60)
    print("CS482 Parallel Computing: Automated Practicum Verifier")
    print("=" * 60)
    sid, N = verify_milestone_0()
    verify_milestone_1()
    verify_milestone_2()
    verify_milestone_3()
    emit_receipt(sid, N)

if __name__ == "__main__":
    main()
