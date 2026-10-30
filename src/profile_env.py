"""Record host CPU, cache, OS, and salted benchmark parameters."""

import argparse
import ctypes
import json
import os
import platform
import re
import struct
import sys
from pathlib import Path

from salt import get_student_params


ROOT = Path(__file__).resolve().parents[1]


def windows_processor() -> dict:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    ) as key:
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        identifier = winreg.QueryValueEx(key, "Identifier")[0]
    frequency = re.search(r"@\s*([\d.]+)\s*GHz", name, re.IGNORECASE)
    return {
        "model": name,
        "identifier": identifier,
        "microarchitecture": "Tiger Lake" if "i5-11400H" in name else None,
        "base_frequency_mhz": round(float(frequency.group(1)) * 1000) if frequency else None,
        "base_frequency_source": "rated frequency in CPU model string" if frequency else None,
    }


def windows_topology() -> dict:
    """Read core and cache records from Windows GetLogicalProcessorInformationEx."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetLogicalProcessorInformationEx
    query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    query.restype = ctypes.c_int
    size = ctypes.c_ulong(0)
    query(0xFFFF, None, ctypes.byref(size))  # RelationAll: query required size.
    if not size.value:
        raise OSError(ctypes.get_last_error(), "Windows CPU topology query failed")
    buffer = ctypes.create_string_buffer(size.value)
    if not query(0xFFFF, buffer, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "Windows CPU topology query failed")

    cores = 0
    caches = {"L1_data": [], "L2": [], "L3": []}
    line_sizes = []
    offset = 0
    raw = buffer.raw
    while offset < size.value:
        relationship, record_size = struct.unpack_from("<II", raw, offset)
        if record_size < 8 or offset + record_size > size.value:
            raise ValueError("Invalid Windows CPU topology record")
        if relationship == 0:  # RelationProcessorCore
            cores += 1
        elif relationship == 2:  # RelationCache
            level, _, line_size, cache_size, cache_type = struct.unpack_from(
                "<BBHII", raw, offset + 8
            )
            if level == 1 and cache_type in (0, 1):  # Unified or Data
                caches["L1_data"].append(cache_size)
            elif level == 2:
                caches["L2"].append(cache_size)
            elif level == 3:
                caches["L3"].append(cache_size)
            if line_size:
                line_sizes.append(line_size)
        offset += record_size
    return {
        "physical_cores": cores or None,
        "logical_threads": os.cpu_count(),
        "smt_enabled": bool(cores and os.cpu_count() and os.cpu_count() > cores),
        "cache_line_bytes": max(line_sizes) if line_sizes else None,
        "cache_bytes_per_instance": {key: sorted(set(values)) for key, values in caches.items()},
        "cache_instance_counts": {key: len(values) for key, values in caches.items()},
    }


def linux_profile() -> tuple[dict, dict]:
    cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace")
    first = cpuinfo.split("\n\n", 1)[0]
    fields = dict(
        line.split(":", 1) for line in first.splitlines() if ":" in line
    )
    fields = {key.strip(): value.strip() for key, value in fields.items()}
    sys_cpu = Path("/sys/devices/system/cpu/cpu0/cache")
    caches = {"L1_data": [], "L2": [], "L3": []}
    lines = []
    if sys_cpu.exists():
        for entry in sys_cpu.glob("index*"):
            level = (entry / "level").read_text().strip()
            kind = (entry / "type").read_text().strip()
            size_text = (entry / "size").read_text().strip()
            match = re.fullmatch(r"(\d+)([KMG])", size_text)
            if not match:
                continue
            size = int(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
            key = "L1_data" if level == "1" and kind in ("Data", "Unified") else f"L{level}"
            if key in caches:
                caches[key].append(size)
            lines.append(int((entry / "coherency_line_size").read_text().strip()))
    physical_pairs = set(
        re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", cpuinfo, re.DOTALL)
    )
    cores = len(physical_pairs) or None
    mhz = fields.get("cpu MHz")
    processor = {
        "model": fields.get("model name") or platform.processor(),
        "identifier": f"family {fields.get('cpu family')} model {fields.get('model')} stepping {fields.get('stepping')}",
        "microarchitecture": None,
        "base_frequency_mhz": float(mhz) if mhz else None,
        "base_frequency_source": "current /proc/cpuinfo cpu MHz (may differ from base)" if mhz else None,
    }
    topology = {
        "physical_cores": cores,
        "logical_threads": os.cpu_count(),
        "smt_enabled": bool(cores and os.cpu_count() and os.cpu_count() > cores),
        "cache_line_bytes": max(lines) if lines else None,
        "cache_bytes_per_instance": {key: sorted(set(values)) for key, values in caches.items()},
        "cache_instance_counts": {key: len(values) for key, values in caches.items()},
    }
    return processor, topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True, help="Student ID used verbatim for SHA-256 salting")
    args = parser.parse_args()
    if sys.platform == "win32":
        processor, topology = windows_processor(), windows_topology()
    elif sys.platform.startswith("linux"):
        processor, topology = linux_profile()
    else:
        processor = {"model": platform.processor(), "microarchitecture": None, "base_frequency_mhz": None}
        topology = {"physical_cores": None, "logical_threads": os.cpu_count(), "cache_line_bytes": None}
    profile = {
        "cpu": processor,
        "topology": topology,
        "os": {"system": platform.system(), "release": platform.release(), "kernel_version": platform.version()},
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(), "compiler": platform.python_compiler()},
    }
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "host_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    params = get_student_params(args.student_id)
    (output / "salted_params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"host_profile": profile, "salted_params": params}, indent=2))


if __name__ == "__main__":
    main()
