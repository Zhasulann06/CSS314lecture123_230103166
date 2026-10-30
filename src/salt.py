"""Deterministic benchmark dimensions derived from the supplied student ID."""

import hashlib


def get_student_params(student_id: str) -> dict:
    """Return the exact SHA-256-based parameters specified by the practicum."""
    if not student_id or not student_id.isdecimal():
        raise ValueError("student_id must contain decimal digits")

    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    return {
        "student_id": student_id,
        "N": 1024 + (h % 9) * 64,
        "buffer_elements": 16_777_216,
        "seed": h % 100_007,
    }
