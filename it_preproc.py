"""
it_preproc.py — Constants and helpers for the InceptionTime pipeline

(S1 channel names, orbit-filter map, sensor-to-base mapping).
Per-file loading lives in common_preproc.py.
"""
from __future__ import annotations

S1_TS_COLS = ["VV", "VH", "VV_VH_ratio"]

# Orbit filter per single S1 sensor
_S1_ORBIT_FILTER: dict[str, str | None] = {
    "S1":      None,
    "S1_ASC":  "ASCENDING",
    "S1_DESC": "DESCENDING",
}


def _s1_base(sensor: str) -> str:
    """'S1' for any single S1 variant, 'S2' otherwise."""
    return "S1" if sensor.startswith("S1") else "S2"


