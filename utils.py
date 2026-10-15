"""
utils.py — Filename parsing, year detection, and input validation.
"""
import re
import numpy as np
from pathlib import Path

# Matches e.g. "S1_site_agriyear_YYYY-MM_partX" or "S2_site_agriyear_YYYY-MM_partX"
_FILE_RE = re.compile(r"(?:S1|S2)_[\w]*?_(\d{4})_\d{4}-(\d{2})", re.IGNORECASE)

VALID_MONTHS = {f"m{m:02d}" for m in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]}


# -------------------------------------------------------------
# FILENAME PARSER
# -------------------------------------------------------------

def parse_filename(path: Path) -> tuple[int, str]:
    """Return (agri_year, month_tag) from a filename, e.g. (2022, 'm09')."""
    m = _FILE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot parse agri_year / month from filename: {path.name}")
    return int(m.group(1)), f"m{m.group(2)}"


# -------------------------------------------------------------
# YEAR DETECTION
# -------------------------------------------------------------

def detect_years(
    s1_files: list[Path],
    s2_files: list[Path],
    sensor: str,
) -> tuple[list[int], int, list[int]]:
    """
    Scan filenames and return (all_years, test_year, train_years).

    - all_years  : sorted list of years present for the chosen sensor(s)
    - test_year  : the latest year (held-out)
    - train_years: every year except test_year
    """
    def _years_in(files: list[Path]) -> set[int]:
        yrs = set()
        for p in files:
            m = _FILE_RE.search(p.stem)
            if m:
                yrs.add(int(m.group(1)))
        return yrs

    s1_years = _years_in(s1_files)
    s2_years = _years_in(s2_files)

    if sensor == "S1":
        common = sorted(s1_years)
    elif sensor == "S2":
        common = sorted(s2_years)
    else:  # "both"
        common = sorted(s1_years & s2_years)

    if not common:
        raise ValueError(
            f"No years found for sensor='{sensor}'. "
            f"S1: {sorted(s1_years)}, S2: {sorted(s2_years)}."
        )

    test_year   = max(common)
    train_years = [y for y in common if y != test_year]

    print(f"  [years] Detected   : {common}")
    print(f"  [years] Train years: {train_years}")
    print(f"  [years] Test year  : {test_year}")
    return common, test_year, train_years


# -------------------------------------------------------------
# INPUT VALIDATION
# -------------------------------------------------------------

def validate_input_files(
    s1_files: list[Path],
    s2_files: list[Path],
    all_years: list[int],
    sensor: str,
) -> None:
    """
    Raise ValueError if any of the following checks fail:
      - directories are non-empty
      - filenames match the expected pattern
      - every year in all_years has at least one file
      - month tags are within the agricultural season (Sep–Jun)
    """
    errors: list[str] = []

    files_to_check: list[tuple[str, list[Path]]] = []
    if sensor in ("S1", "both"):
        files_to_check.append(("S1", s1_files))
    if sensor in ("S2", "both"):
        files_to_check.append(("S2", s2_files))

    for label, files in files_to_check:
        if not files:
            errors.append(f"{label}: no files found in directory")
            continue

        years_found: set[int] = set()
        for p in files:
            m = _FILE_RE.search(p.stem)
            if not m:
                errors.append(f"{label}: filename does not match expected pattern: {p.name}")
                continue
            year      = int(m.group(1))
            month_tag = f"m{m.group(2)}"
            if year in all_years:
                years_found.add(year)
            if month_tag not in VALID_MONTHS:
                errors.append(f"{label}: unexpected month '{month_tag}' in {p.name}")

        for yr in all_years:
            if yr not in years_found:
                errors.append(f"{label}: no files found for year {yr}")

    if errors:
        print("\n[validate] FILE VALIDATION ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
        raise ValueError(f"File validation failed with {len(errors)} error(s). See above.")

    print("  [validate] All files passed validation.")


def to_db(linear: np.ndarray, nan_on_nonpositive: bool = True) -> np.ndarray:
    """
    Convert linear backscatter/reflectance values to dB.
    nan_on_nonpositive=True  --> non-positive values become NaN (for filters, so
                                  diff()-based comparisons don't treat them as real dB)
    nan_on_nonpositive=False --> non-positive values are clipped to 1e-10 (for plots,
                                  so lines don't break)
    """
    arr = np.asarray(linear, dtype=float)
    if nan_on_nonpositive:
        safe = np.where(arr > 0, arr, np.nan)
    else:
        safe = np.clip(arr, 1e-10, None)
    return 10 * np.log10(safe)

# -------------------------------------------------------------
# DATA REQUIREMENTS
# -------------------------------------------------------------
def required_data(modes: list[str]) -> str:
    """Which raw data a list of sensor modes needs: 'S1', 'S2' or 'both'."""
    needs = set()
    for m in modes:
        if m.startswith("S1"):
            needs.add("S1")
        elif m == "S2":
            needs.add("S2")
        elif m.startswith("both"):
            needs.update({"S1", "S2"})
        else:
            raise ValueError(f"Unknown sensor mode: {m!r}")
    if not needs:
        raise ValueError("No sensor modes configured.")
    return "both" if needs == {"S1", "S2"} else needs.pop()
