"""
s1_monthly_fetcher_v3_quadrants.py
===================================
Extract Sentinel-1 VV, VH, angle per RPG parcel and per image date.

Architecture:
  - Single site, single GEE asset per crop year
  - Per-month batches over the agronomic season (Sept --> June)
  - Parcels split into 4 geographic quadrants to handle ~250k parcels
  - Images stacked into a single multi-band image per month
  - reduceRegions() called ONCE per month × quadrant on the stacked image
  - Server-side unstack --> one row per (parcel × image)
  - Sequential blocking execution for stable error detection
  - One CSV per (month × quadrant) exported to Google Drive
  - Completed-task check: skips already-SUCCEEDED exports on resume

Known limitation:
  Parcels straddling S1 orbit boundaries produce duplicate rows
  (same id_parcel × date). Fix in post-processing:
      key = ["id_parcel", "code_cultu", "code_group", "date", "orbit_direction", "orbit_number"]
      df  = df.groupby(key, as_index=False)[["VV", "VH", "angle"]].mean()


Modes:
    -- blocking : submit 4 quadrants per month, poll until they finish, then move to the next month (default,
    current behavior).
    -- submit   : fire every task for every year/month/quadrant without waiting,
    then print a summary, check the GEE Task Manager for progress.

Usage:
    python s1_monthly_fetcher_v3_quadrants.py --years 2021 2022 2023
"""

import argparse
import calendar
import sys
import time
from datetime import date, timedelta

import ee

# -- Auth ----------------------------------------------------------------------
EE_PROJECT = "your-gcp-project-id"      # SET this to your GCP project ID (must have Earth Engine enabled)

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# -- Config --------------------------------------------------------------------
SITE                 = "Bapaume"
S1_COLLECTION        = "COPERNICUS/S1_GRD"
ASSET_FOLDER         = f"projects/{EE_PROJECT}/assets/rpg_{SITE}"
DRIVE_FOLDER         = f"RPG_S1_Exports_{SITE}"

SEASON_MONTHS        = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
SCALE_M              = 10
WGS84                = "EPSG:4326"

STAT_BANDS  = ["VV", "VH", "angle"]
OUTPUT_COLS = [
    "id_parcel", "code_cultu", "code_group",
    "VV", "VH", "angle",
    "date", "orbit_direction", "orbit_number",
]

POLL_INTERVAL = 30
LOG_FILE      = f"logs/s1_run_{SITE}.log"
GRID_COLS     = 2                       # tiles west -> east
GRID_ROWS     = 2                       # tiles north -> south
N_PARTS       = GRID_COLS * GRID_ROWS


# -- Tee logger ----------------------------------------------------------------
class Tee:
    def __init__(self, filepath: str):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file     = open(filepath, "a", encoding="utf-8")
        self.terminal = sys.__stdout__

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# -- Temporal helpers ----------------------------------------------------------
def _month_bounds(year: int, month: int) -> tuple[str, str]:
    first = date(year, month, 1)
    last  = date(year, month, calendar.monthrange(year, month)[1])
    return first.isoformat(), (last + timedelta(days=1)).isoformat()


def season_months(crop_year: int) -> list[tuple[int, int, str, str]]:
    result   = []
    wraps    = SEASON_MONTHS[0] > SEASON_MONTHS[-1]   # True if season crosses Dec->Jan
    cal_year = crop_year - 1 if wraps else crop_year
    prev_m   = None

    for m in SEASON_MONTHS:
        if prev_m is not None and m < prev_m:
            cal_year += 1        # crossed into the next calendar year
        start, end = _month_bounds(cal_year, m)
        result.append((cal_year, m, start, end))
        prev_m = m

    return result


# -- Geographic quadrant split -------------------------------------------------
def make_tiles(fc: ee.FeatureCollection,
               n_cols: int = GRID_COLS,
               n_rows: int = GRID_ROWS) -> list[tuple[ee.FeatureCollection, str]]:
    """Split parcels into n_cols x n_rows tiles of the bounding box, by parcel centroid."""
    coords = ee.List(fc.geometry().bounds().coordinates().get(0))
    sw, se, nw = ee.List(coords.get(0)), ee.List(coords.get(1)), ee.List(coords.get(3))
    lon_min, lon_max = ee.Number(sw.get(0)), ee.Number(se.get(0))
    lat_min, lat_max = ee.Number(sw.get(1)), ee.Number(nw.get(1))
    dlon = lon_max.subtract(lon_min).divide(n_cols)
    dlat = lat_max.subtract(lat_min).divide(n_rows)

    fc_cx = fc.map(
        lambda f: f.set(
            "_cx", f.geometry().centroid(1).coordinates().get(0),
            "_cy", f.geometry().centroid(1).coordinates().get(1),
        )
    )

    def band(prop, lo, hi, last):
        upper = ee.Filter.lte(prop, hi) if last else ee.Filter.lt(prop, hi)
        return ee.Filter.And(ee.Filter.gte(prop, lo), upper)

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            lon0, lon1 = lon_min.add(dlon.multiply(c)), lon_min.add(dlon.multiply(c + 1))
            lat0, lat1 = lat_min.add(dlat.multiply(r)), lat_min.add(dlat.multiply(r + 1))
            sub = fc_cx.filter(ee.Filter.And(
                band("_cx", lon0, lon1, c == n_cols - 1),
                band("_cy", lat0, lat1, r == n_rows - 1),
            ))
            tiles.append((sub, f"r{r + 1}c{c + 1}"))
    return tiles


# -- Sentinel-1: dB --> linear -------------------------------------------------
def _db_to_linear(img: ee.Image) -> ee.Image:
    linear = ee.Image(10).pow(img.select(["VV", "VH"]).divide(10))
    return (
        img.addBands(linear, overwrite=True)
           .set("date",            img.date().format("YYYY-MM-dd"))
           .set("orbit_direction", img.get("orbitProperties_pass"))
           .set("orbit_number",    img.get("relativeOrbitNumber_stop"))
    )


# -- Collection builder --------------------------------------------------------
def build_s1_monthly(start: str, end: str, aoi: ee.Geometry) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(STAT_BANDS)
        .map(_db_to_linear)
    )


# -- Zonal statistics ----------------------------------------------------------
def zonal_stats_monthly(
    ic: ee.ImageCollection,
    fc: ee.FeatureCollection,
) -> ee.FeatureCollection:

    def reduce_one(image):
        image = ee.Image(image)
        return image.select(STAT_BANDS).reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=SCALE_M,
            crs=WGS84,
        ).map(lambda f: f.set(
            "date",            image.get("date"),
            "orbit_direction", image.get("orbit_direction"),
            "orbit_number",    image.get("orbit_number"),
        ))

    return (
        ic.map(reduce_one)
        .flatten()
        .filter(ee.Filter.notNull(["VV", "VH"]))
    )


# -- Completed-task cache ------------------------------------------------------
def completed_tasks() -> set[str]:
    ops = ee.data.listOperations()
    return {
        op.get("metadata", {}).get("description", "")
        for op in ops
        if op.get("metadata", {}).get("state") == "SUCCEEDED"
    }

def legacy_month_done(done: set[str], site: str, yr: int, month_str: str) -> bool:
    """Return True if a pre-quadrant export exists for this (site, year, month)."""
    legacy = f"S1_{site}_{yr}_{month_str}"
    return legacy in done


# -- Async export (submit only) ------------------------------------------------
def submit_export(fc: ee.FeatureCollection, description: str) -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=OUTPUT_COLS,
    )
    task.start()
    print(f"  [GEE] '{description}' submitted")
    return task


# -- Poll until all tasks in a batch finish ------------------------------------
def poll_batch(active: dict[str, ee.batch.Task], done: set[str]) -> tuple[int, int]:
    """
    Block until every task in `active` reaches a terminal state.
    Updates `done` in-place for resume safety.
    Returns (n_done, n_fail).
    """
    n_done = n_fail = 0
    pending = dict(active)
    start_time = time.time()

    while pending:
        time.sleep(POLL_INTERVAL)
        elapsed  = int(time.time() - start_time)
        finished = {}

        for desc, task in pending.items():
            state = task.status()["state"]
            if state == "COMPLETED":
                print(f"  [DRIVE] ✓ '{desc}' done  (+{elapsed//60:02d}:{elapsed%60:02d})")
                done.add(desc)
                n_done += 1
                finished[desc] = task
            elif state in ("FAILED", "CANCELLED"):
                err = task.status().get("error_message", "unknown")
                if "Not enough space" in err:
                    raise RuntimeError(f"Google Drive is full — cannot export '{desc}'")
                print(f"  [ERROR] ✗ '{desc}' failed: {err}")
                n_fail += 1
                finished[desc] = task

        for desc in finished:
            del pending[desc]

        if pending:
            print(
                f"  [Poll]  {len(pending)} still running  "
                f"+{elapsed//60:02d}:{elapsed%60:02d}",
                end="\r", flush=True,
            )

    return n_done, n_fail


# -- Main pipeline -------------------------------------------------------------
def extract_s1_monthly(crop_years: list[int], mode: str = "blocking") -> None:
    done = completed_tasks()

    total = len(crop_years) * len(SEASON_MONTHS) * N_PARTS
    print(f"\n{'='*60}")
    print(f"  S1 — Monthly extraction | site : {SITE}")
    print(f"  Mode     : {mode}")
    print(f"  Years   : {crop_years}")
    print(f"  Drive    : {DRIVE_FOLDER}")
    print(f"  Total    : ~{total} tasks  ({N_PARTS} quadrants × months)")
    print(f"{'='*60}")

    n_submitted = n_skip = n_fail = n_done = 0
    all_submitted: list[str] = []   # only used in submit mode, for the closing summary

    for yr in crop_years:
        asset_id = f"{ASSET_FOLDER}/rpg_{yr}_{SITE}"
        print(f"\n{'─'*60}")
        print(f"  [Year {yr}]  Asset : {asset_id}")

        try:
            ee.data.getAsset(asset_id)
        except ee.EEException:
            print(f"  [WARN] Asset introuvable : {asset_id} — ignoré.")
            continue

        fc        = ee.FeatureCollection(asset_id)
        quadrants = make_tiles(fc)
        months    = season_months(yr)
        print(f"  Season : {months[0][2][:7]} --> {months[-1][2][:7]}")
        print("  Months resolved :", ", ".join(f"{cy}-{m:02d}" for cy, m, _, _ in months))

        for cal_year, month, start, end in months:
            month_str = f"{cal_year}-{month:02d}"
            print(f"\n  [{month_str}]  {start} --> {end}")

            # Skip entire month if a legacy (pre-quadrant) export exists
            if legacy_month_done(done, SITE, yr, month_str):
                print(f"  [SKIP]  S1_{SITE}_{yr}_{month_str} (legacy export)")
                n_skip += N_PARTS
                continue

            active: dict[str, ee.batch.Task] = {}

            for part_idx, (sub_fc, quad_label) in enumerate(quadrants, start=1):
                description = f"S1_{SITE}_{yr}_{month_str}_part{part_idx}of{N_PARTS}"
                if description in done:
                    print(f"  [SKIP]  {description}")
                    n_skip += 1
                    continue

                aoi   = sub_fc.geometry().bounds()
                s1    = build_s1_monthly(start, end, aoi)
                stats = zonal_stats_monthly(s1, sub_fc)

                try:
                    task = submit_export(stats, description)
                    n_submitted += 1
                    if mode == "submit":
                        all_submitted.append(description)
                        if n_submitted % 20 == 0:
                            print(f"  [Progress] {n_submitted} tasks submitted so far...")
                        time.sleep(0.1)  # light courtesy delay, optional at this scale
                    else:
                        active[description] = task
                except Exception as e:
                    print(f"  [ERROR] Failed to submit {description}: {e}")
                    n_fail += 1

            # Blocking mode: wait for this month's batch before moving on
            if mode == "blocking" and active:
                d, f    = poll_batch(active, done)
                n_done += d
                n_fail += f

    print(f"\n{'='*60}")
    if mode == "submit":
        print(f"  Finished submitting — {n_submitted} Submitted | {n_skip} Skipped | {n_fail} Failed to submit")
        print("  All tasks are now running asynchronously on Earth Engine.")
        print("  Check progress in the GEE Task Manager: https://code.earthengine.google.com/tasks")
        if all_submitted:
            print("\n  Submitted tasks by year:")
            by_year: dict[str, list[str]] = {}
            for desc in all_submitted:
                # description format: S1_{SITE}_{yr}_{month_str}_part{n}
                yr_token = desc.split("_")[2]
                by_year.setdefault(yr_token, []).append(desc)
            for yr_token, descs in sorted(by_year.items()):
                print(f"    {yr_token}: {len(descs)} tasks")
    else:
        print(f"  Finished — {n_done} Exported | {n_skip} Skipped | {n_fail} Failed")
    print(f"{'='*60}")


# -- CLI -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=f"S1 — Monthly Extraction RPG ({SITE}) --> Google Drive"
    )
    p.add_argument(
        "--years", nargs="+", type=int, required=True,
        help="Year(s) of harvest (e.g., 2021 2022 2023)",
    )
    p.add_argument(
        "--mode", choices=["blocking", "submit"], default="blocking",
        help=(
            "blocking   : submit 4 quadrants per month, poll until they finish, "
            "then move to the next month (default, current behavior). "
            "submit : fire every task for every year/month/quadrant without waiting, "
            "then print a summary — check the GEE Task Manager for progress."
        ),
    )
    args = p.parse_args()

    tee = Tee(LOG_FILE)
    sys.stdout = tee
    try:
        extract_s1_monthly(crop_years=args.years, mode=args.mode)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    main()