"""
rpg_fetcher.py
==============
Fetch Registre Parcellaire Graphique (RPG) parcels for an AOI over N years.
Uses IGN APICarto RPG v2 (2015–2024).

For each AOI:
    1. The AOI geometry is loaded, reprojected to WGS84, and unioned.
    2. The reference year is validated against APICarto (falls back to the
       nearest earlier available year if needed).
    3. Parcels are downloaded for `--nyears` consecutive years ending at
       `--year`, optionally filtered server-side by crop code (`code_cultu`)
       or crop group (`code_groupe`).
    4. Each year's parcels are saved as a separate Shapefile.
    5. If `--filtered` is given, parcels are additionally filtered by area:
       geometries are reprojected to Lambert 93 (EPSG:2154), buffered inward
       by 10 m, and any parcel smaller than 500 m² after buffering is
       dropped. This removes small/thin parcels unsuitable for pixel-based
       analysis (e.g. 10 m resolution satellite time series).

Output structure:
    <output>/<aoi_name>/rpg_<year>/rpg_<year>_<aoi_name>.shp      (raw)
    <filtered>/<aoi_name>/rpg_<year>/rpg_<year>_<aoi_name>.shp    (filtered, optional)

Years already present in the output directory are skipped automatically,
so re-running the script resumes rather than re-downloading.

Usage:
    # Single AOI, single year, raw output only
    python rpg_fetcher.py --aoi my_area.shp --year 2024 --output output/

    # Multiple AOIs, 3 years of history, raw + area-filtered output
    python rpg_fetcher.py --aoi my_area.shp my_other_area.shp
        --year 2024 --nyears 3 --output output/ --filtered output_filtered/

    # Restrict to a specific crop code
    python rpg_fetcher.py --aoi my_area.shp --year 2024 
        --code_cultu PPH --output output/

Requirements:
    geopandas, requests, shapely, tqdm
"""

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
from urllib3.util.retry import Retry

from tqdm import tqdm

BASE_URL  = "https://apicarto.ign.fr/api/rpg/v2"
PAGE_SIZE = 1000
WGS84     = "EPSG:4326"
PAGE_DELAY = 0.5   # seconds between paginated requests

# ── Filtering config ──────────────────────────────────────────────────────────
FILTER_CRS   = "EPSG:2154"   # Lambert 93 — metres
BUFFER_DIST  = -10           # metres, inner buffer
MIN_AREA     = 500           # m² after buffer (1 pixel S2 is 100m²)


# ── Session ───────────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ── Year validation ───────────────────────────────────────────────────────────

def resolve_year(session: requests.Session, geom_json: str, requested_year: int) -> int:
    for yr in range(requested_year, 2014, -1):
        params = {"annee": yr, "geom": geom_json, "_limit": 1, "_start": 0}
        try:
            resp = session.get(BASE_URL, params=params, timeout=30)
        except requests.exceptions.RequestException:
            continue
        if resp.status_code == 200:
            if yr != requested_year:
                print(f"[year] {requested_year} not available — using {yr} instead.")
            return yr
    raise RuntimeError(f"No available RPG year found from {requested_year} down to 2015.")


# ── AOI ───────────────────────────────────────────────────────────────────────

def load_aoi(filepath: str) -> tuple[str, str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"AOI file not found: {filepath}")

    gdf = gpd.read_file(filepath)
    print(f"[AOI] {len(gdf)} feature(s), CRS = {gdf.crs}")

    if gdf.crs is None:
        raise ValueError("AOI has no CRS defined.")
    if gdf.crs.to_epsg() != 4326:
        print(f"[AOI] Reprojecting {gdf.crs} --> {WGS84}")
        gdf = gdf.to_crs(WGS84)

    invalid = gdf[~gdf.geometry.is_valid]
    if not invalid.empty:
        print(f"[AOI] Repairing {len(invalid)} invalid geometry(ies) ...")
        gdf["geometry"] = gdf["geometry"].apply(make_valid)

    union = unary_union(gdf.geometry)
    return json.dumps(mapping(union), separators=(",", ":")), path.stem


# ── Save helpers ──────────────────────────────────────────────────────────────

def out_file(out_dir: Path, year: int) -> Path:
    site = out_dir.name
    return out_dir / f"rpg_{year}" / f"rpg_{year}_{site}.shp"

def year_already_fetched(out_dir: Path, year: int) -> bool:
    return out_file(out_dir, year).exists()

def save_year(gdf: gpd.GeoDataFrame, year: int, out_dir: Path):
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: make_valid(geom).buffer(0) if geom is not None else geom
    )
    gdf.columns = [
        c if c == "geometry" else c.replace(" ", "_").replace(".", "_")[:10]
        for c in gdf.columns
    ]
    fp = out_file(out_dir, year)
    fp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(fp, driver="ESRI Shapefile")
    size_mb = fp.stat().st_size / 1_048_576
    print(f"[save] rpg_{year} — {len(gdf):,} parcels --> {fp.parent}  ({size_mb:.1f} MB)")


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_shp(shp_in: Path, shp_out: Path):
    gdf = gpd.read_file(shp_in)
    n_init = len(gdf)

    # Repair
    gdf = gdf[gdf.geometry.notna()]
    gdf.geometry = gdf.geometry.buffer(0)

    # Reproject to metres
    if gdf.crs is None or gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(FILTER_CRS)

    # Filter by buffered area
    mask = gdf.geometry.buffer(BUFFER_DIST).area >= MIN_AREA
    gdf_filtered = gdf[mask].copy()

    n_final   = len(gdf_filtered)
    n_removed = n_init - n_final
    pct       = n_removed / n_init * 100 if n_init else 0

    shp_out.parent.mkdir(parents=True, exist_ok=True)
    gdf_filtered.to_file(shp_out, driver="ESRI Shapefile")
    print(f"[filter] {shp_in.parent.name} — {n_init:,} → {n_final:,} parcels "
          f"({n_removed:,} removed, {pct:.1f}%)")


def filter_site(raw_site_dir: Path, filtered_root: Path):
    """Filter all shapefiles under a site directory and mirror structure."""
    shp_files = list(raw_site_dir.rglob("*.shp"))
    if not shp_files:
        return

    site_name = raw_site_dir.name
    print(f"\n[filter] ── Filtering site: {site_name} ({len(shp_files)} file(s)) ──")

    for shp_in in shp_files:
        rel     = shp_in.relative_to(raw_site_dir.parent)
        shp_out = filtered_root / rel
        filter_shp(shp_in, shp_out)


# ── API ───────────────────────────────────────────────────────────────────────

def fetch_year(
    session: requests.Session,
    geom_json: str,
    year: int,
    code_cultu: str | None = None,
    code_groupe: str | None = None,
) -> gpd.GeoDataFrame:
    features, start = [], 0
    bar = None

    while True:
        params = {"annee": year, "geom": geom_json,
                  "_limit": PAGE_SIZE, "_start": start}
        if code_cultu:
            params["code_cultu"]  = code_cultu
        if code_groupe:
            params["code_groupe"] = code_groupe

        try:
            resp = session.get(BASE_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"[{year}] Request failed at start={start}") from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"[{year}] HTTP {resp.status_code} at start={start}: {resp.text[:200]}"
            )

        data  = resp.json()
        total = data.get("totalFeatures", 0) or data.get("numberMatched", 0)
        batch = data.get("features", [])

        if bar is None:
            bar = tqdm(total=total, desc=f"{year}", unit="parcel", leave=True)

        features.extend(batch)
        bar.update(len(batch))

        if len(features) >= total or len(batch) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(PAGE_DELAY)

    if bar:
        bar.close()

    if not features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(features, crs=WGS84)
    gdf["annee"] = year
    return gdf


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_rpg(
    aoi_path: str,
    ref_year: int,
    n_years: int = 1,
    code_cultu: str | None = None,
    code_groupe: str | None = None,
    out_dir: Path | None = None,
    filtered_dir: Path | None = None,
) -> gpd.GeoDataFrame:

    session   = make_session()
    geom_json, aoi_name = load_aoi(aoi_path)
    ref_year  = resolve_year(session, geom_json, ref_year)
    years     = list(range(ref_year - n_years + 1, ref_year + 1))
    print(f"\n[RPG] AOI={aoi_path}  years={years}")

    if out_dir:
        out_dir = out_dir / aoi_name
        out_dir.mkdir(parents=True, exist_ok=True)

    collected, fetched_years = [], []

    for yr in years:
        if out_dir and year_already_fetched(out_dir, yr):
            print(f"[{yr}] Already saved in {out_dir} — skipping.")
            fetched_years.append(yr)
            continue

        try:
            gdf = fetch_year(session, geom_json, yr, code_cultu, code_groupe)
        except RuntimeError as exc:
            print(f"[{yr}] ERROR — skipping year: {exc}")
            continue

        if gdf.empty:
            continue

        if out_dir:
            save_year(gdf, yr, out_dir)
            fetched_years.append(yr)
        else:
            collected.append(gdf)

    if out_dir:
        total = sum(
            len(gpd.read_file(out_file(out_dir, yr))) for yr in fetched_years
        )
        print(f"[done] {total:,} total parcels across {fetched_years} --> {out_dir}")

        # ── Filtering step ────────────────────────────────────────────────────
        if filtered_dir:
            filter_site(out_dir, filtered_dir)

        return gpd.GeoDataFrame()

    if not collected:
        return gpd.GeoDataFrame()

    result = gpd.GeoDataFrame(
        gpd.pd.concat(collected, ignore_index=True),
        geometry="geometry", crs=WGS84,
    )
    print(f"\n[done] {len(result):,} total parcels | years: {sorted(result['annee'].unique())}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _t0 = time.perf_counter()

    p = argparse.ArgumentParser()
    p.add_argument("--aoi",           required=True, nargs="+")
    p.add_argument("--year",          required=True, type=int)
    p.add_argument("--nyears",        default=1,     type=int)
    p.add_argument("--code_cultu",    default=None)
    p.add_argument("--code_groupe",   default=None)
    p.add_argument("--output",        default=None,
                   help="Output directory (one Shapefile per year)")
    p.add_argument("--filtered",      default=None,
                   help="Filtered output directory (mirrors --output structure)")
    args = p.parse_args()

    out_dir      = Path(args.output)   if args.output   else None
    filtered_dir = Path(args.filtered) if args.filtered else None

    for aoi_path in args.aoi:
        print(f"\n{'='*60}\n[AOI] Processing: {aoi_path}\n{'='*60}")
        fetch_rpg(
            aoi_path, args.year, args.nyears,
            args.code_cultu, args.code_groupe,
            out_dir=out_dir,
            filtered_dir=filtered_dir,
        )

    elapsed = time.perf_counter() - _t0
    print(f"\n[time] Total runtime: {elapsed/60:.1f} min ({elapsed:.1f}s)")