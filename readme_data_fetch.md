# Data Preparation: RPG parcels and Sentinel-1/2 time series

This guide explains how to get the input data for the Winter Crops Classifier
(see `classifier_readme.md`). The result is generic: for any set of agricultural parcels
you obtain CSV files with Sentinel-1 (radar) and Sentinel-2 (optical) values
per parcel and per acquisition date. They can be reused for other classifiers.

**In short:** you provide an area of interest, download the official
crop-parcel outlines (RPG) for several years, fetch from Google Earth Engine
the satellite signal inside every parcel, and download the resulting CSV files
from Google Drive.

---

## 1. Overview

```
 Area of interest (.shp / .geojson / .gpkg)
        │  rpg_fetcher.py                          runs on your computer
        ▼
 RPG parcel shapefiles, one per year
        │  upload by hand (once per year)          Earth Engine web page
        ▼
 Earth Engine assets  rpg_<year>_<SITE>
        │  s1_monthly_fetcher_v3_quadrants.py      start jobs on Google's servers
        │  s2_monthly_fetcher_v3_quadrants.py
        ▼
 CSV files in Google Drive (one per month and tile)
        │  download by hand
        ▼
 Data/S1_extract_<SITE>/  and  Data/S2_extract_<SITE>/   --> classifier
```

| Step | Script | Runs where | Result |
|---|---|---|---|
| 1. Get parcels | `rpg_fetcher.py` | your computer | One shapefile per year |
| 2. Upload parcels | manual | Earth Engine web page | One asset per year |
| 3. Sentinel-1 | `s1_monthly_fetcher_v3_quadrants.py` | Earth Engine | CSVs in Drive |
| 4. Sentinel-2 | `s2_monthly_fetcher_v3_quadrants.py` | Earth Engine | CSVs in Drive |
| 5. Download | manual | your computer | `Data/` folders |

Data sources: parcel outlines and crop codes from the RPG (IGN, through
APICarto); Sentinel-1 `COPERNICUS/S1_GRD` and Sentinel-2
`COPERNICUS/S2_SR_HARMONIZED` from Google Earth Engine.

---

## 2. Before you start

- Python 3.10 or newer.
- `pip install geopandas requests shapely tqdm earthengine-api`
- A Google account with **Earth Engine access** and a **Google Cloud project**
  registered for Earth Engine. You need the project ID.
- **Google Drive space.** A large site can produce several GB of CSV per year.
  A full Drive stops the export with an error.
- An **AOI file** (area of interest): a polygon in any format geopandas can
  read, with a CRS defined. Keep it simple (a rectangle or a coarse outline).
  The geometry is sent inside a web request, and very detailed outlines can be
  rejected.

---

## 3. Decide names and years first

**Site name (`SITE`).** Use letters and digits only, with no spaces or
underscores (for example `Bapaume`, `LaRochelle`). The same name must appear in
all of these places:

| Where | Example |
|---|---|
| AOI file name | `Bapaume.shp` |
| `SITE` in both Sentinel scripts | `SITE = "Bapaume"` |
| Earth Engine asset names | `rpg_2024_Bapaume` |
| Data folders for the classifier | `S1_extract_Bapaume` |
| `SITE` in the classifier `config.py` | `SITE = "Bapaume"` |

**Years :** A year always means the **harvest year**: `2024` is the season from
September 2023 to June 2024. You need several past years with crop labels for
training (the project used 2020-2023) and the year to predict.

**Be the custodian of your data :** Each monthly file is a snapshot taken when
the export ran. Run the scripts for a month only after that month has ended,
and keep the downloaded files safe.

---
# Use case
## Step 1: Get the parcels (`rpg_fetcher.py`)

```bash
python rpg_fetcher.py --aoi Bapaume.shp --year 2024 --nyears 5 --output rpg_raw/ --filtered rpg_filtered/
```

This downloads the parcels of the 5 years ending in 2024 (2020-2024).

| Option | Meaning |
|---|---|
| `--aoi` | One or more AOI files. The file name without extension becomes the site name. |
| `--year` | Last year to fetch. |
| `--nyears` | How many consecutive years, ending at `--year`. |
| `--output` | Folder for the raw shapefiles. **Always set it**: without it nothing is saved. |
| `--filtered` | Folder for the area-filtered shapefiles (recommended). |
| `--code_cultu`, `--code_groupe` | Download only some crops. **Do not use them for this classifier**: it needs all crops, including "Other vegetation". |

Result:

```
rpg_raw/Bapaume/rpg_2024/rpg_2024_Bapaume.shp        (and the other years)
rpg_filtered/Bapaume/rpg_2024/rpg_2024_Bapaume.shp
```

What to know:

- **Resume.** A year that already exists in the output folder is skipped. If a
  year fails, the log shows `ERROR - skipping year`. Run the same command
  again.
- **Check the end of the log.** The line `[done] ... across [years]` lists the
  years actually saved. A year that is missing there was not downloaded, even
  if no error was printed.
- **Area filter.** Each parcel is shrunk by 10 m on every side, and parcels
  below 500 m² after shrinking are dropped. A Sentinel pixel is 10 m × 10 m
  (100 m²), so very small or thin parcels contain no clean pixel. The saved
  outlines are the **original ones**: the shrinking only decides which parcels
  are kept. The filtered files are written in Lambert 93 (EPSG:2154), which
  Earth Engine handles on upload.
- **Use the filtered parcems files** for the next step.

###  The newest year has no RPG yet

RPG outlines are published after the season, so **the year you want to predict
usually does not exist yet**. The Sentinel scripts only need parcel **limits**,
so reuse the latest outlines:

1. Fetch the labeled years, ending at the last published year (for example
   `--year 2025 --nyears 4`). If you ask for an unpublished year, the fetcher
   may move the whole window back (`[year] N not available - using M instead`)
   or skip it silently.
2. Copy the latest year's shapefile under the new year name. Copy **all** its
   files (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) and rename the folder and
   files, for example `rpg_2026/rpg_2026_Bapaume.shp`.
3. Upload it as the 2026 asset.

Parcels merged or split since then are not captured. The crop codes in the copy
belong to the previous year. The classifier ignores them for the predict year,
but never use them as 2026 ground truth.

---

## Step 2: Upload the parcels to Earth Engine

Do this once per year, using the **filtered** files.

1. Open the Earth Engine Code Editor (code.earthengine.google.com), tab
   **Assets**.
2. **New → Folder.** Name it `rpg_<SITE>`. The full path is
   `projects/<your-project>/assets/rpg_<SITE>`.
3. **New → Table upload → Shape files.** Select the `.shp`, `.shx`, `.dbf` and
   `.prj` files of one year.
4. Set the asset ID to exactly
   `projects/<your-project>/assets/rpg_<SITE>/rpg_<year>_<SITE>`. **This is very important so satelite data extraction scripts work directly after.**
5. Start the upload and wait until it shows as completed in the **Tasks** tab.
6. Repeat for every year, **including the year to predict**.
7. Click each asset and check that it has the properties `id_parcel`,
   `code_cultu` and `code_group`, and a feature count close to the one the
   fetcher printed. The export selects these three names, so a different name
   makes the export fail.

The scripts look for `{ASSET_FOLDER}/rpg_{year}_{SITE}`. A wrong name means
that year is skipped with `Asset introuvable` (asset not found).

---

## Step 3: Configure the two Sentinel scripts

The settings are duplicated at the top of both scripts. **Keep them identical
in both files.**

| Setting | What to set |
|---|---|
| `EE_PROJECT` | Your Google Cloud project ID. |
| `SITE` | The site name  |
| `ASSET_FOLDER` | Derived from `EE_PROJECT` and `SITE`. |
| `DRIVE_FOLDER` | Name of the Drive folder that receives the CSVs (created automatically). |
| `SEASON_MONTHS` | For this classifier keep `[9, 10, 11, 12, 1, 2, 3, 4, 5, 6]`. September to December belong to the year before the harvest year, which the scripts handle automatically. |
| `GRID_COLS`, `GRID_ROWS` | Number of tiles the parcels are split into (default 2 × 2 = 4).  |

The first run opens a browser window for the Google login. The login also runs
when you only ask for `--help`.

---

## Step 4: Run the Sentinel scripts

```bash
python s1_monthly_fetcher_v3_quadrants.py --years 2020 2021 2022 2023 2024
python s2_monthly_fetcher_v3_quadrants.py --years 2020 2021 2022 2023 2024
```

`--years` are harvest years, and each needs its asset. For every
year and month, the parcels are split into tiles  and one CSV is
exported per month and tile.

| Option | Meaning |
|---|---|
| `--mode blocking` (default) | Submits all tiles of a month, waits until they finish, then moves to the next month. Easy to follow, slower. |
| `--mode submit` | Submits everything at once and exits. Follow progress in the Earth Engine Task Manager (code.earthengine.google.com/tasks). Faster overall. |

Things to know:

- **Run after the month has ended.** Earth Engine exports the images that exist
  at that moment. A month exported on the 15th contains only 15 days, and
  re-running later does not refresh it (see Resume).
- **The current season.** For the season in progress, the scripts also submit
  the months that have not happened yet. Those have no images, so their tasks
  may fail or give empty files. This is harmless: ignore the failures and
  delete any empty (0-byte) CSV before using the data.
- **Resume.** Tasks that already succeeded in Earth Engine are skipped on a new
  run, matched **by name only**. Earth Engine does not check that the file is
  still in Drive. If you delete a file, or change `DRIVE_FOLDER`, that month
  is not exported again. To re-export, use a new `DRIVE_FOLDER`, or rename the
  tasks by changing the grid. Remove old copies first to avoid
  duplicates.
- **Duplicates in Drive.** Drive does not overwrite files. If an export runs
  twice you get two files with the same name. Remove the duplicates before
  using the data.
- **Logs** are appended to `logs/s1_run_<SITE>.log` and
  `logs/s2_run_<SITE>.log`.
- **Duration.** A full season of a large site can take hours. Quotas depend on
  your Earth Engine account type (see the Earth Engine documentation).

###  Tiles (`GRID_COLS` × `GRID_ROWS`) ---

A big site cannot be exported in one task, so the parcels are split into a grid
of `GRID_COLS × GRID_ROWS` tiles over the site's bounding box, and each tile
becomes one CSV per month.

- Each parcel goes to **exactly one tile**, chosen by its centroid. There are
  no duplicates between tiles.
- Tiles are equal in area, not in parcel count. A dense town or a sparse zone
  gives uneven tiles.
- **Choosing the size.** The default 2 × 2 handled about 250,000 parcels (about
  60,000 per tile). For a larger site raise it (for example 3 × 3). If a task
  fails with a memory or timeout error, increase the grid. Limits depend on
  Earth Engine, which I did not verify here.
- More tiles means more tasks per month. In `blocking` mode all tiles of a
  month run at the same time, so a very large grid may queue.
- **Use the same grid for every year and for both sensors.** Do not change it
  halfway through a dataset.
- The grid size is part of the export name (`..._part2of4`). Changing the grid
  therefore never clashes with older names, but files exported under another
  grid must not be mixed with the new ones (a parcel would appear in two
  tiles). Start a new `DRIVE_FOLDER` after any change.

---

## Step 5: Download and place the files

1. In Google Drive, open `RPG_S1_Exports_<SITE>` and `RPG_S2_BANDS_Exports_<SITE>`
   (or the `DRIVE_FOLDER` names you chose).
2. Download all CSVs. For large folders, Drive for desktop is more reliable than
   a zip download.
3. Put them, without renaming, into:
```
   Data/S1_extract_<SITE>/     S1_<SITE>_<year>_<YYYY-MM>_part<n>of<N>.csv
   Data/S2_extract_<SITE>/     S2_<SITE>_<year>_<YYYY-MM>_part<n>of<N>.csv
```
   For example `S2_Bapaume_2024_2023-09_part1of4.csv`. The folder names must be
   exactly `S1_extract_<SITE>` and `S2_extract_<SITE>`.

The classifier reads the year and month from the file name and combines the
parts itself. Keep a copy of the raw downloads: the exports cannot be
reproduced exactly later, because the Sentinel collections can be updated.

---

## Check the data

- **File count.** For each sensor: `GRID_COLS × GRID_ROWS × months × years`
  (for example 4 × 10 × 5 = 200). The season in progress has fewer files. A few
  missing files elsewhere can be normal (a tile without images in a month).
  Many missing files are not.
- **S1 and S2 files in the right folder.** An S2 file must start with `S2_`.
  Check that no `S1_` file is in the S2 folder, or the other way round.
- **No duplicates** such as `...part1of4 (1).csv`, and no 0-byte files.
- **Columns and values:**

```python
import glob, pandas as pd
for s in ("S1", "S2"):
    files = sorted(glob.glob(f"Data/{s}_extract_Bapaume/{s}_*.csv"))
    df = pd.read_csv(files[0])
    print(s, len(files), "files |", list(df.columns))
    print(df.head(3), "\n")
```

For Sentinel-1, `VV` and `VH` should be small positive numbers (linear scale,
for example 0.01-0.5), not negative dB values.

---

## What the files contain

Each row is **one parcel on one acquisition date**: the mean of the pixels
inside the parcel.

**Sentinel-1** (`COPERNICUS/S1_GRD`, IW mode, VV and VH both present)

| Column | Meaning |
|---|---|
| `id_parcel`, `code_cultu`, `code_group` | Taken from the asset |
| `date` | Acquisition date, `YYYY-MM-DD` |
| `VV`, `VH` | Backscatter, converted from dB to **linear** before averaging |
| `angle` | Incidence angle (degrees) |
| `orbit_direction` | `ASCENDING` or `DESCENDING` |
| `orbit_number` | Relative orbit number |

**Sentinel-2** (`COPERNICUS/S2_SR_HARMONIZED`)

| Column | Meaning |
|---|---|
| `id_parcel`, `code_cultu`, `code_group`, `date` | As above |
| `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12` | Surface reflectance, **scaled by 10000** (1500 means 0.15) |
| `MSK_CLDPRB` | Mean cloud probability (0-100) of the pixels that were kept |

**Cleaning applied during export**

- *Sentinel-2.* Scenes with 50% or more cloudy pixels are dropped. Pixels are
  masked when the cloud probability is 50 or more, the snow probability is 10
  or more, or the scene classification marks cloud shadow, dark area or cirrus.
  A parcel with no valid pixel on a date produces no row.
- *Sentinel-1.* No cloud issue. Values are used as delivered by Earth Engine
  (calibrated and terrain-corrected), without speckle filtering or
  border-noise masking. The classifier removes outliers with its own spike
  filter. Another classifier must do its own cleaning.

**Resolution.** Sentinel-2 bands at 20 m (`B5 B6 B7 B8A B11 B12`) are sampled
at 10 m (`SCALE_M`), so these pixels are resampled.

**Cloud tightening later.** Because masking uses 50% cloud probability,
`MSK_CLDPRB` can only be used to **tighten** the filter after download, never to
loosen it. The classifier does this with `CLOUD_THRESHOLD` (default 1) without a
new download.

**Duplicate rows.** A parcel that straddles a tile or orbit boundary of the
satellite can appear twice for the same date. The classifier takes the median
per parcel and date. If you use the files elsewhere, do the same:

```python
key = ["id_parcel", "code_cultu", "code_group", "date"]   # S1: add "orbit_direction", "orbit_number"
df = df.groupby(key, as_index=False)[value_columns].mean()
```

---

## What you can change

The scripts are not tied to this classifier. The right-hand column says
whether the classifier still works after a change.

| What | Where | Classifier compatibility |
|---|---|---|
| **Season** | `SEASON_MONTHS` (both scripts) | The classifier accepts only September to June. Another calendar (for example March-December) works for other uses, and the year/month pattern in the file names is kept. |
| **Years** | `--years` | Any years that have an asset. |
| **S2 bands** | `STAT_BANDS` (S2; `OUTPUT_COLS` follows automatically) | The classifier needs `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12` and `MSK_CLDPRB`. Extra columns are ignored. Keep `B2`: the script uses it to drop empty rows. |
| **S1 columns** | `STAT_BANDS` **and** `OUTPUT_COLS` (S1; edit both) | The classifier needs `VV`, `VH`, `angle`, `orbit_direction`. |
| **S2 cloud and snow** | `MAX_CLOUD_PERCENTAGE`, `MAX_CLOUD_PROB`, `MAX_SNOW_PROB` | Stricter gives fewer but cleaner observations. Looser gives room for a higher `CLOUD_THRESHOLD` in the classifier. |
| **S2 scene classes masked** | `_process_image` (S2) | Free to change. |
| **Statistic per parcel** | `ee.Reducer.mean()` in `zonal_stats_monthly` | Keep a single reducer so column names equal band names. |
| **Pixel size** | `SCALE_M` (10) | Free. 10 m is the native resolution of the main bands. |
| **Tiles** | `GRID_COLS`, `GRID_ROWS` (same in both scripts) | Free. `N_PARTS` follows automatically. |
| **Parcel size filter** | `MIN_AREA`, `BUFFER_DIST` in `rpg_fetcher.py` | Free, but results will differ from the project report. |
| **Parcel source** | The asset | Any FeatureCollection with `id_parcel`, `code_cultu`, `code_group` works. For other properties, change `OUTPUT_COLS`. |
| **Drive, logs, polling** | `DRIVE_FOLDER`, `LOG_FILE`, `POLL_INTERVAL` | Free. |

Changes that need more work: adding indices (compute them in `_process_image`),
other collections, or per-pixel instead of per-parcel output.

---

## Using the data for another classifier

- The files are plain CSV with a stable layout, so any Python or R
  workflow can read them.
- Remove duplicate rows first.
- Remember the units: S1 linear, S2 reflectance × 10000.
- For a different crop calendar, change `SEASON_MONTHS`. The calendar year of
  each month is derived automatically when the season crosses New Year.

---

## Known limitations

- RPG outlines are French. Other countries need another parcel source (see
  "Parcel source").
- Outlines come from the declaration year. Reusing outlines for the predict
  year is an approximation.
- Edge pixels are included in the parcel mean. The inward buffer only selects
  which parcels are kept.
- A month exported before it ends is incomplete, and resume does not refresh it.
- Exports are limited by your Earth Engine and Drive quotas.
- Very large sites need a bigger grid.

---

## Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `[year] N not available - using M instead` | RPG for year N is not published. |
| A year is missing from the final `[done]` line | It was not downloaded. Check the log above and run again. |
| HTTP error from the RPG fetcher at the first request | The AOI geometry is too detailed. Simplify it. |
| `Asset introuvable` | Asset path, site name or year does not match |
| Export fails on a column name | The asset lacks `id_parcel`, `code_cultu` or `code_group`. Check the asset properties |
| Task fails with a memory or timeout error | A tile is too heavy. Increase `GRID_COLS` / `GRID_ROWS` in both scripts and use a new `DRIVE_FOLDER`  |
| `Google Drive is full` | Free space, then run again. Finished months are skipped. |
| Tasks for future months fail | Expected for a season in progress |
| A month is skipped but its file is not in Drive | Resume matches task names only. Use a new `DRIVE_FOLDER`  |
| Files named `... (1).csv` | Duplicate exports. Keep one of each. |
| Classifier: `unexpected month 'm07'` | `SEASON_MONTHS` includes months outside September-June. |
| Classifier: `no files found for year ...` | File names do not follow `S1_<SITE>_<year>_<YYYY-MM>...csv`. |
| Classifier: `EmptyDataError` | A 0-byte CSV is in the folder. Delete it. |