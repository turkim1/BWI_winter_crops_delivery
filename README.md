# Winter Crops Classifier

Classifies agricultural parcels into **Winter crops**, **CZH (rapeseed)** and
**Other vegetation** from Sentinel-1 (radar) and Sentinel-2 (optical) time
series. It learns from past labeled seasons and predicts a newer season, even
if that season is not finished.

The repository has a flat structure, but the workflow has two parts.

## Start with the READMEs

| Part | README | What it covers |
|---|---|---|
| **1. Data preparation** | [`readme_data_fetch.md`](readme_data_fetch.md) | Get RPG parcel outlines and extract Sentinel-1/2 data per parcel with Google Earth Engine. Reusable for other classifiers. |
| **2. Classification** | [`readme_classifier.md`](readme_classifier.md) | Run the classifiers (XGBoost, InceptionTime), read the output CSV, reproduce the report experiments. |

The `.html` versions are the same content, for reading in a browser.

## Quick start with the sample data

1. Download the sample data from **<link>** and unzip `Data.7z` so that the
   `Data/` folder sits next to `config.py`.
2. Install (Python 3.11 or 3.12):
```bash
   python -m venv venv
   venv\Scripts\activate                 # Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements_it.txt    # only for InceptionTime
```
3. Check `SITE`, `MODEL_NAME` and `SENSOR_MODES` in `config.py`.
4. Run:
```bash
   python predict.py
```
   The result is written to `Runs/<timestamp>_.../predictions_<year>_<model>_<mode>.csv`.

## Scripts

**Data preparation**

| File | Role |
|---|---|
| `rpg_fetcher.py` | Downloads RPG parcel outlines for an area and several years. |
| `s1_monthly_fetcher_tiled.py` | Exports Sentinel-1 values per parcel and date (Earth Engine to Drive). |
| `s2_monthly_fetcher_tiled.py` | Exports Sentinel-2 values per parcel and date (Earth Engine to Drive). |

**Classification**

| File | Role |
|---|---|
| `predict.py` | **Main entry point.** Trains on all labeled years and predicts the newest year. |
| `ml_train.py` | Accuracy evaluation for XGBoost (newest year must be labeled). |
| `it_train.py` | Accuracy evaluation for InceptionTime (newest year must be labeled). |
| `config.py` | All settings. |
| `common_preproc.py`, `utils.py`, `indices.py` | Shared cleaning and helpers. |
| `ml_*.py`, `ndvi_postfilter.py` | XGBoost branch and the NDVI split of "Other vegetation". |
| `it_*.py` | InceptionTime branch. |
| `plots.py` | Diagnostic plots used by the evaluation scripts. |

A file-by-file description is in the appendix of `readme_classifier.md`.

## Status

The documentation could be updated.