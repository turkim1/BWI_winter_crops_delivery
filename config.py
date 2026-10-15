"""
config.py — All constants, paths, and hyperparameters.
Edit this file to change runtime behaviour.
"""

from datetime import datetime
from pathlib import Path

from indices import S2_INDEX_COLS, S2_BAND_COLS as _S2_BAND_COLS, S2_BAND_10M_COLS as _S2_BAND_10M_COLS, S2_BAND_AND_INDEX_COLS as _S2_BAND_AND_INDEX_COLS

# ################################################################################
# -- SHARED - used by both ML and IT ---------------------------------------------------------------
# ################################################################################

MODEL_NAME = "xgboost"  # "xgboost" or"inceptiontime"
SITE = "Bapaume"  # "Bapaume", "Toulouse"

# SENSOR SELECTION
# # Each entry in SENSOR_MODES produces one trained model + one run directory.
# Edit this list to match the data you actually have:
#   - S1 only  : SENSOR_MODES = ["S1", "S1_ASC", "S1_DESC", "S1_SPLIT_ORBIT"]
#   - S2 only  : SENSOR_MODES = ["S2"]
#   - S1 + S2  : SENSOR_MODES = ["S1", "S2", "both", "S1_ASC", "S1_DESC", "S1_SPLIT_ORBIT", "both_split_orbit"]

# Only the data needed by the listed modes is loaded:
#   S1* modes need the S1 folder, "S2" needs the S2 folder, "both*" need both.
# "S1_ASC"/"S1_DESC"/"S1_SPLIT_ORBIT"/"both_split_orbit" need an 'orbit_direction'
# column in the raw S1 files.
# -------------------------------------------------------------
SENSOR_MODES: list[str] = [
"both_split_orbit",  # S1 + S2, split ascending/descending orbits
]

# PATHS
# -------------------------------------------------------------
_BASE = Path(__file__).resolve().parent

S1_DIR    = _BASE / "Data" / f"S1_extract_{SITE}"
S2_DIR    = _BASE / "Data" / f"S2_extract_{SITE}"
CACHE_DIR = _BASE / "cache_"

# CLASS DEFINITIONS
# -------------------------------------------------------------
WINTER_CROPS = {"BTH", "BDH", "ORH", "AVH", "CHA", "CHH", "CHS", "CHT", "SGH", "TTH", "CAH"} # code_cultu wheat and related winter cereal
CZH_CROPS    = {"CZH"} # code_cultu rapeseed
CLASS_NAMES  = {0: "Winter crops", 1: "CZH", 2: "Other vegetation"} 

# code_group values to exclude (vines, orchards, miscellaneous, non cultivated)
EXCLUDED_CODE_GROUPS = {20, 21, 28, 11} # vines, orchards, miscellaneous, non cultivated

MONTH_ORDER = ["m09","m10","m11","m12","m01","m02","m03","m04","m05","m06"]


# -------------------------------------------------------------
# S2 FEATURE MODE
# "indices" : compute NDVI, NDMI, NDRE, PSRI, NDYI 
# "bands"   : keep raw bands B2…B12  
# "bands_10m" : keep only 10m bands B2, B3, B4, B8
# "bands_10m+indices" : keep 10m bands + all indices
# -------------------------------------------------------------
S2_MODE = "bands"

S2_FEATURE_COLS: list[str] = (
    S2_INDEX_COLS               if S2_MODE == "indices"           else
    _S2_BAND_AND_INDEX_COLS     if S2_MODE == "bands_10m+indices" else
    _S2_BAND_COLS               if S2_MODE == "bands"             else
    _S2_BAND_10M_COLS           # bands_10m
)

# PREPROCESSING
# -------------------------------------------------------------
CLOUD_THRESHOLD         = 1
REF_ANGLE               = None      # None --> computed from training data

S1_NUMERIC_COLS = ["VV", "VH", "angle"]
AGG_FUNCS       = ["median"]        # can add "mean", "std", "skew" ...

# Outlier spike filter
S1_SPIKE_FILTER         = True       # set False to disable
S1_SPIKE_THRESHOLD_DB   = 5.0

S2_NDVI_DIP_FILTER     = True   # set False to disable
S2_NDVI_DIP_THRESHOLD  = 0.4

# NDVI post filter
NDVI_POSTFILTER = True   # after predicting, split "Other vegetation" into cultivated / non_cultivated (needs S2 files)
NDVI_FILTER_START_MONTH = "m02"   # e.g. "m02"
NDVI_FILTER_END_MONTH = "m05"     # e.g. "m05"
NDVI_FILTER_THRESHOLD   =  0.45   # e.g. 0.45


# ################################################################################
# -- ML XGBoost ---------------------------------------------------------------
# ################################################################################

# MODEL
# -------------------------------------------------------------
RANDOM_SEED      = 5

# XGB #
BEST_PARAMS_XGB = {
    "n_estimators":      100,
    "max_depth":         6,       
    "learning_rate":     0.3,     
    "subsample":         1.0,     
    "colsample_bytree":  1.0,     
    "min_child_weight":  1,    
    "gamma":             0.0,    
    "reg_alpha":         0.0,     
    "reg_lambda":        1.0,     
    "objective":         "multi:softprob",
    "tree_method":       "hist"
}

# ################################################################################
# -- InceptionTime ---------------------------------------------------------------
# ################################################################################

# Harmonic resampling (shared by S1 and S2 dataloaders)
IT_HARMONIC_STEP_DAYS    = 6      # grid spacing --> T = ⌊303/6⌋+1 = 51
IT_HARMONIC_N_PER_SEASON = 4       # harmonic order

# Architecture
IT_NB_FILTERS      = 32
IT_DEPTH           = 6
IT_KERNEL_SIZES    = [5, 9, 15] # before [3, 7, 15]
IT_BOTTLENECK_DIM  = 16

# Training
IT_N_EPOCHS    = 100
IT_BATCH_SIZE  = 16 #  changed to match saiedeh paper
IT_LR          = 1e-5 # before 1e-4 but changed to match saiedeh paper
IT_WEIGHT_DECAY= 1e-6 #before 1e-4 but changed to match saiedeh paper

IT_EARLY_STOPPING_PATIENCE  = 10  # if set to IT_N_EPOCHS this will prevent early stopping and to have the full epochs # before it was 10   # epochs sans amélioration avant arrêt
IT_EARLY_STOPPING_MIN_DELTA = 1e-4  # amélioration minimale considérée
IT_LR_SCHEDULER             = True  # ReduceLROnPlateau
IT_LR_PATIENCE              = 5     # epochs avant de réduire le LR
IT_LR_FACTOR                = 0.5   # facteur de réduction
IT_LR_MIN                   = 1e-6  # LR plancher


# ################################################################################
# -- Output Path---------------------------------------------------------------
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = _BASE / "Runs" / f"{_timestamp}_{MODEL_NAME}_{S1_DIR.name.rstrip('_').split('_')[-1]}_{S2_MODE}_{AGG_FUNCS}"