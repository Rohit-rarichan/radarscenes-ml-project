from pathlib import Path

# -------------------------------
# DATASET PATHS
# -------------------------------

# IMPORTANT: Fixes dataset path for your SSH server
DATASET_ROOT = Path("~/Datasets/RadarScenes/").expanduser()

# Pattern for locating radar_data.h5 inside sequence folders
H5_GLOB_PATTERN = "**/radar_data.h5"

PATHS = {
    "sequences": DATASET_ROOT / "sequences.json",
    "sensors": DATASET_ROOT / "sensors.json",
    "odom_in_h5": "odometry",      # keys inside the H5 file
    "radar_in_h5": "radar_data"
}

# -------------------------------
# CONSTANTS & SETTINGS
# -------------------------------

MOTION_THRESHOLD_MS = 0.1
RANGE_BINS_M = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AZIMUTH_BIN_DEG = 5

LABEL_MAP = {
    0: "car",                  #regular car
    1: "large_vehicle",
    2: "two_wheeler",
    3: "pedestrian",
    4: "pedestrian_group",     #group of people
    5: "car",                  #large cars like trucks
    6: "large_vehicle",
    7: "two_wheeler",
    8: "pedestrian",
    9: "pedestrian_group",      #large crowd
    10: "car"                  #any other 4 wheeler
}

STATIC_LABEL = 11

NEEDED_COLS_A456 = [
    "sensor_id", "range_sc", "azimuth_sc",
    "rcs", "vr", "vr_compensated", "label_id"
]

# -------------------------------
# OUTPUT DIRECTORIES
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"
TAB_DIR = OUT_DIR / "tables"

# Make sure directories exist
for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)
