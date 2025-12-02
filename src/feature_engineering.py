from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------------------
# Utility: Feature Engineering for Radar Data
# -----------------------------------------

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived spatial & physical radar features:
    - x, y Cartesian coordinates
    - distance (range magnitude)
    - angle_deg (human-readable azimuth)
    - speed_abs (absolute velocity)
    - power_norm (normalized RCS)
    """

    # Convert radar polar → Cartesian
    df["x"] = df["range_sc"] * np.cos(df["azimuth_sc"])
    df["y"] = df["range_sc"] * np.sin(df["azimuth_sc"])

    # Derived features
    df["distance"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["angle_deg"] = np.degrees(df["azimuth_sc"])
    df["speed_abs"] = df["vr"].abs()

    # Normalize RCS into a [0,1] scale
    rcs_min, rcs_max = df["rcs"].min(), df["rcs"].max()
    df["power_norm"] = (df["rcs"] - rcs_min) / (rcs_max - rcs_min)

    return df


def clean_radar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes radar outliers & missing data:
    - drops NaNs
    - removes extreme range/velocity/rcs noise
    """

    df = df.dropna()

    df = df[
        (df["range_sc"] > 0) &
        (df["range_sc"] < 120) &          # Max valid radar range
        (df["vr"].abs() < 150) &          # Remove crazy Doppler points
        (df["rcs"] > -40) &               # Low-end noise floor
        (df["rcs"] < 60)                  # Max physical return
    ]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main wrapper:
    - clean data
    - compute features
    """

    df = clean_radar(df)
    df = compute_basic_features(df)
    return df


def save_parquet(df: pd.DataFrame, out_path: Path):
    """
    Writes engineered DataFrame to Parquet for super-fast loading.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved engineered features → {out_path}")
