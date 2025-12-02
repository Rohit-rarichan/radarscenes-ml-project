from pathlib import Path
import h5py
import pandas as pd
from src.config import NEEDED_COLS_A456

def load_radar_h5(h5_path: Path) -> pd.DataFrame:
    """
    Loads a radar_data.h5 file and returns a pandas DataFrame.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Radar H5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "radar_data" not in f:
            raise KeyError("Missing 'radar_data' dataset in H5 file")

        dset = f["radar_data"]
        arr = dset[:]

        df = pd.DataFrame.from_records(arr)

        # subset columns
        df = df[[c for c in NEEDED_COLS_A456 if c in df.columns]]

    return df

def list_radar_timestamps(dataset_root: Path, seq_id: str):
    """
    Returns sorted unique radar timestamps from radar_data.h5.
    """
    radar_path = dataset_root / seq_id / "radar_data.h5"
    df = load_radar_h5(radar_path)
    return sorted(df["timestamp"].unique())


def load_radar_frame(dataset_root: Path, seq_id: str, timestamp: int):
    """
    Returns all radar points for a specific timestamp.
    """
    radar_path = dataset_root / seq_id / "radar_data.h5"
    df = load_radar_h5(radar_path)
    return df[df["timestamp"] == timestamp].reset_index(drop=True)