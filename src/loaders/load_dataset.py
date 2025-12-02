from pathlib import Path
from loaders.load_radar import load_radar_h5
from loaders.load_camera import load_camera_image, get_camera_image_path
from loaders.load_sequences import load_scenes_and_sensors

import pandas as pd


def list_sequence_ids(dataset_root: Path):
    """
    Returns sorted list of all sequence IDs (e.g., ['0001','0002',...]).
    """
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])


def load_radar_for_sequence(dataset_root: Path, seq_id: str):
    seq_path = dataset_root / seq_id

    # Radar file can be located in two different formats depending on dataset version
    radar_h5_path = seq_path / "radar" / "radar_data.h5"
    if not radar_h5_path.exists():
        radar_h5_path = seq_path / "radar_data.h5"

    return load_radar_h5(radar_h5_path)

def load_camera_for_sequence(dataset_root: Path, seq_id: str, frame_idx: int):
    img_path = get_camera_image_path(dataset_root, seq_id, frame_idx)
    return load_camera_image(img_path)
