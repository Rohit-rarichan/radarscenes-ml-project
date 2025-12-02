from pathlib import Path
from typing import Optional
import json
from PIL import Image


def load_any_camera_image(dataset_root: Path, seq_id: str, idx: int = 0):
    """
    Loads the idx-th camera image for the sequence using scenes.json timestamps.
    """
    scene_json = dataset_root / seq_id / "scenes.json"

    with open(scene_json, "r") as f:
        meta = json.load(f)

    timestamps = sorted(meta["scenes"].keys(), key=lambda x: int(x))

    # pick the idx-th timestamp
    ts = timestamps[idx]

    # get file name
    img_name = meta["scenes"][ts]["image_name"]

    img_path = dataset_root / seq_id / "camera" / img_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image does not exist: {img_path}")

    return Image.open(img_path).convert("RGB"), int(ts)


def load_camera_image(img_path: Path) -> Optional[Image.Image]:
    """
    Loads a camera image using PIL.
    """
    if not img_path.exists():
        raise FileNotFoundError(f"Camera image not found: {img_path}")

    return Image.open(img_path).convert("RGB")


def get_camera_image_path(dataset_root: Path, sequence_id: str, timestamp: int) -> Path:
    """
    Given a radar timestamp, find the matching camera image.
    Uses scenes.json: `image_name` field.
    """
    scene_json = dataset_root / sequence_id / "scenes.json"

    if not scene_json.exists():
        raise FileNotFoundError(f"scenes.json missing: {scene_json}")

    # Load JSON
    with open(scene_json, "r") as f:
        meta = json.load(f)

    scenes = meta["scenes"]

    if str(timestamp) not in scenes:
        raise KeyError(f"Timestamp {timestamp} not found in scenes.json")

    img_name = scenes[str(timestamp)]["image_name"]

    return dataset_root / sequence_id / "camera" / img_name

