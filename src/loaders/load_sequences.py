import json
from pathlib import Path
from typing import Dict, Any

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_scenes_and_sensors(dataset_root: Path):
    """
    Loads scenes.json and sensors.json from RadarScenes directory
    """
    scenes_path = dataset_root / "sequences.json"
    sensors_path = dataset_root / "sensors.json"

    scenes = load_json(scenes_path)
    sensors = load_json(sensors_path)

    return scenes, sensors
