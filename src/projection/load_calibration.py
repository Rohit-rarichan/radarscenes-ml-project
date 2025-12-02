# src/calibration/load_calibration.py
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from loaders.load_sequences import load_scenes_and_sensors


def load_sensor_calibration(dataset_root: Path,
                            radar_id: str = "radar_1",
                            camera_id: str = "camera") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        K (3x3): camera intrinsic matrix
        T_cam_radar (4x4): transform from radar frame → camera frame
    """

    scenes, sensors = load_scenes_and_sensors(dataset_root)

    cam_calib = sensors[camera_id]
    radar_calib = sensors[radar_id]

    # These keys/shape may differ a bit depending on the JSON,
    # but the idea is: they’re 4x4 extrinsics and 3x3 intrinsics.
    K = np.array(cam_calib["intrinsic"], dtype=float)

    # Assume extrinsics are given as sensor→vehicle transforms
    T_vehicle_radar = np.array(radar_calib["extrinsic"], dtype=float)
    T_vehicle_camera = np.array(cam_calib["extrinsic"], dtype=float)

    # We want: radar → camera
    # radar → vehicle → camera
    T_camera_vehicle = np.linalg.inv(T_vehicle_camera)
    T_cam_radar = T_camera_vehicle @ T_vehicle_radar

    return K, T_cam_radar
