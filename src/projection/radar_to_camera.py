# src/projection/radar_to_camera.py
from typing import Tuple
import numpy as np


def project_radar_to_camera(points_radar: np.ndarray,
                            K: np.ndarray,
                            T_cam_radar: np.ndarray,
                            img_width: int,
                            img_height: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points_radar: (N, 3) array of XYZ in radar frame.
        K: (3, 3) camera intrinsic matrix.
        T_cam_radar: (4, 4) transform radar → camera.
        img_width, img_height: for filtering points inside the image.

    Returns:
        uv: (M, 2) pixel coordinates inside the image.
        mask: (N,) boolean mask of which original points are valid & visible.
    """

    N = points_radar.shape[0]

    # Homogeneous radar points
    pts_h = np.ones((N, 4), dtype=float)
    pts_h[:, :3] = points_radar

    # Radar → camera
    pts_cam_h = (T_cam_radar @ pts_h.T).T          # (N, 4)
    pts_cam = pts_cam_h[:, :3]                    # (Xc, Yc, Zc)

    Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # Only keep points in front of the camera
    in_front = Zc > 0.1

    Xc, Yc, Zc = Xc[in_front], Yc[in_front], Zc[in_front]

    # Camera intrinsics: project to pixels
    # [u, v, w]^T = K @ [Xc/Zc, Yc/Zc, 1]^T
    x_norm = Xc / Zc
    y_norm = Yc / Zc

    ones = np.ones_like(x_norm)
    pts_norm = np.stack([x_norm, y_norm, ones], axis=0)   # (3, M)

    uvw = K @ pts_norm
    u = uvw[0, :] / uvw[2, :]
    v = uvw[1, :] / uvw[2, :]

    # Filter to those inside the image bounds
    in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

    u_img = u[in_img]
    v_img = v[in_img]

    uv = np.stack([u_img, v_img], axis=1)  # (M, 2)

    # Build full mask w.r.t original N points
    mask = np.zeros(N, dtype=bool)
    idx_front = np.nonzero(in_front)[0]
    mask[idx_front[in_img]] = True

    return uv, mask
