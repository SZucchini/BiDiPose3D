"""Preprocess utils for Bidipose."""

import numpy as np
import spacepy.pycdf


def get_kpts_from_cdf(cdf_file: str, kpts_indices: list[int] | None = None) -> np.ndarray:
    """Get keypoints from a CDF file.

    Args:
        cdf_file (str): Path to the CDF file.
        kpts_indices (list[int], optional): List of indices of keypoints to extract. Defaults to None.

    Returns:
        kpts (np.ndarray): 3D Keypoints. The shape is (T, J, 3). Z is human height.

    """
    if kpts_indices is None:  # General Human3.6M 17 keypoints Format
        kpts_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    cdf = spacepy.pycdf.CDF(cdf_file)
    data = cdf["Pose"][:]
    kpts = data.reshape(-1, 32, 3)
    kpts = kpts[:, kpts_indices, :]
    return kpts
