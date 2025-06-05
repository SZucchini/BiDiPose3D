"""Preprocess utils for Bidipose."""

import numpy as np
import spacepy.pycdf


def hml3d_to_h36m(kpts_hml3d: np.ndarray) -> np.ndarray:
    """Convert HML3D keypoints to H36M keypoints.

    Args:
        kpts_hml3d (np.ndarray): HML3D keypoints.

    Returns:
        kpts_h36m (np.ndarray): H36M keypoints.

    """
    h36m_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
    hml3d_idx = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]
    kpts_h36m = np.zeros((kpts_hml3d.shape[0], 17, 3))
    kpts_h36m[:, h36m_idx, :] = kpts_hml3d[:, hml3d_idx, :]
    kpts_h36m[:, 8, :] = (kpts_hml3d[:, 9, :] + kpts_hml3d[:, 12, :]) / 2
    return kpts_h36m


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


def get_kpts_from_npy(npy_file: str) -> np.ndarray:
    """Get keypoints from a numpy file of HumanML3D.

    Args:
        npy_file (str): Path to the numpy file.

    Returns:
        kpts (np.ndarray): 3D Keypoints. The shape is (T, J, 3). Z is human height.

    """
    kpts = np.load(npy_file)
    kpts = hml3d_to_h36m(kpts)
    kpts = kpts[:, :, [0, 2, 1]]
    return kpts


def create_frame_indices(
    sequence_length: int,
    target_length: int,
    replay: bool = False,
    random_sampling: bool = True,
) -> np.ndarray:
    """Create indices for frame sampling with specified length.

    Args:
        sequence_length (int): Length of the original sequence.
        target_length (int): Desired length of the sampled sequence.
        replay (bool): If True, creates continuous frame indices that can be replayed.
        random_sampling (bool): If True, applies random sampling within intervals.

    Returns:
        indices (np.ndarray): Array of frame indices for sampling.

    """
    rng = np.random.default_rng()
    if replay:
        if sequence_length > target_length:
            start_idx = rng.integers(sequence_length - target_length)
            return np.array(range(start_idx, start_idx + target_length))
        else:
            return np.array(range(target_length)) % sequence_length
    else:
        if random_sampling:
            sample_points = np.linspace(0, sequence_length, num=target_length, endpoint=False)
            if sequence_length < target_length:
                floor_values = np.floor(sample_points)
                ceil_values = np.ceil(sample_points)
                random_choice = rng.integers(2, size=sample_points.shape)
                indices = np.sort(random_choice * floor_values + (1 - random_choice) * ceil_values)
            else:
                interval = sample_points[1] - sample_points[0]
                indices = rng.random(sample_points.shape) * interval + sample_points
            indices = np.clip(indices, a_min=0, a_max=sequence_length - 1).astype(np.uint32)
        else:
            indices = np.linspace(0, sequence_length, num=target_length, endpoint=False, dtype=int)
        return indices


def split_clips(
    total_frames: int,
    clip_length: int = 81,
    stride: int = 27,
) -> list[list[int]]:
    """Split a sequence into overlapping clips of specified length.

    Args:
        total_frames (int): Total number of frames in the sequence.
        clip_length (int): Number of frames in each clip.
        stride (int): Number of frames to move forward for each new clip.

    Returns:
        clips (list[list[int]]): List of frame indices for each clip.

    """
    assert stride > 0, "Stride must be greater than 0."

    clips = []
    clip_start = 0

    while clip_start < total_frames:
        if total_frames - clip_start < clip_length // 2:
            break
        if clip_start + clip_length > total_frames:
            clip_indices = create_frame_indices(total_frames - clip_start, clip_length) + clip_start
            clips.append(clip_indices)
            break
        else:
            clips.append(range(clip_start, clip_start + clip_length))
            clip_start += stride

    return clips
