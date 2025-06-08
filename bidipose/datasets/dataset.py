"""Dataset for stereo camera keypoints from Human3.6M or HML3D datasets."""

import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from bidipose.preprocess.camera_sampler import StereoCameraSampler
from bidipose.preprocess.utils import get_kpts_from_cdf, get_kpts_from_npy, split_clips


class StereoCameraDataset(Dataset):
    """Dataset for stereo camera keypoints from Human3.6M or HML3D datasets."""

    def __init__(self, data_root: str, data_name: str = "H36M", split: str = "train"):
        """Initialize the StereoCameraDataset.

        Args:
            data_root (str): Root directory of the dataset.
            data_name (str): Name of the dataset, either "H36M" or "HML3D".
            split (str): Split of the dataset, either "train" or "test". Only used for H36M dataset.

        """
        if data_name == "H36M":
            self.data_files = self._load_h36m_files(data_root, split)
        elif data_name == "HML3D":
            self.data_files = self._load_hml3d_files(data_root)
        else:
            raise ValueError(f"Unsupported dataset name: {data_name}")
        self.index = self._create_index()
        self.stereo_camera_sampler = StereoCameraSampler()

    def _load_h36m_files(self, data_root: str, split: str) -> list[str]:
        """Load files from the Human3.6M dataset.

        Args:
            data_root (str): Root directory of the Human3.6M dataset.
            split (str): Split of the dataset, either "train" or "test".

        Returns:
            result_files (list[str]): List of file paths containing keypoints in CDF format.

        """
        result_files = []
        if split == "train":
            target_subjects = ["S1", "S5", "S6", "S7", "S8"]
        elif split == "test":
            target_subjects = ["S9", "S11"]
        else:
            raise ValueError(f"Unsupported split: {split}")

        subject_dirs = glob.glob(f"{data_root}/*")
        for subject_dir in subject_dirs:
            subject = subject_dir.split("/")[-1]
            if subject in target_subjects:
                files = glob.glob(f"{subject_dir}/Poses_D3_Positions/*.cdf")
                result_files.extend(files)
        return result_files

    def _load_hml3d_files(self, data_root: str) -> list[str]:
        """Load files from the HML3D dataset.

        Args:
            data_root (str): Root directory of the HML3D dataset.

        Returns:
            result_files (list[str]): List of file paths containing keypoints in numpy format.

        """
        result_files = glob.glob(f"{data_root}/new_joints/*.npy")
        return result_files

    def _create_index(self) -> list[tuple[int, list[int]]]:
        """Create an index of clips from the dataset files.

        Returns:
            index (list[tuple[int, list[int]]]): A list of the file index and clip indices.

        """
        index = []
        for i, file in enumerate(self.data_files):
            if file.endswith(".cdf"):
                kpts = get_kpts_from_cdf(file)
            elif file.endswith(".npy"):
                kpts = np.load(file)

            clips = split_clips(len(kpts), clip_length=81, stride=81)
            for clip in clips:
                index.append((i, clip))
        return index

    def __len__(self) -> int:
        """Return the total number of clips in the dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            x (torch.Tensor): Normalized 2D keypoints from two views (shape: (T, J, 6)).
            quat (torch.Tensor): Quaternion representing the relative pose from cam1 to cam2 (shape: (4,)).
            trans (torch.Tensor): Translation vector representing the relative pose from cam1 to cam2 (shape: (3,)).

        """
        file_idx, clip_indices = self.index[idx]
        file_path = self.data_files[file_idx]
        if file_path.endswith(".cdf"):
            kpts = get_kpts_from_cdf(file_path)
        elif file_path.endswith(".npy"):
            kpts = get_kpts_from_npy(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        kpts_world = kpts[clip_indices]

        cams, tgts = self.stereo_camera_sampler.sample_camera_and_target(kpts_world)
        quat, trans = self.stereo_camera_sampler.get_relative_pose(cams, tgts)
        x1, x2 = self.stereo_camera_sampler.project_and_normalize(kpts_world, cams, tgts)
        x = np.concatenate([x1, x2], axis=-1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(quat, dtype=torch.float32),
            torch.tensor(trans, dtype=torch.float32),
        )
