"""Test for pose visualization."""

import numpy as np
import pytest

from bidipose.datasets.dataset import StereoCameraDataset
from bidipose.visualize.animation_3d import vis_pose3d


@pytest.fixture
def stereo_camera_dataset():
    """Fixture for StereoCameraDataset."""
    data_root = "/fsws1/share/database/3d_human_pose_datasets/Human36M/"
    return StereoCameraDataset(data_root=data_root, data_name="H36M", split="test")


def test_vis_pose3d(stereo_camera_dataset):
    """Test the 3D pose visualization."""
    # Get a sample from the dataset
    pose, quat, trans = stereo_camera_dataset[1]
    pose = np.array(pose)
    quat = np.array(quat).squeeze(-1)
    trans = np.array(trans).squeeze(-1)
    noise = 0.005
    pred_pose = pose + noise
    pred_pose[:, 0, :] = pose[:, 0, :]
    anim = vis_pose3d(
        pred_pose=pred_pose,
        pred_quat=quat,
        pred_trans=trans,
        gt_pose=pose,
        gt_quat=quat,
        gt_trans=trans,
    )
    anim.save("tests/outputs/test_3d_pose.gif", writer="ffmpeg", fps=30)
