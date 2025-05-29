"""Test for the preprocess."""

import pytest

from bidipose.preprocess.sampler import StereoCameraSampler
from bidipose.preprocess.utils import get_kpts_from_cdf


@pytest.fixture
def keypoints():
    """Fixture to load keypoints from a CDF file."""
    cdf_file = "/fsws1/share/database/3d_human_pose_datasets/Human36M/S1/Poses_D3_Positions/Directions.cdf"
    kpts3d_world = get_kpts_from_cdf(cdf_file)
    return kpts3d_world


def test_get_kpts_from_cdf(keypoints):
    """Test the function to get keypoints from a CDF file."""
    assert keypoints.shape[1] == 17
    assert keypoints.shape[2] == 3


def test_sample_stereo_cameras_and_targets(keypoints):
    """Test the function to sample stereo cameras and targets."""
    kpts3d_world = keypoints
    sampler = StereoCameraSampler()
    cameras, targets = sampler.sample_camera_and_target(kpts3d_world)
    assert cameras.shape == (2, 3)
    assert targets.shape == (2, 3)
