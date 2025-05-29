"""Test for the pose models."""

import torch

from bidipose.models.MotionAGFormer.model import MotionAGFormer


def test_motion_agformer():
    """Test the MotionAGFormer model."""
    model = MotionAGFormer(n_layers=2, dim_in=3, dim_feat=32, n_frames=81)
    assert model is not None

    x = torch.randn(1, 81, 17, 3)
    out = model(x)
    assert out.shape == (1, 81, 17, 3)
