"""Test for the pose models."""

from functools import partial

import torch
import torch.nn as nn

from bidipose.models.MotionAGFormer.model import MotionAGFormer
from bidipose.models.MotionBERT.model import DSTformer


def test_motion_agformer():
    """Test the MotionAGFormer model."""
    model = MotionAGFormer(n_layers=2, dim_in=3, dim_feat=32, n_frames=81)
    assert model is not None

    x = torch.randn(1, 81, 17, 3)
    out = model(x)
    assert out.shape == (1, 81, 17, 3)


def test_dstformer():
    """Test the DSTformer model."""
    model = DSTformer(dim_feat=512, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    assert model is not None

    x = torch.randn(1, 81, 17, 3)
    out = model(x)
    assert out.shape == (1, 81, 17, 3)
