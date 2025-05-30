"""Test for the pose models."""

from functools import partial

import torch
import torch.nn as nn

from bidipose.models.MotionAGFormer.model import MotionAGFormer
from bidipose.models.MotionBERT.model import DSTformer


def test_motion_agformer():
    """Test the MotionAGFormer model."""
    model = MotionAGFormer(n_layers=2, dim_feat=32, dim_in=4, n_frames=88)
    model.eval()

    x = torch.randn(1, 81, 17, 4)
    quat = torch.randn(1, 4, 1)
    trans = torch.randn(1, 3, 1)
    with torch.no_grad():
        pred_pose, pred_quat, pred_trans = model(x, quat, trans)
    assert pred_pose.shape == (1, 81, 17, 4)
    assert pred_quat.shape == (1, 4, 1)
    assert pred_trans.shape == (1, 3, 1)

    quat_norm = torch.norm(pred_quat, dim=1, keepdim=True)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6), f"Quaternion norm: {quat_norm}"
    trans_norm = torch.norm(pred_trans, dim=1, keepdim=True)
    assert torch.allclose(trans_norm, torch.ones_like(trans_norm), atol=1e-6), f"Translation norm: {trans_norm}"


def test_dstformer():
    """Test the DSTformer model."""
    model = DSTformer(dim_feat=512, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    x = torch.randn(1, 81, 17, 3)
    out = model(x)
    assert out.shape == (1, 81, 17, 3)
