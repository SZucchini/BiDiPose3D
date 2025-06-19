"""Test for the pose models."""

from functools import partial

import pytest
import torch
import torch.nn as nn

from bidipose.models.MixSTE.model import MixSTE2
from bidipose.models.MotionAGFormer.model import MotionAGFormer
from bidipose.models.MotionBERT.model import DSTformer


@pytest.mark.parametrize(
    "t",
    [
        None,
        torch.randn(4),
    ],
)
def test_motion_agformer(t):
    """Test the MotionAGFormer model."""
    model = MotionAGFormer(n_layers=2, dim_feat=32, dim_in=6, n_frames=88)
    model.eval()

    x = torch.randn(4, 81, 17, 6)
    quat = torch.randn(4, 4)
    trans = torch.randn(4, 3)
    with torch.no_grad():
        pred_pose, pred_quat, pred_trans = model(x, quat, trans, t=t)
    assert pred_pose.shape == (4, 81, 17, 6)
    assert pred_quat.shape == (4, 4)
    assert pred_trans.shape == (4, 3)

    quat_norm = torch.norm(pred_quat, dim=1, keepdim=True)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6), f"Quaternion norm: {quat_norm}"
    trans_norm = torch.norm(pred_trans, dim=1, keepdim=True)
    assert torch.allclose(trans_norm, torch.ones_like(trans_norm), atol=1e-6), f"Translation norm: {trans_norm}"


@pytest.mark.parametrize("t", [None, torch.randn(4)])
def test_motionbert(t):
    """Test the MotionBERT model."""
    model = DSTformer(dim_feat=256, mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    x = torch.randn(4, 81, 17, 6)
    quat = torch.randn(4, 4)
    trans = torch.randn(4, 3)
    with torch.no_grad():
        pred_pose, pred_quat, pred_trans = model(x, quat, trans, t=t)
    assert pred_pose.shape == (4, 81, 17, 6)
    assert pred_quat.shape == (4, 4)
    assert pred_trans.shape == (4, 3)

    quat_norm = torch.norm(pred_quat, dim=1, keepdim=True)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6), f"Quaternion norm: {quat_norm}"
    trans_norm = torch.norm(pred_trans, dim=1, keepdim=True)
    assert torch.allclose(trans_norm, torch.ones_like(trans_norm), atol=1e-6), f"Translation norm: {trans_norm}"


@pytest.mark.parametrize(
    "batch_size,frames,joints,dim_in,dim_feat,depth,num_tokens",
    [
        (2, 81, 17, 6, 128, 3, 7),
        (4, 81, 17, 6, 256, 5, 7),
        (1, 81, 17, 6, 512, 2, 5),
    ],
)
def test_motionbert_different_configs(batch_size, frames, joints, dim_in, dim_feat, depth, num_tokens):
    """Test MotionBERT with different model configurations."""
    model = DSTformer(
        dim_in=dim_in,
        dim_feat=dim_feat,
        depth=depth,
        num_joints=joints,
        pose_frames=frames,
        num_tokens=num_tokens,
    )
    model.eval()

    x = torch.randn(batch_size, frames, joints, dim_in)
    quat = torch.randn(batch_size, 4)
    trans = torch.randn(batch_size, 3)
    t = torch.randn(batch_size)

    with torch.no_grad():
        pred_pose, pred_quat, pred_trans = model(x, quat, trans, t=t)

    assert pred_pose.shape == (batch_size, frames, joints, model.dim_out)
    assert pred_quat.shape == (batch_size, 4)
    assert pred_trans.shape == (batch_size, 3)

    quat_norm = torch.norm(pred_quat, dim=1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6)
    trans_norm = torch.norm(pred_trans, dim=1)
    assert torch.allclose(trans_norm, torch.ones_like(trans_norm), atol=1e-6)


def test_motionbert_return_representation():
    """Test MotionBERT with return_rep=True."""
    model = DSTformer(dim_feat=128, depth=2)
    model.eval()

    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)

    with torch.no_grad():
        rep = model(x, quat, trans, return_rep=True)

    assert rep.shape == (2, 81, 17, model.pre_logits.fc.out_features)
    assert isinstance(rep, torch.Tensor)


def test_motionbert_gradients():
    """Test that MotionBERT produces gradients during training."""
    model = DSTformer(dim_feat=64, depth=2)
    model.train()

    x = torch.randn(2, 81, 17, 6, requires_grad=True)
    quat = torch.randn(2, 4, requires_grad=True)
    trans = torch.randn(2, 3, requires_grad=True)
    t = torch.randn(2)

    pred_pose, pred_quat, pred_trans = model(x, quat, trans, t=t)
    loss = pred_pose.sum() + pred_quat.sum() + pred_trans.sum()
    loss.backward()

    assert x.grad is not None
    assert quat.grad is not None
    assert trans.grad is not None
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_motionbert_quaternion_properties():
    """Test MotionBERT quaternion output properties."""
    model = DSTformer(dim_feat=128, depth=2)
    model.eval()

    x = torch.randn(3, 81, 17, 6)
    quat = torch.randn(3, 4)
    trans = torch.randn(3, 3)

    with torch.no_grad():
        _, pred_quat, _ = model(x, quat, trans)

    # Test quaternion normalization
    quat_norm = torch.norm(pred_quat, dim=1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6)

    # Test w-component is always positive (canonical form)
    assert torch.all(pred_quat[:, 0] >= 0), f"W-components: {pred_quat[:, 0]}"


def test_motionbert_camera_tokens():
    """Test MotionBERT with different number of camera tokens."""
    for num_tokens in [3, 5, 7, 10]:
        model = DSTformer(dim_feat=64, depth=2, num_tokens=num_tokens)
        model.eval()

        x = torch.randn(2, 81, 17, 6)
        quat = torch.randn(2, 4)
        trans = torch.randn(2, 3)

        with torch.no_grad():
            pred_pose, pred_quat, pred_trans = model(x, quat, trans)

        assert pred_pose.shape == (2, 81, 17, 6)
        assert pred_quat.shape == (2, 4)
        assert pred_trans.shape == (2, 3)


@pytest.mark.parametrize("mode", ["stage_st", "stage_ts"])
def test_motionbert_attention_modes(mode):
    """Test MotionBERT Block with different attention modes."""
    from bidipose.models.MotionBERT.model import Block

    block = Block(
        dim=64,
        num_heads=4,
        st_mode=mode,
    )
    block.eval()

    batch_size, frames, joints, dim = 2, 81, 17, 64
    num_tokens = 7

    x = torch.randn(batch_size * frames, joints, dim)
    cam_params = torch.randn(batch_size, num_tokens, dim)

    with torch.no_grad():
        x_out, cam_params_out = block(x, cam_params)

    assert x_out.shape == (batch_size * frames, joints, dim)
    assert cam_params_out.shape == (batch_size, num_tokens, dim)


def test_motionbert_deterministic():
    """Test that MotionBERT produces deterministic outputs."""
    torch.manual_seed(42)
    model1 = DSTformer(dim_feat=64, depth=2)
    model1.eval()

    torch.manual_seed(42)
    model2 = DSTformer(dim_feat=64, depth=2)
    model2.eval()

    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    t = torch.randn(2)

    with torch.no_grad():
        out1 = model1(x, quat, trans, t=t)
        out2 = model2(x, quat, trans, t=t)

    for o1, o2 in zip(out1, out2, strict=True):
        assert torch.allclose(o1, o2, atol=1e-6)


def test_mixste():
    """Test the MixSTE model."""
    model = MixSTE2()

    x = torch.randn(4, 81, 17, 6)
    quat = torch.randn(4, 4)
    trans = torch.randn(4, 3)
    t = torch.randn(4)
    with torch.no_grad():
        pred_pose, pred_quat, pred_trans = model(x, quat, trans, t=t)
    assert pred_pose.shape == (4, 81, 17, 6)
    assert pred_quat.shape == (4, 4)
    assert pred_trans.shape == (4, 3)

    quat_norm = torch.norm(pred_quat, dim=1, keepdim=True)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-6), f"Quaternion norm: {quat_norm}"
    trans_norm = torch.norm(pred_trans, dim=1, keepdim=True)
    assert torch.allclose(trans_norm, torch.ones_like(trans_norm), atol=1e-6), f"Translation norm: {trans_norm}"
