"""Test for the pose models."""

import torch

from bidipose.models.MotionAGFormer.model import MotionAGFormer
from bidipose.diffusion.sampler import DDPMSampler


def test_sampler():
    """Test the DDPMSampler."""
    timesteps = 1000
    sampler = DDPMSampler(
        beta_scheduler_name='linear_beta_schedule',
        beta_scheduler_params={'timesteps': timesteps},
        device=torch.device('cpu')
    )
    model = MotionAGFormer(n_layers=2, dim_feat=32, dim_in=6, n_frames=88)
    model.eval()

    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    t = torch.randint(0, timesteps, (2,), device=x.device)

    # Test forward process
    x_noise, quat_noise, trans_noise = sampler.q_sample(x, quat, trans, t)
    assert x_noise.shape == x.shape
    assert quat_noise.shape == quat.shape
    assert trans_noise.shape == trans.shape
    
    # Test reverse process
    x_recon, quat_recon, trans_recon = sampler.p_sample(model, x_noise, quat_noise, trans_noise, 100)
    assert x_recon.shape == x.shape
    assert quat_recon.shape == quat.shape
    assert trans_recon.shape == trans.shape

    # Test reverse process with masks
    x_mask = torch.randint(0, 2, x.shape, dtype=torch.bool)
    quat_mask = torch.randint(0, 2, quat.shape, dtype=torch.bool)
    trans_mask = torch.randint(0, 2, trans.shape, dtype=torch.bool)
    x_recon_masked, quat_recon_masked, trans_recon_masked = sampler.p_sample(
        model, x_noise, quat_noise, trans_noise, 100,
        x_init=x, quat_init=quat, trans_init=trans,
        x_mask=x_mask, quat_mask=quat_mask, trans_mask=trans_mask,
    )
    assert x_recon_masked.shape == x.shape
    assert quat_recon_masked.shape == quat.shape
    assert trans_recon_masked.shape == trans.shape

    # Test sampling
    x_sample, quat_sample, trans_sample = sampler.sample(model, x.shape, quat.shape, trans.shape)
    assert x_sample.shape == x.shape
    assert quat_sample.shape == quat.shape
    assert trans_sample.shape == trans.shape

    # Test sampling with masks
    x_mask = torch.randint(0, 2, x.shape, dtype=torch.bool)
    quat_mask = torch.randint(0, 2, quat.shape, dtype=torch.bool)
    trans_mask = torch.randint(0, 2, trans.shape, dtype=torch.bool)
    x_sample_masked, quat_sample_masked, trans_sample_masked = sampler.sample(
        model, x.shape, quat.shape, trans.shape,
        x_init=x, quat_init=quat, trans_init=trans,
        x_mask=x_mask, quat_mask=quat_mask, trans_mask=trans_mask,
    )
    assert x_sample_masked.shape == x.shape
    assert quat_sample_masked.shape == quat.shape
    assert trans_recon_masked.shape == trans.shape
    assert torch.all(x_sample_masked.masked_select(x_mask) == x.masked_select(x_mask)), "Masked x values mismatch"
    assert torch.all(quat_sample_masked.masked_select(quat_mask) == quat.masked_select(quat_mask)), "Masked quat values mismatch"
    assert torch.all(trans_sample_masked.masked_select(trans_mask) == trans.masked_select(trans_mask)), "Masked trans values mismatch"
