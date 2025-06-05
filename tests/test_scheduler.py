import torch

from bidipose.diffusion.scheduler import (
    beta_to_alpha, 
    linear_beta_schedule, 
    cosine_beta_schedule, 
    quadratic_beta_schedule,
)


def test_linear_beta_scheduler():
    """Test the LinearBetaScheduler."""
    num_steps = 1000
    betas = linear_beta_schedule(num_steps)
    alphas, alphas_cumprod = beta_to_alpha(betas)
    
    assert len(betas) == num_steps, "Betas length mismatch"
    assert len(alphas) == num_steps, "Alphas length mismatch"
    assert len(alphas_cumprod) == num_steps, "Alphas cumulative product length mismatch"
    assert torch.all(betas > 0) and torch.all(betas <= 1), "Betas should be in the range [0, 1]"
    assert torch.all(alphas > 0) and torch.all(alphas <= 1), "Alphas should be in the range [0, 1]"
    assert torch.all(alphas_cumprod >= 0) and torch.all(alphas_cumprod <= 1), "Alphas cumulative product should be in the range [0, 1]"
    assert torch.all(alphas_cumprod[:-1] >= alphas_cumprod[1:]), "Alphas cumulative product should be non-increasing"

def test_cosine_beta_scheduler():
    """Test the CosineBetaScheduler."""
    num_steps = 1000
    betas = cosine_beta_schedule(num_steps)
    alphas, alphas_cumprod = beta_to_alpha(betas)
    
    assert len(betas) == num_steps, "Betas length mismatch"
    assert len(alphas) == num_steps, "Alphas length mismatch"
    assert len(alphas_cumprod) == num_steps, "Alphas cumulative product length mismatch"
    assert torch.all(betas > 0) and torch.all(betas <= 1), "Betas should be in the range [0, 1]"
    assert torch.all(alphas > 0) and torch.all(alphas <= 1), "Alphas should be in the range [0, 1]"
    assert torch.all(alphas_cumprod >= 0) and torch.all(alphas_cumprod <= 1), "Alphas cumulative product should be in the range [0, 1]"
    assert torch.all(alphas_cumprod[:-1] >= alphas_cumprod[1:]), "Alphas cumulative product should be non-increasing"

def test_quadratic_beta_scheduler():
    """Test the QuadraticBetaScheduler."""
    num_steps = 1000
    betas = quadratic_beta_schedule(num_steps)
    alphas, alphas_cumprod = beta_to_alpha(betas)
    
    assert len(betas) == num_steps, "Betas length mismatch"
    assert len(alphas) == num_steps, "Alphas length mismatch"
    assert len(alphas_cumprod) == num_steps, "Alphas cumulative product length mismatch"
    assert torch.all(betas > 0) and torch.all(betas <= 1), "Betas should be in the range [0, 1]"
    assert torch.all(alphas > 0) and torch.all(alphas <= 1), "Alphas should be in the range [0, 1]"
    assert torch.all(alphas_cumprod >= 0) and torch.all(alphas_cumprod <= 1), "Alphas cumulative product should be in the range [0, 1]"
    assert torch.all(alphas_cumprod[:-1] >= alphas_cumprod[1:]), "Alphas cumulative product should be non-increasing"
