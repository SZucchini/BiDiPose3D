from typing import Tuple
import torch
import math

def beta_to_alpha(
    betas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert beta values to alpha and cumulative alpha values.

    Args:
        betas (torch.Tensor): Tensor of beta values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - alphas (torch.Tensor): 1 - betas.
            - alphas_cumprod (torch.Tensor): Cumulative product of alphas.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod

def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """
    Generate linear beta schedule for diffusion models.

    Args:
        timesteps (int): Number of diffusion steps.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.

    Returns:
        torch.Tensor: Tensor of beta values for each timestep.
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return betas

def cosine_beta_schedule(
    timesteps: int, s: float = 0.008
) -> torch.Tensor:
    """
    Generate cosine beta schedule for diffusion models.

    Args:
        timesteps (int): Number of diffusion steps.
        s (float): Small offset to prevent singularities.

    Returns:
        torch.Tensor: Tensor of beta values for each timestep.
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    t = steps / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod.float()
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=0.0001, max=0.9999)
    return betas

def quadratic_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """
    Generate quadratic beta schedule for diffusion models.

    Args:
        timesteps (int): Number of diffusion steps.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.

    Returns:
        torch.Tensor: Tensor of beta values for each timestep.
    """
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
    return betas
