import torch
from typing import Optional, Tuple

import torch.nn as nn

class DiffusionSampler:
    """
    DiffusionSampler performs sampling from a diffusion model, supporting mask inpainting.

    Args:
        betas (torch.Tensor): Beta values for the noise schedule (1D tensor).
        device (Optional[torch.device]): Device for computation.

    Attributes:
        num_steps (int): Number of diffusion steps.
        betas (torch.Tensor): Noise schedule.
        alphas (torch.Tensor): 1 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        device (torch.device): Device for computation.
    """

    def __init__(
        self,
        betas: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize DiffusionSampler.

        Args:
            betas (torch.Tensor): Beta values for the noise schedule (1D tensor).
            device (Optional[torch.device]): Device for computation.
        """
        self.betas = betas.to(device) if device is not None else betas
        self.num_steps = self.betas.shape[0]
        self.device = device if device is not None else torch.device('cpu')
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample(
        self,
        model: nn.Module,
        x_shape: Tuple[int, ...],
        quat_shape: Tuple[int, ...],
        trans_shape: Tuple[int, ...],
        x_mask: Optional[torch.Tensor] = None,
        quat_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
        quat_init: Optional[torch.Tensor] = None,
        trans_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate samples from the diffusion model, supporting mask inpainting.

        Args:
            model (nn.Module): The trained diffusion model.
            x_shape (Tuple[int, ...]): Shape of the 2D pose data to be generated.
            quat_shape (Tuple[int, ...]): Shape of the quaternion data.
            trans_shape (Tuple[int, ...]): Shape of the translation data.
            x_mask (Optional[torch.Tensor]): Mask for 2D pose data (1 for keep, 0 for inpaint).
            quat_mask (Optional[torch.Tensor]): Mask for quaternion data.
            trans_mask (Optional[torch.Tensor]): Mask for translation data.
            x_init (Optional[torch.Tensor]): Initial 2D pose data for masked regions in x.
            quat_init (Optional[torch.Tensor]): Initial quaternion data for masked regions.
            trans_init (Optional[torch.Tensor]): Initial translation data for masked regions.

        Returns:
            torch.Tensor: Generated 2D pose data.
            torch.Tensor: Generated quaternion data.
            torch.Tensor: Generated translation data.
        """
        x = torch.randn(x_shape, device=self.device)
        quat = torch.randn(quat_shape, device=self.device)
        trans = torch.randn(trans_shape, device=self.device)

        for t in reversed(range(self.num_steps)):
            x, quat, trans = self.p_sample(
                model,
                x,
                quat,
                trans,
                t,
                x_mask,
                quat_mask,
                trans_mask,
                x_init,
                quat_init,
                trans_init
            )
        return x, quat, trans

    def q_sample(
        self,
        x_start: torch.Tensor,
        quat_start: torch.Tensor,
        trans_start: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Diffuse the data (add noise) at timestep t.

        Args:
            x_start (torch.Tensor): Original 2D pose data to be noised.
            quat_start (torch.Tensor): Quaternion data to be noised.
            trans_start (torch.Tensor): Translation data to be noised.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Noised 2D pose data at timestep t.
            torch.Tensor: Noised quaternion data at timestep t.
            torch.Tensor: Noised translation data at timestep t.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        noise = torch.randn_like(x_start)
        x = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
        noise = torch.randn_like(quat_start)
        quat = sqrt_alpha_cumprod * quat_start + sqrt_one_minus_alpha_cumprod * noise
        noise = torch.randn_like(trans_start)
        trans = sqrt_alpha_cumprod * quat_start + sqrt_one_minus_alpha_cumprod * noise
        return x, quat, trans

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
        t: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        quat_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
        quat_init: Optional[torch.Tensor] = None,
        trans_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one reverse diffusion step, supporting mask inpainting.

        Args:
            model (nn.Module): The trained diffusion model.
            x (torch.Tensor): Current 2D pose sample.
            quat (torch.Tensor): Current quaternion sample
            trans (torch.Tensor): Current translation sample.
            t (torch.Tensor): Current timestep.
            x_mask (Optional[torch.Tensor]): Mask for 2D pose data (1 for keep, 0 for inpaint).
            quat_mask (Optional[torch.Tensor]): Mask for quaternion data.
            trans_mask (Optional[torch.Tensor]): Mask for translation data.
            x_init (Optional[torch.Tensor]): Initial 2D pose data for masked regions in x.
            quat_init (Optional[torch.Tensor]): Initial quaternion data for masked regions
            trans_init (Optional[torch.Tensor]): Initial translation data for masked regions.

        Returns:
            torch.Tensor: Updated 2D pose sample after reverse diffusion step.
            torch.Tensor: Updated quaternion sample after reverse diffusion step.
            torch.Tensor: Updated translation sample after reverse diffusion step.
        """
        beta = self.betas[t]
        sigma = torch.sqrt(beta)

        # Predict x_0 using the model
        x0_pred, quat0_pred, trans0_pred = model(x, quat, trans, t)

        if t > 0:
            coef1 = torch.sqrt(self.alphas_cumprod[t - 1])
            coef2 = torch.sqrt(1.0 - self.alphas_cumprod[t - 1])
            coef3 = torch.sqrt(self.alphas_cumprod[t])
            coef4 = torch.sqrt(1.0 - self.alphas_cumprod[t])

            x_mean = coef1 * x0_pred + coef2 * (x - coef3 * x0_pred) / coef4
            noise = torch.randn_like(x)
            x_prev = x_mean + sigma * noise

            quat_mean = coef1 * quat0_pred + coef2 * (quat - coef3 * quat0_pred) / coef4
            noise = torch.randn_like(quat)
            quat_prev = quat_mean + sigma * noise

            trans_mean = coef1 * trans0_pred + coef2 * (trans - coef3 * trans0_pred) / coef4
            noise = torch.randn_like(trans)
            trans_prev = trans_mean + sigma * noise

        else: # No noise at the last step
            x_prev = x0_pred  
            quat_prev = quat0_pred
            trans_prev = trans0_pred

        x_dummy, quat_dummy, trans_dummy = self.q_sample(x_init, quat_init, trans_init, t)

        if x_mask is not None and x_init is not None:
            # For masked inpainting, keep known regions from x_init
            x0_prev = x_mask * x_dummy + (1 - x_mask) * x0_prev

        if quat_mask is not None and quat_init is not None:
            # For masked inpainting, keep known regions from quat_init
            quat0_prev = quat_mask * quat_dummy + (1 - quat_mask) * quat0_prev

        if trans_mask is not None and trans_init is not None:
            # For masked inpainting, keep known regions
            trans0_prev = trans_mask * trans_dummy + (1 + trans_mask) * trans0_prev

        return x_prev, quat_prev, trans_prev
