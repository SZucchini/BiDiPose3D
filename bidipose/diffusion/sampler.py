import torch
from typing import Optional, Tuple
from tqdm import tqdm

import torch.nn as nn

import bidipose.diffusion.scheduler as scheduler

class DDPMSampler:
    """
    DDPMSampler performs sampling from a diffusion model, supporting mask inpainting.

    Args:
        betas (torch.Tensor): Beta values for the noise schedule (1D tensor).
        device (Optional[torch.device]): Device for computation.

    Attributes:
        timesteps (int): Number of diffusion steps.
        betas (torch.Tensor): Noise schedule.
        alphas (torch.Tensor): 1 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        device (torch.device): Device for computation.
    """

    def __init__(
        self,
        beta_scheduler_name: str,
        beta_scheduler_params: dict,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize DDPMSampler.

        Args:
            betas (torch.Tensor): Beta values for the noise schedule (1D tensor).
            device (Optional[torch.device]): Device for computation.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.betas = getattr(scheduler, beta_scheduler_name)(**beta_scheduler_params).to(self.device)
        self.alphas, self.alphas_cumprod = scheduler.beta_to_alpha(self.betas)
        self.timesteps = self.betas.shape[0]

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_shape: Tuple[int, ...],
        quat_shape: Tuple[int, ...],
        trans_shape: Tuple[int, ...],
        x_init: Optional[torch.Tensor] = None,
        quat_init: Optional[torch.Tensor] = None,
        trans_init: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        quat_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate samples from the diffusion model, supporting mask inpainting.

        Args:
            model (nn.Module): The trained diffusion model.
            x_shape (Tuple[int, ...]): Shape of the 2D pose data to be generated.
            quat_shape (Tuple[int, ...]): Shape of the quaternion data.
            trans_shape (Tuple[int, ...]): Shape of the translation data.
            x_mask (Optional[torch.Tensor]): Mask for 2D pose data (1 for keep, 0 for inpaint).
            quat_mask (Optional[torch.Tensor]): Mask for quaternion data (1 for keep, 0 for inpaint).
            trans_mask (Optional[torch.Tensor]): Mask for translation data (1 for keep, 0 for inpaint).
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

        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            x, quat, trans = self.p_sample(
                model,
                x,
                quat,
                trans,
                t,
                x_init=x_init,
                quat_init=quat_init,
                trans_init=trans_init,
                x_mask=x_mask,
                quat_mask=quat_mask,
                trans_mask=trans_mask,
            )
        return x, quat, trans
    
    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data (add noise) at timestep t.

        Args:
            x_start (torch.Tensor): Original data to be noised.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Noised data at timestep t.
        """
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        noise = torch.randn_like(x_start)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

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
        x = self._q_sample(x_start, t)
        quat = self._q_sample(quat_start, t)
        trans = self._q_sample(trans_start, t)
        return x, quat, trans
    
    def _p_sample(self, x: torch.Tensor, x0_pred: torch.Tensor, t: int) -> torch.Tensor:
        """
        Perform one reverse diffusion step for data.

        Args:
            x (torch.Tensor): Current sample.
            x0_pred (torch.Tensor): Predicted x_0 from the model.
            t (int): Current timestep.

        Returns:
            torch.Tensor: Updated sample after reverse diffusion step.
        """
        if t > 0:
            beta = self.betas[t]
            sigma = torch.sqrt(beta)
            coef1 = torch.sqrt(self.alphas_cumprod[t - 1])
            coef2 = torch.sqrt(1.0 - self.alphas_cumprod[t - 1])
            coef3 = torch.sqrt(self.alphas_cumprod[t])
            coef4 = torch.sqrt(1.0 - self.alphas_cumprod[t])

            x_mean = coef1 * x0_pred + coef2 * (x - coef3 * x0_pred) / coef4
            noise = torch.randn_like(x)
            return x_mean + sigma * noise
        else:
            # No noise at the last step
            return x0_pred
        
    def _masked_fill(self, x: torch.Tensor, x_init: torch.Tensor, mask: torch.Tensor, t:int) -> torch.Tensor:
        """
        Fill masked regions with initial data for inpainting.

        Args:
            x (torch.Tensor): Current sample.
            x_init (torch.Tensor): Initial data for masked regions.
            mask (torch.Tensor): Mask indicating which regions to keep.

        Returns:
            torch.Tensor: Sample with masked regions filled from initial data.
        """
        if mask is not None and x_init is not None:
            if t > 0:
                x_dummy = self._q_sample(x, torch.full((x.size(0),), t-1, device=x.device))
            else:
                x_dummy = x_init
            # For masked inpainting, keep known regions from x_init
            x_filled = mask * x_dummy + (~mask) * x
        else:
            # If no mask or initial data, return the current sample
            x_filled = x
        return x_filled

    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
        t: int,
        x_init: Optional[torch.Tensor] = None,
        quat_init: Optional[torch.Tensor] = None,
        trans_init: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        quat_mask: Optional[torch.Tensor] = None,
        trans_mask: Optional[torch.Tensor] = None, 
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
            quat_mask (Optional[torch.Tensor]): Mask for quaternion data (1 for keep, 0 for inpaint).
            trans_mask (Optional[torch.Tensor]): Mask for translation data (1 for keep, 0 for inpaint).
            x_init (Optional[torch.Tensor]): Initial 2D pose data for masked regions in x.
            quat_init (Optional[torch.Tensor]): Initial quaternion data for masked regions
            trans_init (Optional[torch.Tensor]): Initial translation data for masked regions.

        Returns:
            torch.Tensor: Updated 2D pose sample after reverse diffusion step.
            torch.Tensor: Updated quaternion sample after reverse diffusion step.
            torch.Tensor: Updated translation sample after reverse diffusion step.
        """
        # Predict x_0 using the model
        x0_pred, quat0_pred, trans0_pred = model(x, quat, trans, t)

        if t > 0:
            # Perform reverse diffusion step
            x_prev = self._p_sample(x, x0_pred, t)
            quat_prev = self._p_sample(quat, quat0_pred, t)
            trans_prev = self._p_sample(trans, trans0_pred, t)
        else: # No noise at the last step
            x_prev = x0_pred  
            quat_prev = quat0_pred
            trans_prev = trans0_pred

        # Apply masks to fill in the initial data for masked regions
        x_prev = self._masked_fill(x_prev, x_init, x_mask, t)
        quat_prev = self._masked_fill(quat_prev, quat_init, quat_mask, t)
        trans_prev = self._masked_fill(trans_prev, trans_init, trans_mask, t)

        return x_prev, quat_prev, trans_prev
