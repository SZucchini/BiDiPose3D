import torch
from typing import Optional, Tuple
from tqdm import tqdm

import torch.nn as nn

import bidipose.diffusion.scheduler as scheduler

class DDPMSampler:
    """
    DDPMSampler performs sampling from a diffusion model, supporting mask inpainting.

    Args:
        beta_scheduler_name (str): Name of the beta scheduler to use.
        beta_scheduler_params (dict): Parameters for the beta scheduler.
        prediction_type (str): Type of prediction ('x0' or 'noise').
        device (Optional[torch.device]): Device for computation.

    Attributes:
        timesteps (int): Number of diffusion steps.
        betas (torch.Tensor): Noise schedule.
        alphas (torch.Tensor): 1 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        device (torch.device): Device for computation.
        prediction_type (str): Type of prediction ('x0' or 'noise').
    """

    def __init__(
        self,
        beta_scheduler_name: str,
        beta_scheduler_params: dict,
        prediction_type: str = 'x0',
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize DDPMSampler.

        Args:
            beta_scheduler_name (str): Name of the beta scheduler to use.
            beta_scheduler_params (dict): Parameters for the beta scheduler.
            predict_x0 (bool): Whether the model predicts x0 or noise.
            device (Optional[torch.device]): Device for computation.
        """
        self.device = device if device is not None else torch.device('cpu')
        self.prediction_type = prediction_type
        self.betas = getattr(scheduler, beta_scheduler_name)(**beta_scheduler_params).to(self.device)
        self.alphas, self.alphas_cumprod = scheduler.beta_to_alpha(self.betas)
        self.timesteps = self.betas.shape[0]

    def _adjust_dimensions(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Adjust the dimensions of the input tensor to match the timestep tensor.

        Args:
            x (torch.Tensor): Input tensor to be adjusted.
            t (torch.Tensor): Timestep tensor.

        Returns:
            torch.Tensor: Adjusted tensor with dimensions matching the timestep.
        """
        for _ in range(x.dim() - t.dim()):
            t = t.unsqueeze(-1)
        return t
                          
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
    
    def clean_to_noise(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert clean data to noise at timestep t.

        Args:
            x_clean (torch.Tensor): Clean data to be noised.
            x_noisy (torch.Tensor): Noise to be added.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Noise at timestep t.
        """
        t = self._adjust_dimensions(x_noisy, t)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        # x_noisy = sqrt_alpha_cumprod * x_clean + sqrt_one_minus_alpha_cumprod * noise
        noise = (x_noisy - sqrt_alpha_cumprod * x_clean) / sqrt_one_minus_alpha_cumprod
        return noise
    
    def noise_to_clean(
        self,
        noise: torch.Tensor,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert noise to clean data at timestep t.

        Args:
            noise (torch.Tensor): Noise to be converted.
            x_noise (torch.Tensor): Noise to be added.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Clean data at timestep t.
        """
        t = self._adjust_dimensions(x_noisy, t)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        # x_noisy = sqrt_alpha_cumprod * x_clean + sqrt_one_minus_alpha_cumprod * noise
        x_clean = (x_noisy - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod
        return x_clean
    
    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data (add noise) at timestep t.

        Args:
            x0 (torch.Tensor): Original data to be noised.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Noised data at timestep t.
        """
        t = self._adjust_dimensions(x0, t)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        noise = torch.randn_like(x0)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def q_sample(
        self,
        x0: torch.Tensor,
        quat0: torch.Tensor,
        trans0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Diffuse the data (add noise) at timestep t.

        Args:
            x0 (torch.Tensor): Original 2D pose data to be noised.
            quat0 (torch.Tensor): Quaternion data to be noised.
            trans0 (torch.Tensor): Translation data to be noised.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Noised 2D pose data at timestep t.
            torch.Tensor: Noised quaternion data at timestep t.
            torch.Tensor: Noised translation data at timestep t.
        """
        x = self._q_sample(x0, t)
        quat = self._q_sample(quat0, t)
        trans = self._q_sample(trans0, t)
        return x, quat, trans
    
    def _p_sample(self, x: torch.Tensor, noise: torch.Tensor, t: int) -> torch.Tensor:
        """
        Perform one reverse diffusion step for data.

        Args:
            x (torch.Tensor): Current sample.
            noise (torch.Tensor): Noise to be removed.
            t (int): Current timestep.

        Returns:
            torch.Tensor: Updated sample after reverse diffusion step.
        """
        beta = self.betas[t]
        sigma = torch.sqrt(beta)
        sqrt_alpha = torch.sqrt(self.alphas[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])

        x_mean = (x - noise * beta / sqrt_one_minus_alpha_cumprod) / sqrt_alpha
        if t > 0:
            sigma = torch.sqrt(beta)
        else:
            # No noise at the last step
            sigma = torch.tensor(0.0, device=x.device)
        # Reverse diffusion step
        noise_added = torch.randn_like(x)
        return x_mean + sigma * noise_added
        
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
                t_tensor = torch.full((x.size(0),), t-1, device=x.device)
                x_dummy = self._q_sample(x_init, t_tensor)
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
            t (int): Current timestep.
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
        t_tensor = torch.full((x.size(0),), t, device=x.device)
        # Predict data using the model
        x_pred, quat_pred, trans_pred = model(x, quat, trans, t_tensor)
        # Convert predictions to noise
        if self.prediction_type == 'x0':
            x_noise = self.clean_to_noise(x_pred, x, t_tensor)
            quat_noise = self.clean_to_noise(quat_pred, quat, t_tensor)
            trans_noise = self.clean_to_noise(trans_pred, trans, t_tensor)
        else:
            x_noise = x_pred
            quat_noise = quat_pred
            trans_noise = trans_pred

        # Perform reverse diffusion step
        x_prev = self._p_sample(x, x_noise, t)
        quat_prev = self._p_sample(quat, quat_noise, t)
        trans_prev = self._p_sample(trans, trans_noise, t)

        # Apply masks to fill in the initial data for masked regions
        x_prev = self._masked_fill(x_prev, x_init, x_mask, t)
        quat_prev = self._masked_fill(quat_prev, quat_init, quat_mask, t)
        trans_prev = self._masked_fill(trans_prev, trans_init, trans_mask, t)

        return x_prev, quat_prev, trans_prev
