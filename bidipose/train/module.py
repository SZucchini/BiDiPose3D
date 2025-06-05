import torch
import pytorch_lightning as pl
from typing import Any

import torch.nn as nn
import torch.optim as optim

from bidipose.diffusion.sampler import DiffusionSampler


class DiffusionLightningModule(pl.LightningModule):
    """
    PyTorch LightningModule for training a diffusion model.

    Args:
        model (nn.Module): Diffusion model.
        lr (float): Learning rate.
        betas (tuple): Adam optimizer betas.
    """
    def __init__(
        self,
        model: nn.Module,
        sampler: DiffusionSampler,
        optimizer_name: str,
        optimizer_params: dict,
    ) -> None:
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.loss_fn = nn.MSELoss()
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params

    def forward(
        self, 
        x: torch.Tensor, 
        quat: torch.Tensor, 
        trans: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            quat (torch.Tensor): Quaternion tensor.
            trans (torch.Tensor): Translation tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Quaternion tensor.
            torch.Tensor: Translation tensor.
        """
        return self.model(x, quat, trans, t)
        
    def criterion(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for a batch.
        Args:
            batch (Any): Batch data.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and individual component losses.
        """
        x, quat, trans = batch
        t = torch.randint(0, self.sampler.num_steps, (x.size(0),), device=x.device)
        x_noise, quat_noise, trans_noise = self.sampler.q_sample(x, quat, trans, t)
        x_pred, quat_pred, trans_pred = self.forward(x_noise, quat_noise, trans_noise, t)
        loss_x = self.loss_fn(x_pred, x)
        loss_quat = self.loss_fn(quat_pred, quat)
        loss_trans = self.loss_fn(trans_pred, trans)
        loss = loss_x + loss_quat + loss_trans
        return loss, loss_x, loss_quat, loss_trans

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Any): Batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        loss, loss_x, loss_quat, loss_trans = self.criterion(batch)
        self.log("train/loss", loss)
        self.log("train/loss_x", loss_x)
        self.log("train/loss_quat", loss_quat)
        self.log("train/loss_trans", loss_trans)
        return loss
    
    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure optimizers for training.

        Returns:
            optim.Optimizer: Adam optimizer.
        """
        optimizer = getattr(optim, self.optimizer_name)(
            self.model.parameters(),
            **self.optimizer_params
        )
        return optimizer
    