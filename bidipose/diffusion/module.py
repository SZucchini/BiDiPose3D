import torch
import pytorch_lightning as pl
from typing import Any, List, Tuple

import torch.nn as nn
import torch.optim as optim

from bidipose.diffusion.sampler import DiffusionSampler
from bidipose.models.base import BaseModel
from bidipose.diffusion.utils import get_spatial_mask, get_temporal_mask, get_camera_mask


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
        model: BaseModel,
        sampler: DiffusionSampler,
        optimizer_name: str,
        optimizer_params: dict,
        num_validation_batches_to_sample: int = 5,
        num_validation_batches_to_inpaint: int = 5,
        num_plot_sample: int = 5,
        num_plot_inpaint: int = 5,
        inpinting_spatial_index: List[int] = None,
        inpainting_temporal_interval: Tuple[int, int] = None,
        inpainting_camera_index: int = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.loss_fn = nn.MSELoss()
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.num_validation_batches_to_sample = num_validation_batches_to_sample
        self.num_validation_batches_to_inpaint = num_validation_batches_to_inpaint
        self.num_plot_sample = num_plot_sample
        self.num_plot_inpaint = num_plot_inpaint
        self.validation_batches: List[Any] = []
        self.inpinting_spatial_index = inpinting_spatial_index
        self.inpainting_temporal_interval = inpainting_temporal_interval
        self.inpainting_camera_index = inpainting_camera_index

    def forward(
        self, 
        x: torch.Tensor, 
        quat: torch.Tensor, 
        trans: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.model.forward(x, quat, trans, t)
        
    def criterion(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for a batch.
        Args:
            batch (Any): Batch data.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Loss and individual component losses.
        """
        x, quat, trans = batch
        x = x.to(self.device)
        quat = quat.to(self.device)
        trans = trans.to(self.device)
        t = torch.randint(0, self.sampler.timesteps, (x.size(0),), device=x.device)
        x_noise, quat_noise, trans_noise = self.sampler.q_sample(x, quat, trans, t)
        x_pred, quat_pred, trans_pred = self.forward(x_noise, quat_noise, trans_noise, t)
        loss_x = self.loss_fn(x_pred, x)
        loss_quat = self.loss_fn(quat_pred, quat)
        loss_trans = self.loss_fn(trans_pred, trans)
        loss = loss_x + loss_quat + loss_trans
        return loss, loss_x, loss_quat, loss_trans
    
    def sample(
        self,
        x_shape: Tuple[int, ...],
        quat_shape: Tuple[int, ...],
        trans_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from the diffusion model.

        Args:
            x_shape (Tuple[int, ...]): Shape of the 2D pose data.
            quat_shape (Tuple[int, ...]): Shape of the quaternion data.
            trans_shape (Tuple[int, ...]): Shape of the translation data.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled 2D pose, quaternion, and translation data.
        """
        x, quat, trans = self.sampler.sample(
            self.model,
            x_shape,
            quat_shape,
            trans_shape,
        )
        return x, quat, trans
    
    def inpaint(
        self,
        x_init: torch.Tensor,
        quat_init: torch.Tensor,
        trans_init: torch.Tensor,
        x_mask: torch.Tensor = None,
        quat_mask: torch.Tensor = None,
        trans_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inpaint missing data using the diffusion model.

        Args:
            x_mask (torch.Tensor): Mask for 2D pose data.
            quat_mask (torch.Tensor): Mask for quaternion data.
            trans_mask (torch.Tensor): Mask for translation data.
            x_init (torch.Tensor): Initial 2D pose data for masked regions.
            quat_init (torch.Tensor): Initial quaternion data for masked regions.
            trans_init (torch.Tensor): Initial translation data for masked regions.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Inpainted 2D pose, quaternion, and translation data.
        """
        x_shape = x_init.shape
        quat_shape = quat_init.shape
        trans_shape = trans_init.shape
        x, quat, trans = self.sampler.sample(
            self.model,
            x_shape,
            quat_shape,
            trans_shape,
            x_init=x_init,
            quat_init=quat_init,
            trans_init=trans_init,
            x_mask=x_mask,
            quat_mask=quat_mask,
            trans_mask=trans_mask,
        )
        return x, quat, trans

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
    
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Validation step.

        Args:
            batch (Any): Batch data.
            batch_idx (int): Batch index.
        """
        loss, loss_x, loss_quat, loss_trans = self.criterion(batch)
        self.log("val/loss", loss)
        self.log("val/loss_x", loss_x)
        self.log("val/loss_quat", loss_quat)
        self.log("val/loss_trans", loss_trans)
        # Save a few batches for later use
        if len(self.validation_batches) < self.num_validation_batches_to_inpaint:
            # Detach tensors to avoid memory leak
            detached_batch = tuple(tensor.detach().cpu() for tensor in batch)
            self.validation_batches.append(detached_batch)
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        Uses the saved validation batches.
        """
        # Example: just print the number of saved batches
        print(f"Number of saved validation batches: {len(self.validation_batches)}")
        # You can use self.validation_batches here for further processing
        # Clear the saved batches for the next epoch

        if len(self.validation_batches) > 0:
            x, quat, trans = self.validation_batches[0]
            x_shape = x.shape
            quat_shape = quat.shape
            trans_shape = trans.shape

        # Validate inpainting
        for i, batch in enumerate(self.validation_batches):
            x, quat, trans = batch
            x = x.to(self.device)
            quat = quat.to(self.device)
            trans = trans.to(self.device)

            # Validate spatial inpainting
            if self.inpinting_spatial_index is not None:
                x_mask = ~get_spatial_mask(x, self.inpinting_spatial_index)
                quat_mask = torch.ones_like(quat, dtype=torch.bool)
                trans_mask = torch.ones_like(trans, dtype=torch.bool)
                x_inpainted, _, _ = self.inpaint(
                    x_init=x,
                    quat_init=quat,
                    trans_init=trans,
                    x_mask=x_mask,
                    quat_mask=quat_mask,
                    trans_mask=trans_mask,
                )
                # TODO: Add code to evaluate and log the inpainted results

            # Validate temporal inpainting
            if self.inpainting_temporal_interval is not None:
                temporal_index = torch.arange(
                    self.inpainting_temporal_interval[0],
                    self.inpainting_temporal_interval[1]
                )
                x_mask = ~get_temporal_mask(x, temporal_index)
                quat_mask = torch.ones_like(quat, dtype=torch.bool)
                trans_mask = torch.ones_like(trans, dtype=torch.bool)
                x_inpainted, _, _ = self.inpaint(
                    x_init=x,
                    quat_init=quat,
                    trans_init=trans,
                    x_mask=x_mask,
                    quat_mask=quat_mask,
                    trans_mask=trans_mask,
                )
                # TODO: Add code to evaluate and log the inpainted results

            # Validate camera inpainting
            if self.inpainting_camera_index is not None:
                x_mask = ~get_camera_mask(x, self.inpainting_camera_index)
                quat_mask = torch.ones_like(quat, dtype=torch.bool)
                trans_mask = torch.ones_like(trans, dtype=torch.bool)
                x_inpainted, _, _ = self.inpaint(
                    x_init=x,
                    quat_init=quat,
                    trans_init=trans,
                    x_mask=x_mask,
                    quat_mask=quat_mask,
                    trans_mask=trans_mask,
                )
                # TODO: Add code to evaluate and log the inpainted results

            # Validate camera parameter inpainting
            x_mask = torch.ones_like(x, dtype=torch.bool)
            quat_mask = torch.zeros_like(quat, dtype=torch.bool)
            trans_mask = torch.zeros_like(trans, dtype=torch.bool)
            _, quat_inpainted, trans_inpainted = self.inpaint(
                x_init=x,
                quat_init=quat,
                trans_init=trans,
                x_mask=x_mask,
                quat_mask=quat_mask,
                trans_mask=trans_mask,
            )
            # TODO: Add code to evaluate and log the inpainted results

        # Clear the validation batches to free memory
        if len(self.validation_batches) > 0:
            print(f"Processed {len(self.validation_batches)} validation batches for inpainting.")
        else:
            print("No validation batches to process for inpainting.")
        self.validation_batches.clear()

        # Validate sampling
        for i in range(self.num_plot_sample):
            x_sample, quat_sample, trans_sample = self.sample(
                x_shape=x_shape,
                quat_shape=quat_shape,
                trans_shape=trans_shape,
            )
            # TODO: Add code to evaluate and log the sampled results
    
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