from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bidipose.diffusion.sampler import DiffusionSampler

class Trainer:
    """
    Trainer class for training models with DiffusionSampler.

    Args:
        model (torch.nn.Module): The model to be trained.
        sampler (DiffusionSampler): The diffusion sampler instance.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device to run training on.
        epochs (int): Number of training epochs.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: DiffusionSampler,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device,
        epochs: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler

    def train(self) -> None:
        """
        Run the training loop.
        """
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                x, quat, trans = batch
                x = x.to(self.device)
                quat = quat.to(self.device)
                trans = trans.to(self.device)
                t = torch.randint(0, self.sampler.num_steps, (x.size(0),), device=self.device).long()
                x_noisy, quat_noisy, trans_noisy = self.sampler.q_sample(
                    x,
                    quat,
                    trans, 
                    t
                )

                self.optimizer.zero_grad()
                # Forward pass using the diffusion sampler
                x_pred, quat_pred, trans_pred = self.model(x_noisy, quat_noisy, trans_noisy, t)
                # Compute 
                x_loss = nn.functional.mse_loss(x_pred, x)
                quat_loss = nn.functional.mse_loss(quat_pred, quat)
                trans_loss = nn.functional.mse_loss(trans_pred, trans)
                loss = x_loss + quat_loss + trans_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.dataloader):.4f}")
