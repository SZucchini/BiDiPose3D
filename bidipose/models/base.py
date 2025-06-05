"""Base class for all pose models."""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all pose models."""

    def __init__(self):
        """Initialize the base model."""
        super(BaseModel, self).__init__()

    def forward(
        self, x: torch.Tensor, quat: torch.Tensor, trans: torch.Tensor, t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input 2D poses from two-views (B, T, J, 3*2).
            quat (torch.Tensor): Quaternions for cam1 to cam2 (B, 4, 1).
            trans (torch.Tensor): Translations for cam1 to cam2 (B, 3, 1).
            t (torch.Tensor | None): Diffusion time step (B, 1, 1).

        Returns:
            pred_pose (torch.Tensor): Predicted 2D poses from two-views (B, T, J, 3*2).
            pred_quat (torch.Tensor): Predicted quaternions for cam1 to cam2 (B, 4, 1).
            pred_trans (torch.Tensor): Predicted translations for cam1 to cam2 (B, 3, 1).

        """
        pass
