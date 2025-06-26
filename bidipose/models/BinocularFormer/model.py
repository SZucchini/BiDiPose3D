"""BinocularFormer model."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bidipose.models.base import BaseModel


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings.

        Args:
            dim (int): Embedding dimension.

        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass for sinusoidal position embeddings.

        Args:
            time (torch.Tensor): Timestep tensor with shape (B,).

        Returns:
            torch.Tensor: Sinusoidal embeddings with shape (B, dim).

        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """Positional Encoding module for adding positional information to input sequences."""

    def __init__(self, latent_dim: int, dropout: float = 0.1, max_len: int = 1000):
        """Initialize the PositionalEncoding module.

        Args:
            latent_dim (int): Dimension of the latent space.
            dropout (float): Dropout rate to apply after positional encoding.
            max_len (int): Maximum length of the input sequences.

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # .transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, latent_dim).

        Returns:
            x (torch.Tensor): Positional encoded tensor of shape (B, T, latent_dim).

        """
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    """InputProcess module for processing input data in the BinocularFormer model."""

    def __init__(
        self,
        latent_dim: int,
        num_joints: int = 17,
        dim_cam_params: int = 7,
        dropout: float = 0.1,
        fusion_type: str = "concat",
    ):
        """Initialize the InputProcess module.

        Args:
            latent_dim (int): Dimension of the latent space.
            num_joints (int): Number of joints in the input data.
            dim_cam_params (int): Dimension of the camera parameters.
            dropout (float): Dropout rate to apply after positional encoding.
            fusion_type (str): Type of fusion to apply ('add' or 'concat').

        """
        super().__init__()
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        self.joint_embedding = nn.Linear(3 * num_joints, latent_dim)
        self.camera_embedding = nn.Linear(dim_cam_params, latent_dim)
        self.positional_encoding = PositionalEncoding(latent_dim, dropout)

        self.fusion_type = fusion_type
        if self.fusion_type == "concat":
            self.fusion = nn.Linear(latent_dim * 2, latent_dim)

    def _embed_joint(self, x: torch.Tensor) -> torch.Tensor:
        """Embed joint positions into a latent space.

        Args:
            x (torch.Tensor): Joint positions of shape (B, T, J, C=3).

        Returns:
            x (torch.Tensor): Embedded joint positions of shape (B, T, latent_dim).

        """
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.joint_embedding(x)  # (B, T, latent_dim)
        return x

    def _embed_camera(self, cam_params: torch.Tensor, frames: int) -> torch.Tensor:
        """Embed camera parameters into a latent space.

        Args:
            cam_params (torch.Tensor): Camera parameters of shape (B, dim_cam_params).
            frames (int): Number of frames in the input sequence.

        Returns:
            cam_embed (torch.Tensor): Embedded camera parameters of shape (B, T, latent_dim).

        """
        cam_embed = self.camera_embedding(cam_params)  # (B, latent_dim)
        cam_embed = cam_embed.reshape(cam_embed.shape[0], 1, self.latent_dim)  # (B, 1, latent_dim)
        cam_embed = cam_embed.expand(cam_embed.shape[0], frames, self.latent_dim)  # (B, T, latent_dim)
        return cam_embed

    def forward(self, x: torch.Tensor, quat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MotionAGFormer model.

        Args:
            x (torch.Tensor): Input 2D from two-views (B, T, J, C=3*2).
            quat (torch.Tensor): Quaternions for cam1 to cam2 (B, 4).
            trans (torch.Tensor): Translation vector for cam1 to cam2 (B, 3).

        Returns:
            x (torch.Tensor): Processed input tensor with positional encoding and camera embeddings.

        """
        bs, frames, _, _ = x.shape
        x1 = x[:, :, :, :3]  # (B, T, J, 3)
        x1 = self._embed_joint(x1)  # (B, T, latent_dim)
        x2 = x[:, :, :, 3:]
        x2 = self._embed_joint(x2)

        cam1_params = torch.zeros((bs, 7), device=x1.device)  # (B, 7)
        cam1_params[:, 0] = 1.0
        cam1_embed = self._embed_camera(cam1_params, frames)  # (B, T, latent_dim)
        cam2_params = torch.cat([quat, trans], dim=-1)
        cam2_embed = self._embed_camera(cam2_params, frames)

        if self.fusion_type == "add":
            x1 = x1 + cam1_embed  # (B, T, latent_dim)
            x1 = self.positional_encoding(x1)  # (B, T, latent_dim)
            x2 = x2 + cam2_embed
            x2 = self.positional_encoding(x2)
        else:
            x1 = torch.cat([x1, cam1_embed], dim=-1)  # (B, T, latent_dim * 2)
            x1 = self.fusion(x1)  # (B, T, latent_dim)
            x1 = self.positional_encoding(x1)  # (B, T, latent_dim)
            x2 = torch.cat([x2, cam2_embed], dim=-1)
            x2 = self.fusion(x2)
            x2 = self.positional_encoding(x2)

        x = torch.cat([x1, x2], dim=1)  # (B, T * 2, latent_dim)
        return x


class BinocularFormer(BaseModel):
    """BinocularFormer model for processing binocular input data."""

    def __init__(
        self,
        latent_dim: int,
        num_head: int,
        num_layers: int,
        ff_size: int,
        dropout: float,
        activation: str,
        timestep_embed_dim: int = 64,
        num_joints: int = 17,
        dim_cam_params: int = 7,
        fusion_type: str = "concat",
    ):
        """Initialize the BinocularFormer model.

        Args:
            latent_dim (int): Dimension of the latent space.
            num_head (int): Number of attention heads in the transformer encoder.
            num_layers (int): Number of layers in the transformer encoder.
            ff_size (int): Dimension of the feedforward network in the transformer encoder.
            dropout (float): Dropout rate to apply after positional encoding.
            activation (str): Activation function to use in the transformer encoder.
            timestep_embed_dim (int): Dimension of the timestep embeddings.
            num_joints (int): Number of joints in the input data.
            dim_cam_params (int): Dimension of the camera parameters.
            fusion_type (str): Type of fusion to apply ('add' or 'concat').

        """
        super().__init__()
        self.input_process = InputProcess(latent_dim, num_joints, dim_cam_params, dropout, fusion_type)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 2),
            nn.GELU(),
            nn.Linear(timestep_embed_dim * 2, latent_dim),
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_head,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(latent_dim, num_joints * 3 + 7)

    def forward(
        self,
        x: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
        t: torch.Tensor | None = None,
    ):
        """Forward pass of the MotionAGFormer model.

        Args:
            x (torch.Tensor): Input 2D from two-views (B, T, J, C=3*2).
            quat (torch.Tensor): Quaternions for cam1 to cam2 (B, 4).
            trans (torch.Tensor): Translation vector for cam1 to cam2 (B, 3).
            t (torch.Tensor): Time step embedding.

        Returns:
            pred_pose (torch.Tensor): Predicted 2D poses from 2 views (B, T, J, 3*2).
            pred_quat (torch.Tensor): Predicted quaternion (B, 4).
            pred_trans (torch.Tensor): Predicted translation (B, 3).

        """
        bs, frames, joints, _ = x.shape
        x = self.input_process(x, quat, trans)  # (B, T * 2, latent_dim)
        if t is not None:
            time_embed = self.time_mlp(t)  # (B, D)
            time_embed = time_embed.unsqueeze(1)  # (B, 1, D)
            time_embed = time_embed.expand(bs, x.size(1), -1)  # (B, T * 2, D)
            x = x + time_embed  # (B, T * 2, latent_dim)

        x = self.seqTransEncoder(x)  # (B, T * 2, latent_dim)
        x = self.head(x)  # (B, T * 2, J * 3 + 7)
        pred_pose = x[:, :, : joints * 3].reshape(bs, frames * 2, joints, 3)  # (B, T * 2, J, 3)
        pred_pose1 = pred_pose[:, :frames, :, :]  # (B, T, J, 3)
        pred_pose2 = pred_pose[:, frames:, :, :]  # (B, T, J, 3)
        pred_pose = torch.cat([pred_pose1, pred_pose2], dim=-1)

        pred_cam_params = x[:, frames:, joints * 3 :].mean(dim=1)  # (B, 7)
        pred_quat = pred_cam_params[:, :4]
        pred_quat = F.normalize(pred_quat, dim=1)
        mask = pred_quat[..., 0] < 0
        pred_quat[mask] = -pred_quat[mask]

        pred_trans = pred_cam_params[:, 4:]
        pred_trans = F.normalize(pred_trans, dim=1)

        return pred_pose, pred_quat, pred_trans
