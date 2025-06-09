import logging
import os
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from bidipose.diffusion.sampler import DDPMSampler
from bidipose.diffusion.utils import get_camera_mask, get_spatial_mask, get_temporal_mask
from bidipose.eval.metrics import camera_direction_error, camera_rotation_error, epipolar_error, key_point_error_2d
from bidipose.models.base import BaseModel
from bidipose.statics.joints import h36m_joints_name_to_index
from bidipose.visualize.animation_2d import vis_pose2d
from bidipose.visualize.animation_3d import vis_pose3d


class DiffusionLightningModule(pl.LightningModule):
    """PyTorch LightningModule for training a diffusion model.

    Args:
        model (nn.Module): Diffusion model.
        lr (float): Learning rate.
        betas (tuple): Adam optimizer betas.

    """

    def __init__(
        self,
        model: BaseModel,
        sampler: DDPMSampler,
        optimizer_name: str,
        optimizer_params: dict,
        num_validation_batches_to_sample: int = 2,
        num_validation_batches_to_inpaint: int = 2,
        num_plot_sample: int = 2,
        num_plot_inpaint: int = 2,
        inpainting_spatial_name: List[str] = None,
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
        self.inpainting_spatial_index = (
            [h36m_joints_name_to_index[name] for name in inpainting_spatial_name]
            if inpainting_spatial_name is not None
            else None
        )
        self.inpainting_temporal_interval = inpainting_temporal_interval
        self.inpainting_camera_index = inpainting_camera_index

    def forward(
        self, x: torch.Tensor, quat: torch.Tensor, trans: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

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

    def criterion(self, batch: Any) -> torch.Tensor:
        """Compute the loss for a batch.

        Args:
            batch (Any): Batch data.

        Returns:
            torch.Tensor: Loss value.

        """
        x, quat, trans = batch
        x = x.to(self.device)
        quat = quat.to(self.device)
        trans = trans.to(self.device)
        t = torch.randint(0, self.sampler.timesteps, (x.size(0),), device=x.device)
        x_noisy, quat_noisy, trans_noisy = self.sampler.q_sample(x, quat, trans, t)
        x_pred, quat_pred, trans_pred = self.forward(x_noisy, quat_noisy, trans_noisy, t)
        gt = torch.cat(
            [
                x.flatten(1),
                quat.flatten(1),
                trans.flatten(1),
            ],
            dim=-1,
        )
        noisy = torch.cat(
            [
                x_noisy.flatten(1),
                quat_noisy.flatten(1),
                trans_noisy.flatten(1),
            ],
            dim=-1,
        )
        pred = torch.cat(
            [
                x_pred.flatten(1),
                quat_pred.flatten(1),
                trans_pred.flatten(1),
            ],
            dim=-1,
        )
        noise_gt = self.sampler.clean_to_noise(gt, noisy, t)
        if self.sampler.predict_x0:
            noise_pred = self.sampler.clean_to_noise(pred, noisy, t)
        else:
            noise_pred = pred
        loss = self.loss_fn(noise_pred, noise_gt)
        return loss

    def sample(
        self,
        x_shape: Tuple[int, ...],
        quat_shape: Tuple[int, ...],
        trans_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the diffusion model.

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
        """Inpaint missing data using the diffusion model.

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
        """Training step.

        Args:
            batch (Any): Batch data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.

        """
        loss = self.criterion(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step.

        Args:
            batch (Any): Batch data.
            batch_idx (int): Batch index.

        """
        loss = self.criterion(batch)
        self.log("val/loss", loss)
        # Save a few batches for later use
        if len(self.validation_batches) < self.num_validation_batches_to_inpaint:
            # Detach tensors to avoid memory leak
            detached_batch = tuple(tensor.detach().cpu() for tensor in batch)
            self.validation_batches.append(detached_batch)

    def _process_data_for_logging(
        self,
        x: Optional[list[torch.Tensor]] = None,
        x_gt: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x is not None or x_gt is not None, "At least one of x or x_gt must be provided."
        if x is not None:
            x = torch.cat(x, dim=0)
            x = x.detach().cpu().numpy()
        if x_gt is not None:
            x_gt = torch.cat(x_gt, dim=0)
            x_gt = x_gt.detach().cpu().numpy()
            if x is None:
                x = x_gt
        else:
            x_gt = [None] * x.shape[0]
        return x, x_gt

    def _log_animation(
        self,
        key1: str,
        key2: str,
        x: Optional[list[torch.Tensor]] = None,
        quat: Optional[list[torch.Tensor]] = None,
        trans: Optional[list[torch.Tensor]] = None,
        x_gt: Optional[list[torch.Tensor]] = None,
        quat_gt: Optional[list[torch.Tensor]] = None,
        trans_gt: Optional[list[torch.Tensor]] = None,
    ) -> int:
        x, x_gt = self._process_data_for_logging(x, x_gt)
        quat, quat_gt = self._process_data_for_logging(quat, quat_gt)
        trans, trans_gt = self._process_data_for_logging(trans, trans_gt)

        save_root = self.trainer.default_root_dir
        media_dir = os.path.join(save_root, "media")
        os.makedirs(media_dir, exist_ok=True)

        paths_2d = []
        paths_3d = []
        logging.info(f"Saving animations for key: {key1}/{key2} at epoch {self.current_epoch}")
        for i, (s_x, s_quat, s_trans, s_x_gt, s_quat_gt, s_trans_gt) in enumerate(
            zip(x, quat, trans, x_gt, quat_gt, trans_gt, strict=False)
        ):
            logging.info(f"Processing animation {i + 1}/{len(x)} for key: {key1}/{key2}")
            # Log 2D pose animation
            filename_2d = (
                f"2d_{key1.replace('/', '_')}_{key1.replace('/', '_')}_epoch_{self.current_epoch:03d}_{i}.mp4"
            )
            video_path_2d = os.path.join(media_dir, filename_2d)
            ani = vis_pose2d(
                pred_pose=s_x,
                gt_pose=s_x_gt,
            )
            ani.save(video_path_2d, writer="ffmpeg", fps=30)
            paths_2d.append(video_path_2d)

            # Log 3D pose animation
            filename_3d = (
                f"3d_{key1.replace('/', '_')}_{key2.replace('/', '_')}_epoch_{self.current_epoch:03d}_{i}.mp4"
            )
            video_path_3d = os.path.join(media_dir, filename_3d)
            ani = vis_pose3d(
                pred_pose=s_x,
                pred_quat=s_quat,
                pred_trans=s_trans,
                gt_pose=s_x_gt,
                gt_quat=s_quat_gt,
                gt_trans=s_trans_gt,
            )
            ani.save(video_path_3d, writer="ffmpeg", fps=30)
            paths_3d.append(video_path_3d)
        logging.info(f"Saved {len(paths_2d)} 2D and {len(paths_3d)} 3D animations for key: {key1}/{key2}")

        # Log the paths to the videos
        logging.info(f"Logging videos for key: {key1}/{key2} at epoch {self.current_epoch}")
        self.logger.log_video(
            key=f"{key1}/2d/{key2}", videos=paths_2d, step=self.current_epoch, format=["mp4"] * len(paths_2d)
        )
        self.logger.log_video(
            key=f"{key1}/3d/{key2}", videos=paths_3d, step=self.current_epoch, format=["mp4"] * len(paths_2d)
        )
        logging.info(f"Logged {len(paths_2d)} 2D and {len(paths_3d)} 3D animations for key: {key1}/{key2}")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch.
        Uses the saved validation batches.
        """
        if self.trainer.sanity_checking:
            return

        if len(self.validation_batches) > 0:
            x, quat, trans = self.validation_batches[0]
            x_shape = x.shape
            quat_shape = quat.shape
            trans_shape = trans.shape

        # Validate inpainting
        plot_counter = 0
        error_dict = defaultdict(lambda: defaultdict(list))
        gt_dict = defaultdict(list)
        data_dict = defaultdict(lambda: defaultdict(list))
        logging.info(f"Validating {len(self.validation_batches)} batches for inpainting...")
        for i, batch in enumerate(self.validation_batches):
            logging.info(f"Processing validation batch {i + 1}/{len(self.validation_batches)} for inpainting...")
            x, quat, trans = batch
            x = x.to(self.device)
            quat = quat.to(self.device)
            trans = trans.to(self.device)
            num_plot = min(self.num_plot_inpaint - plot_counter, x.shape[0])
            plot_counter += num_plot

            if num_plot > 0:
                gt_dict["x_gt"].append(x[:num_plot])
                gt_dict["quat_gt"].append(quat[:num_plot])
                gt_dict["trans_gt"].append(trans[:num_plot])

            # Validate spatial inpainting
            if self.inpainting_spatial_index is not None:
                x_mask = ~get_spatial_mask(x, self.inpainting_spatial_index)
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
                epipolar_error_value = epipolar_error(x_inpainted, quat, trans, mask=~x_mask[0, :, :, 0])
                keypoint_error_value = key_point_error_2d(x_inpainted, x, mask=~x_mask)
                error_dict["epipolar_error"]["spatial"].append(epipolar_error_value.item())
                error_dict["keypoint_error"]["spatial"].append(keypoint_error_value.item())
                if num_plot > 0:
                    data_dict["spatial"]["x"].append(x_inpainted[:num_plot])

            # Validate temporal inpainting
            if self.inpainting_temporal_interval is not None:
                temporal_index = torch.arange(
                    self.inpainting_temporal_interval[0], self.inpainting_temporal_interval[1]
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
                epipolar_error_value = epipolar_error(x_inpainted, quat, trans, mask=~x_mask[0, :, :, 0])
                keypoint_error_value = key_point_error_2d(x_inpainted, x, mask=~x_mask)
                error_dict["epipolar_error"]["temporal"].append(epipolar_error_value.item())
                error_dict["keypoint_error"]["temporal"].append(keypoint_error_value.item())
                if num_plot > 0:
                    data_dict["temporal"]["x"].append(x_inpainted[:num_plot])

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
                epipolar_error_value = epipolar_error(x_inpainted, quat, trans)
                keypoint_error_value = key_point_error_2d(x_inpainted, x, mask=~x_mask)
                error_dict["epipolar_error"]["camera"].append(epipolar_error_value.item())
                error_dict["keypoint_error"]["camera"].append(keypoint_error_value.item())
                if num_plot > 0:
                    data_dict["camera"]["x"].append(x_inpainted[:num_plot])

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
            epipolar_error_value = epipolar_error(x, quat_inpainted, trans_inpainted)
            camera_direction_error_value = camera_rotation_error(quat_inpainted, quat)
            camera_translation_error_value = camera_direction_error(trans_inpainted, trans)
            error_dict["epipolar_error"]["camera_params"].append(epipolar_error_value.item())
            error_dict["camera_direction_error"]["camera_params"].append(camera_direction_error_value.item())
            error_dict["camera_translation_error"]["camera_params"].append(camera_translation_error_value.item())
            if num_plot > 0:
                data_dict["camera_params"]["quat"].append(quat_inpainted[:num_plot])
                data_dict["camera_params"]["trans"].append(trans_inpainted[:num_plot])
        logging.info(f"Processed {len(self.validation_batches)} validation batches for inpainting.")
        # Log the errors
        for key1, e_dict in error_dict.items():
            for key2, values in e_dict.items():
                if len(values) > 0:
                    mean_value = torch.tensor(values, device=self.device).mean().item()
                    self.log(f"metrics/{key1}/{key2}_inpainting", mean_value)
                    logging.info(f"Validation {key1} for {key2} inpainting: {mean_value:.4f}")
        # Log the data
        for key, data in data_dict.items():
            self._log_animation("videos", f"{key}_inpainting", **data, **gt_dict)

        # Clear the validation batches to free memory
        if len(self.validation_batches) > 0:
            logging.info(f"Processed {len(self.validation_batches)} validation batches for inpainting.")
        else:
            logging.info("No validation batches to process for inpainting.")
        self.validation_batches.clear()
        del data_dict, gt_dict, error_dict

        # Validate sampling
        plot_counter = 0
        error_list = []
        data_dict = defaultdict(list)
        logging.info(f"Validating sampling with {self.num_plot_sample} samples...")
        for i in range(self.num_plot_sample):
            logging.info(f"Processing validation sample {i + 1}/{self.num_plot_sample} for sampling...")
            num_plot = min(self.num_plot_sample - plot_counter, x_shape[0])
            plot_counter += num_plot

            x_sample, quat_sample, trans_sample = self.sample(
                x_shape=x_shape,
                quat_shape=quat_shape,
                trans_shape=trans_shape,
            )
            epipolar_error_value = epipolar_error(x_sample, quat_sample, trans_sample)
            error_list.append(epipolar_error_value.item())

            if num_plot > 0:
                data_dict["x"].append(x_sample[:num_plot])
                data_dict["quat"].append(quat_sample[:num_plot])
                data_dict["trans"].append(trans_sample[:num_plot])

        logging.info(f"Processed {self.num_plot_sample} samples for validation sampling.")

        # Log the sampling error
        if len(error_list) > 0:
            mean_error = torch.tensor(error_list, device=self.device).mean().item()
            self.log("metrics/epipolar_error/sampling", mean_error)
            logging.info(f"Validation epipolar_error for sampling: {mean_error:.4f}")
        # Log the sampled data
        if len(data_dict["x"]) > 0:
            self._log_animation("videos", "sampling", **data_dict)
        # Clear the data dictionary to free memory
        if len(data_dict) > 0:
            logging.info(f"Processed {len(data_dict['x'])} samples for validation sampling.")
        else:
            logging.info("No samples to process for validation sampling.")
        data_dict.clear()
        del data_dict, error_list

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizers for training.

        Returns:
            optim.Optimizer: Adam optimizer.

        """
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), **self.optimizer_params)
        return optimizer
