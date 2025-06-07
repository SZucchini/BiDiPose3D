import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import bidipose.models as models
from bidipose.diffusion.sampler import DDPMSampler
from bidipose.diffusion.module import DiffusionLightningModule
import torch
from torch.utils.data import Dataset, DataLoader

# Here is an example directory structure for storing Hydra configuration files.
# Typically, you create a `configs` directory at the project root and place various config files inside.

# Example directory structure:
# configs/
# ├── config.yaml                # Main configuration file
# ├── data/
# │   └── data.yaml              # Dataset-related configuration
# ├── model/
# │   └── model.yaml             # Model-related configuration
# ├── sampler/
# │   └── sampler.yaml           # Sampler-related configuration
# ├── module/
# │   └── module.yaml            # LightningModule-related configuration
# ├── logger/
# │   └── logger.yaml            # Logger-related configuration
# ├── trainer/
# |   └── trainer.yaml           # Trainer-related configuration
# └── local/
#     └── local.yaml             # Local overrides for configuration

# Example config.yaml:
# defaults:
#   - data: data
#   - model: model
#   - sampler: sampler
#   - module: module
#   - logger: logger
#   - trainer: trainer
#   - local: local

class DummyStereoDataset(Dataset):
    """
    Dummy dataset for testing StereoCameraDataModule.

    Attributes:
        length (int): Number of samples in the dataset.
        T (int): Number of frames per sample.
        J (int): Number of joints per frame.
    """

    def __init__(self, length: int = 100, T: int = 81, J: int = 17) -> None:
        """
        Initialize the dummy dataset.

        Args:
            length (int): Number of samples.
            T (int): Number of frames per sample.
            J (int): Number of joints per frame.
        """
        self.length = length
        self.T = T
        self.J = J

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            x (torch.Tensor): Normalized 2D keypoints from two views (shape: (T, J, 6)).
            quat (torch.Tensor): Quaternion representing the relative pose from cam1 to cam2 (shape: (4,)).
            trans (torch.Tensor): Translation vector representing the relative pose from cam1 to cam2 (shape: (3,)).
        """
        x = torch.randn(self.T, self.J, 6)
        quat = torch.randn(4)
        quat = quat / quat.norm()  # Normalize quaternion
        trans = torch.randn(3)
        return x, quat, trans

class DummyStereoDataModule(pl.LightningDataModule):
    """
    DataModule for the DummyStereoDataset.

    Attributes:
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker processes for DataLoader.
        length (int): Number of samples in the dataset.
        T (int): Number of frames per sample.
        J (int): Number of joints per frame.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        length: int = 100,
        T: int = 81,
        J: int = 17
    ) -> None:
        """
        Initialize the DummyStereoDataModule.

        Args:
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of worker processes for DataLoader.
            length (int): Number of samples in the dataset.
            T (int): Number of frames per sample.
            J (int): Number of joints per frame.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length
        self.T = T
        self.J = J
        self.dataset: DummyStereoDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training or validation.

        Args:
            stage (str | None): Stage to setup ('fit', 'validate', etc.).
        """
        self.dataset = DummyStereoDataset(
            length=self.length,
            T=self.T,
            J=self.J
        )

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for training.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for validation.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

# You define detailed parameters in each subdirectory's yaml file.
@hydra.main(config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Main training routine using Hydra for hyperparameter management.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # DataModule
    datamodule = DummyStereoDataModule()
    
    # Model
    model = getattr(models, cfg.model.name)(**cfg.model.params)

    # Sampler
    sampler = DDPMSampler(
        beta_scheduler_name=cfg.sampler.beta_scheduler_name,
        beta_scheduler_params=cfg.sampler.beta_scheduler_params,
        device=cfg.sampler.device
    )

    # Lightning Module
    module = DiffusionLightningModule(
        model=model,
        sampler=sampler,
        **cfg.module
    )

    # WandbLogger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        save_dir=cfg.logger.save_dir
    )

    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        **cfg.trainer
    )

    # Start training
    trainer.fit(
        module,
        datamodule=datamodule
    )

if __name__ == "__main__":
    main()