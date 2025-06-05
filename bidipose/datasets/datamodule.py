"""DataModule for StereoCameraDataset."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bidipose.datasets.dataset import StereoCameraDataset


class StereoCameraDataModule(pl.LightningDataModule):
    """DataModule for StereoCameraDataset."""

    def __init__(self, data_root: str, data_name: str = "H36M", batch_size: int = 32, num_workers: int = 4):
        """Initialize the StereoCameraDataModule.

        Args:
            data_root (str): Root directory of the dataset.
            data_name (str): Name of the dataset, either "H36M" or "HML3D".
            batch_size (int): Batch size for training and validation.
            num_workers (int): Number of workers for data loading.

        """
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None) -> None:
        """Set up datasets.

        Args:
            stage (str, optional): Stage of the data module. Can be "fit", "validate", or "test". Defaults to None.

        """
        if stage == "fit" or stage is None:
            self.train_dataset = StereoCameraDataset(
                data_root=self.data_root,
                data_name=self.data_name,
                split="train",
            )
            self.valid_dataset = StereoCameraDataset(
                data_root=self.data_root,
                data_name=self.data_name,
                split="test",
            )

        if stage == "validate":
            self.valid_dataset = StereoCameraDataset(
                data_root=self.data_root,
                data_name=self.data_name,
                split="test",
            )

        if stage == "test":
            self.test_dataset = StereoCameraDataset(
                data_root=self.data_root,
                data_name=self.data_name,
                split="test",
            )

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Validate dataloader."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
