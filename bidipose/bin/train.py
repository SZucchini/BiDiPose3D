import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from bidipose.datasets.datamodule import StereoCameraDataModule
import bidipose.models as models
from bidipose.diffusion.sampler import DiffusionSampler
from bidipose.diffusion.module import DiffusionLightningModule

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """
    Main training routine using Hydra for hyperparameter management.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # DataModule
    datamodule = StereoCameraDataModule(
        data_root=cfg.data.data_root,
        data_name=cfg.data.data_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    # Model
    model = getattr(models, cfg.model.name)(**cfg.model.params)

    # Sampler
    sampler = DiffusionSampler(
        beta_scheduler_name=cfg.sampler.beta_scheduler_name,
        beta_scheduler_params=cfg.sampler.beta_scheduler_params,
        device=cfg.sampler.device
    )

    # Lightning Module
    module = DiffusionLightningModule(
        model=model,
        sampler=sampler,
        optimizer_name=cfg.trainer.optimizer_name,
        optimizer_params=cfg.trainer.optimizer_params,
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
        **cfg.trainer.trainer_params
    )

    # Start training
    trainer.fit(
        module,
        datamodule=datamodule
    )

if __name__ == "__main__":
    main()