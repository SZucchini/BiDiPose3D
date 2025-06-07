import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from bidipose.datasets.datamodule import StereoCameraDataModule
import bidipose.models as models
from bidipose.diffusion.sampler import DDPMSampler
from bidipose.diffusion.module import DiffusionLightningModule

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

# You define detailed parameters in each subdirectory's yaml file.
@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Main training routine using Hydra for hyperparameter management.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # DataModule
    datamodule = StereoCameraDataModule(
        data_root=cfg.local.data_root,
        data_name=cfg.data.data_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
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