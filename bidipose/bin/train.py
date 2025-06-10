import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pathlib import Path
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import bidipose.models as models
from bidipose.datasets.datamodule import StereoCameraDataModule
from bidipose.diffusion.module import DiffusionLightningModule
from bidipose.diffusion.sampler import DDPMSampler

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
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training routine using Hydra for hyperparameter management.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    """
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()[:8]
    except Exception:
        commit_hash = "nogit"

    overrides_path = Path(cfg.base_dir) / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        overrides = overrides_path.read_text().strip().splitlines()
        clean_overrides = [o.strip("- ") for o in overrides]
        overrides_str = "_".join(clean_overrides)
        exp_name = f"{commit_hash}-{overrides_str}"
    else:
        exp_name = commit_hash

    print(f"Experiment Name: {exp_name}")

    # fix seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # DataModule
    datamodule = StereoCameraDataModule(
        h36m_root=cfg.local.h36m_root,
        hml3d_root=cfg.local.hml3d_root,
        data_name=cfg.data.data_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = getattr(models, cfg.model.name)(**cfg.model.params)

    # Sampler
    sampler = DDPMSampler(
        beta_scheduler_name=cfg.sampler.beta_scheduler_name,
        beta_scheduler_params=cfg.sampler.beta_scheduler_params,
        device=cfg.sampler.device,
    )

    # Lightning Module
    module = DiffusionLightningModule(model=model, sampler=sampler, **cfg.module)

    # WandbLogger
    wandb_logger = WandbLogger(project=cfg.logger.project, name=exp_name, save_dir=cfg.logger.save_dir)

    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)

    # Trainer
    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **cfg.trainer)

    # Start training
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
