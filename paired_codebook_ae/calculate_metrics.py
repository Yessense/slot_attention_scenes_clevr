import os
import pathlib
from typing import Union

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from paired_codebook_ae.dataset import PairedClevrDatamodule
from .utils import find_best_model
from .model.sa_fe_model import SlotAttentionFeatureSwap
from .config import VSADecoderConfig

cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)

path_to_dataset = pathlib.Path().absolute()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)
    # cfg.metrics.metrics_dir = "/home/yessense/data/paired_codebook_ae/outputs/2023-01-15/21-52-26"

    datamodule: Union[PairedClevrDatamodule] = instantiate(cfg.dataset.datamodule)

    if not cfg.metrics.ckpt_path:
        cfg.metrics.ckpt_path = find_best_model(
            os.path.join(cfg.metrics.metrics_dir, "checkpoints"))

    # print(cfg.metrics.ckpt_path)

    model = SlotAttentionFeatureSwap.load_from_checkpoint(cfg.metrics.ckpt_path)

    wandb_logger = WandbLogger(
        project=f"metrics_{cfg.dataset.datamodule.mode}_scenes",
        name=f'{cfg.dataset.datamodule.mode} -l {cfg.model.latent_dim} '
             f'-s {cfg.experiment.seed} '
             f'-bs {cfg.experiment.batch_size} '
             f'vsa',
        save_dir=cfg.experiment.logging_dir)

    # trainer
    trainer = pl.Trainer(accelerator=cfg.experiment.accelerator,
                         devices=cfg.experiment.devices,
                         profiler=cfg.experiment.profiler,
                         logger=wandb_logger,
                         )

    trainer.test(model,
                 datamodule=datamodule)


if __name__ == '__main__':
    main()
