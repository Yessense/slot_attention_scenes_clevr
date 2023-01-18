import pathlib
from typing import Any, Optional, List
from torchmetrics.image.fid import FrechetInceptionDistance

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler

from paired_codebook_ae.dataset._dataset_info import DatasetInfo
from paired_codebook_ae.dataset.paired_clevr import PairedClevr
from .attention import AttentionModule
from .exchange import ExchangeModule
from ..utils import iou_pytorch
from ..codebook.codebook import Codebook
from ..config import VSADecoderConfig
from .binder import Binder, FourierBinder
from .decoder import Decoder
from .encoder import Encoder


class FeatureExchange(pl.LightningModule):
    binder: Binder
    cfg: VSADecoderConfig
    dataset_info: DatasetInfo

    def __init__(self, cfg: VSADecoderConfig, dataset_info: DatasetInfo):
        super().__init__()
        self.cfg = cfg
        self.dataset_info = dataset_info

        features = Codebook.make_features_from_dataset(dataset_info)
        if cfg.dataset.requires_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)

        self.encoder = Encoder(image_size=cfg.dataset.image_size,
                               latent_dim=cfg.model.latent_dim,
                               hidden_channels=cfg.model.encoder_config.hidden_channels)

        self.decoder = Decoder(image_size=cfg.dataset.image_size,
                               latent_dim=cfg.model.latent_dim,
                               in_channels=cfg.model.decoder_config.in_channels,
                               hidden_channels=cfg.model.decoder_config.hidden_channels)
        self.codebook = Codebook(features=features,
                                 latent_dim=cfg.model.latent_dim,
                                 seed=cfg.experiment.seed)

        self.attention = AttentionModule(vsa_features=self.codebook.vsa_features,
                                         n_features=cfg.dataset.n_features,
                                         latent_dim=cfg.model.latent_dim,
                                         scale=None)
        self.exchange_module = ExchangeModule()
        self.loss_f = F.mse_loss

        if cfg.model.binder == 'fourier':
            self.binder = FourierBinder(placeholders=self.codebook.placeholders)
        else:
            raise NotImplemented(f"Wrong binder type {cfg.model.binder}")

        self.save_hyperparameters()

    def forward(self, *batch):
        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        image_latent = self.encoder(image)
        donor_latent = self.encoder(donor)

        image_features, image_max_values = self.attention(image_latent)
        donor_features, donor_max_values = self.attention(donor_latent)

        image_with_same_donor_elements, donor_with_same_image_elements = self.exchange_module(
            image_features, donor_features, exchange_labels)

        image_like_binded = self.binder(image_with_same_donor_elements)
        donor_like_binded = self.binder(donor_with_same_image_elements)

        recon_image_like = self.decoder(torch.sum(image_like_binded, dim=1))
        recon_donor_like = self.decoder(torch.sum(donor_like_binded, dim=1))

        image_loss = self.loss_f(recon_image_like, image, reduction=self.cfg.experiment.reduction)
        donor_loss = self.loss_f(recon_donor_like, donor, reduction=self.cfg.experiment.reduction)
        total_loss = (image_loss + donor_loss) * 0.5  # + self.kld_coef * kld_loss

        recons = (recon_image_like, recon_donor_like)
        features = (image_like_binded, donor_like_binded)
        return recons, features


if __name__ == '__main__':
    image_size = (3, 128, 128)
    model = FeatureExchange(VSADecoderConfig(), dataset_info=PairedClevr.dataset_info)

    batch_size = 10
    n_features = 6
    image = torch.randn((batch_size, 3, 128, 128))
    donor = torch.randn((batch_size, 3, 128, 128))
    image_labels = torch.zeros((batch_size, n_features))
    donor_labels = torch.zeros((batch_size, n_features))
    exchange_labels = torch.zeros((batch_size, n_features)).unsqueeze(-1).bool()

    answer = model.forward((image, donor), (image_labels, donor_labels), exchange_labels)

    print()
