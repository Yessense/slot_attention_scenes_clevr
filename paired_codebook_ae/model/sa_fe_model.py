import math
import os.path
from argparse import ArgumentParser
from typing import Tuple

import wandb
from torch import nn
from torch.optim import lr_scheduler

from .encoder import Encoder
from .decoder import Decoder
from .sa_autoencoder import SlotAttentionAutoEncoder
from .sa import SlotAttention

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from .feature_exchange import FeatureExchange
from .sa_utils import spatial_flatten, spatial_broadcast, unstack_and_split, \
    get_similar_image_indexes

from torchmetrics.image.fid import FrechetInceptionDistance

from ..config import VSADecoderConfig
from ..dataset.paired_clevr import PairedClevr


class SlotAttentionFeatureSwap(pl.LightningModule):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, cfg: VSADecoderConfig):
        super().__init__()

        self.cfg = cfg

        self.feature_exchange_model = FeatureExchange(cfg, dataset_info=PairedClevr.dataset_info)
        # Make slot from latent of image 1024 -> 64
        self.latent_to_slot_mlp = torch.nn.Sequential(nn.Linear(cfg.model.latent_dim, cfg.model.slot_size),
                                                      nn.ReLU(),
                                                      nn.Linear(cfg.model.slot_size, cfg.model.slot_size))

        # Slot Attention Auto Encoder
        self.slot_attention_autoencoder = SlotAttentionAutoEncoder(cfg)
        self.save_hyperparameters()

        if cfg.dataset.requires_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)

    def _step(self, batch, batch_idx, mode='Train'):
        """ Base step"""

        # Log Train batches once per epoch
        # Log Validation images triple per epoch
        # for batch_size == 64
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        elif mode == 'Test':
            log_images = lambda x: True
        else:
            raise ValueError

        # get batch
        (image, donor), (image_scene, donor_scene), exchange_labels = batch

        # exchange_labels, image, donor, image_scene, donor_scene, object_images, object_masks = batch
        batch_size = image.shape[0]

        # ----------------------------------------
        # -- Apply Slot Attention to initial scene
        # ----------------------------------------

        recon_scene_combined, recons_scene, masks_scene, slots_scene = self.slot_attention_autoencoder.forward(
            image_scene)

        # Slot Attention loss
        sa_scene_loss = F.mse_loss(recon_scene_combined, image_scene, reduction=self.cfg.experiment.reduction)
        self.log(f'{mode}/SA scene MSE', sa_scene_loss)

        # ----------------------------------------
        # -- Exchange features on training images
        # ----------------------------------------
        most_similar_image_slot = get_similar_image_indexes(image, recons_scene)

        batch_indices = torch.arange(batch_size)

        ((recon_image_like, recon_donor_like),
         (image_like_binded, donor_like_binded)
         ) = self.feature_exchange_model.forward(
            (recons_scene[batch_indices, most_similar_image_slot], donor),
            exchange_labels)

        image_loss = F.mse_loss(recon_image_like, image, reduction=self.cfg.experiment.reduction)
        donor_loss = F.mse_loss(recon_donor_like, donor, reduction=self.cfg.experiment.reduction)
        total_loss = (image_loss + donor_loss) * 0.5  # + self.kld_coef * kld_loss

        self.log(f"{mode}/FE Total", total_loss)
        self.log(f"{mode}/FE Reconstruct Image", image_loss)
        self.log(f"{mode}/FE Reconstruct Donor", donor_loss)

        # # Log reconstruction
        # if log_images(batch_idx):
        #     self.logger.experiment.log({
        #         f"{mode}/FE": [
        #             wandb.Image(image[0], caption='Image'),
        #             wandb.Image(donor[0], caption='Pair image'),
        #             wandb.Image(recon_image_like[0], caption='Recon like Image'),
        #             wandb.Image(recon_donor_like[0], caption='Recon like Pair image'),
        #         ]}, commit=False)
        #
        # ----------------------------------------
        # -- Change one slot
        # ----------------------------------------

        # most_similar_image_slot = get_similar_image_indexes(image, recons_scene)
        #
        # batch_indices = torch.arange(batch_size)
        new_image_slots = self.latent_to_slot_mlp(torch.sum(donor_like_binded, dim=1))

        slots_changed = slots_scene[:]
        slots_changed[batch_indices, most_similar_image_slot] = new_image_slots

        recon_scene_changed, recons_changed, masks_changed = self.slot_attention_autoencoder.decode(
            slots_changed)

        # Loss for reconstructed scene
        sa_changed_loss = F.mse_loss(recon_scene_changed, donor_scene, reduction=self.cfg.experiment.reduction)
        self.log(f'{mode}/SA scene changed MSE', sa_changed_loss)

        # Total loss
        triple_loss = sa_scene_loss + total_loss + sa_changed_loss
        self.log(f'{mode}/Triple loss', triple_loss)
        #
        # self.logger.experiment.log({f"experiment/scenes": [
        #     wandb.Image(image_scene[0], caption='Image Scene'),
        #     wandb.Image(donor_scene[0], caption='Pair Scene'),
        #     wandb.Image(recon_scene_combined[0], caption='Image rec'),
        #     wandb.Image(recon_scene_changed[0], caption='pair rec'),
        # ]})
        #
        return triple_loss

    def training_step(self, batch, batch_idx):
        total_loss = self._step(batch, batch_idx, mode='Train')
        return total_loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, mode='Validation')

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, mode='Test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.experiment.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.experiment.lr,
                                            epochs=self.cfg.experiment.max_epochs,
                                            steps_per_epoch=self.cfg.experiment.steps_per_epoch,
                                            pct_start=self.cfg.experiment.pct_start)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }


if __name__ == '__main__':
    sa_model = SlotAttentionFeatureSwap(cfg=VSADecoderConfig())
    device = torch.device("cuda:0")
    sa_model = sa_model.to(device)
    batch_size = 10
    n_features = 6
    image = torch.randn((batch_size, 3, 128, 128), device=device)
    exchange_labels = torch.zeros((batch_size, n_features),device=device).unsqueeze(-1).bool()

    batch = (image, image), (image, image), exchange_labels

    loss = sa_model._step(batch, 0, mode='Train')

    print("Done")
