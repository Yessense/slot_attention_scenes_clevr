from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from ..config import VSADecoderConfig
from .sa_decoder import Decoder
from .sa_encoder import Encoder
from .sa import SlotAttention
from .sa_utils import spatial_flatten, spatial_broadcast, unstack_and_split, SoftPositionEmbed


class SlotAttentionAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: VSADecoderConfig):
        super().__init__()
        # ----------------------------------------
        # -- Hyper parameters
        # ----------------------------------------
        self.cfg = cfg

        # ----------------------------------------
        # -- Layers
        # ----------------------------------------

        # Encoder
        self.encoder = Encoder(image_size=cfg.dataset.image_size)
        self.encoder_pos = SoftPositionEmbed(cfg.model.slot_size,
                                             cfg.model.encoder_config.output_size)

        # Decoder
        self.decoder = Decoder(image_size=cfg.dataset.image_size)
        self.decoder_pos = SoftPositionEmbed(cfg.model.slot_size,
                                             cfg.model.decoder_config.initial_size)

        # Spatial MLP
        self.mlp = torch.nn.Sequential(nn.Linear(cfg.model.slot_size, cfg.model.slot_size),
                                       nn.ReLU(),
                                       nn.Linear(cfg.model.slot_size, cfg.model.slot_size))
        # Slot attention
        self.scene_slot_attention = SlotAttention(num_iterations=cfg.model.num_iterations,
                                                  num_slots=cfg.model.num_slots,
                                                  slot_size=cfg.model.slot_size,
                                                  mlp_hidden_size=cfg.model.mlp_hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.model.slot_size)

    def encode(self, scene):
        # image -> (batch_size, num_channels, width, height)
        x_scene = self.encoder(scene)
        x_scene = self.encoder_pos(x_scene)
        x_scene = spatial_flatten(x_scene)
        # x -> (batch_size, width * height, 64)

        x_scene = self.mlp(self.layer_norm(x_scene))
        # x -> (batch_size, width * height, 64)

        slots = self.scene_slot_attention(x_scene)
        # slots -> (batch_size, num_slots, slot_size)
        return slots

    def decode(self, slots):
        batch_size = slots.shape[0]

        x_scene = spatial_broadcast(slots, self.cfg.model.decoder_config.initial_size)
        # x -> (batch_size * num_slots, slot_size, width_init, height_init)

        x_scene = self.decoder_pos(x_scene)
        x_scene = self.decoder(x_scene)
        # x -> (batch_size * num_slots, num_channels + 1, width, height)
        recons, masks = unstack_and_split(x_scene, batch_size=batch_size, num_slots=self.cfg.model.num_slots,
                                          in_channels=self.cfg.dataset.image_size[0])

        # Normalize alpha masks over slots.
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, recons, masks

    def forward(self, scene):
        slots = self.encode(scene)
        recon_combined, recons, masks = self.decode(slots)

        return recon_combined, recons, masks, slots


if __name__ == '__main__':
    sa_model = SlotAttentionAutoEncoder(cfg=VSADecoderConfig())
    device = torch.device("cuda:0")
    sa_model = sa_model.to(device)
    with torch.autograd.set_detect_anomaly(True):
        batch = torch.randn((10, 3, 128, 128), device=device)
        recon_combined, recons, masks, slots = sa_model.forward(batch)
        loss = F.mse_loss(recon_combined, batch)
        loss.backward()

    print("Done")
