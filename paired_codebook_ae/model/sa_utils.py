import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    grid = np.concatenate([grid, 1.0 - grid], axis=-1)
    return grid


class SoftPositionEmbed(pl.LightningModule):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        in_features = 4
        self.linear = nn.Linear(in_features, out_features=hidden_size)
        self.grid = nn.Parameter(torch.Tensor(build_grid(resolution)), requires_grad=False)

    def forward(self, inputs):
        pos_embed = self.linear(self.grid)
        pos_embed = pos_embed.moveaxis(3, 1)
        return inputs + pos_embed


def spatial_flatten(x):
    """ Flatten image with shape (-1, num_channels, width, height)
    to shape of (-1, width * height, num_channels)"""
    x = torch.swapaxes(x, 1, -1)
    return x.reshape(-1, x.shape[1] * x.shape[2], x.shape[-1])


def spatial_broadcast(x, resolution):
    # x -> (batch_size, num_slots, slot_size)

    slot_size = x.shape[-1]
    x = x.reshape(-1, slot_size, 1, 1)
    x = x.expand(-1, slot_size, *resolution)
    return x


def unstack_and_split(x, batch_size, num_slots, in_channels=3):
    unstacked = x.reshape(batch_size, num_slots, *x.shape[1:])
    channels, masks = torch.split(unstacked, in_channels, dim=2)
    return channels, masks


if __name__ == '__main__':
    grid = build_grid((4, 5))

    print("Done")
