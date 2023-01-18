from typing import Tuple

import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (3, 128, 128)):
        super(Decoder, self).__init__()

        # Layer parameters
        kernel_size = 4

        self.image_size = image_size
        self.decoder_initial_size = (64, kernel_size, kernel_size)

        n_channels = self.image_size[0] + 1

        # Convolutional layers
        cnn_kwargs = dict(kernel_size=kernel_size, stride=2, padding=1)

        self.convT5 = nn.ConvTranspose2d(64, 64, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(64, 64, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(64, 64, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(64, 64, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(64, n_channels, **cnn_kwargs)

        self.activation = torch.nn.ReLU()

    def forward(self, x):

        x = self.activation(self.convT5(x))
        x = self.activation(self.convT4(x))
        x = self.activation(self.convT3(x))
        x = self.activation(self.convT2(x))
        x = self.convT1(x)

        return x
