from typing import Tuple

from torch import nn


class Encoder(nn.Module):
    def __init__(self,
                 image_size: Tuple[int, int, int] = (3, 128, 128)):
        """ CNN encoder,  which reduces the spatial resolution 5 times """
        super(Encoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        self.image_size = image_size
        self.out_resolution = (64, 2 * kernel_size, 2 * kernel_size)
        in_features = self.image_size[0]

        # Convolutional layers
        cnn_kwargs = dict(kernel_size=kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_features, 64, **cnn_kwargs)
        self.conv2 = nn.Conv2d(64, 64, **cnn_kwargs)
        self.conv3 = nn.Conv2d(64, 64, **cnn_kwargs)
        self.conv4 = nn.Conv2d(64, self.out_resolution[0], **cnn_kwargs)
        # Activation
        self.activation = nn.ReLU()

    def forward(self, x):
        # x -> (batch_size, 3, 128, 128)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # x -> (batch_size, 256, 4, 4)
        return x
