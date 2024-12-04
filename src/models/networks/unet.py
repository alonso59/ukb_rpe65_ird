import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.parametrize as p

class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """
    @property
    def __name__(self):
        return "unet"

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        dp=0.5,
        kernel_size: tuple = (3, 3),
        padding: int = 1,
        stride: int = 1
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start, dp, kernel_size, padding, stride)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2, dp, kernel_size, padding, stride))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear, dp, kernel_size, padding, stride))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=(1, 1)))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])

class DoubleConv(nn.Module):
    """[ Conv2d => LeakyReLU => BatchNorm (optional) ] x 2."""

    def __init__(self, in_ch: int, out_ch: int, dp: float, kernel_size: tuple,  padding: int, stride: int):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dp != 0.0:
            layers.append(nn.Dropout(dp))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int, dp=0.5, kernel_size=(3, 3),  padding: int = 1, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                 DoubleConv(in_ch, out_ch, dp, kernel_size, padding, stride))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, dp=0.5, kernel_size=(3, 3), padding=1, stride=1):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=(1, 1)),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=(2, 2), stride=(2, 2))

        self.conv = DoubleConv(in_ch, out_ch, dp, kernel_size, padding, stride)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)