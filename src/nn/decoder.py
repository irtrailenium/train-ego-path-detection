import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
        )  # kernel_size=3, stride=1, padding=1, bias=False | BatchNorm2d | ReLU
        self.conv2 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
        )  # kernel_size=3, stride=1, padding=1, bias=False | BatchNorm2d | ReLU

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UNetDecoder, self).__init__()
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = tuple([head_channels] + list(decoder_channels[:-1]))
        skip_channels = tuple(list(encoder_channels[1:]) + [0])
        out_channels = decoder_channels
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, skip_ch, out_ch)
                for in_ch, skip_ch, out_ch in zip(
                    in_channels, skip_channels, out_channels
                )
            ]
        )

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:] + [None]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x
