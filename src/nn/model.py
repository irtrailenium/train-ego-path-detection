import math

import torch.nn as nn

from .backbone import EfficientNetBackbone, ResNetBackbone
from .decoder import UNetDecoder


class ClassificationNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        classes,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the classification method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is classified.
            classes (int): Number of classes (grid cells) for each anchor. Background class is not included.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(ClassificationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        else:
            raise NotImplementedError
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )  # stride=1, padding=0
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * (classes + 1) * 2),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        clf = self.fc(fea)
        return clf


class RegressionNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the regression method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is regressed.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(RegressionNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(version=backbone[6:], pretrained=pretrained)
        else:
            raise NotImplementedError
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )  # stride=1, padding=0
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * 2 + 1),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        reg = self.fc(fea)
        return reg


class SegmentationNet(nn.Module):
    def __init__(
        self,
        backbone,
        decoder_channels,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the segmentation method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            decoder_channels (tuple): Number of output channels of each decoder block.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(SegmentationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.encoder = EfficientNetBackbone(
                version=backbone[13:],
                out_levels=(1, 3, 4, 6, 8),
                pretrained=pretrained,
            )
        elif backbone.startswith("resnet"):
            self.encoder = ResNetBackbone(
                version=backbone[6:],
                out_levels=(1, 2, 3, 4, 5),
                pretrained=pretrained,
            )
        else:
            raise NotImplementedError
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=1,  # binary segmentation
            kernel_size=3,
            padding=1,
        )  # stride=1

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks
