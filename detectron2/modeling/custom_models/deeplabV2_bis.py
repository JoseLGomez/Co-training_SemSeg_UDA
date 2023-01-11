#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from PIL import Image

from .resnet_deeplabV2_bis import _ConvBnReLU, _ResLayer, _Stem
from ..backbone import (Backbone, BACKBONE_REGISTRY) 
from ..meta_arch import SEM_SEG_HEADS_REGISTRY
from detectron2.layers import ShapeSpec


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


@BACKBONE_REGISTRY.register()
class DeepLabV2_backbone(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, cfg, input_shape):
        super(DeepLabV2_backbone, self).__init__()
        n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        n_blocks = [3, 4, 23, 3]
        atrous_rates = [6, 12, 18, 24]
        self.ch = [64 * 2 ** p for p in range(6)]
        self.size_divisibility = 0
        self.add_module("layer1", _Stem(self.ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], self.ch[0], self.ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], self.ch[2], self.ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], self.ch[3], self.ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], self.ch[4], self.ch[5], 1, 4))
        self.add_module("aspp", _ASPP(self.ch[5], n_classes, atrous_rates))

    def output_shape(self):
        return {"aspp": ShapeSpec(channels=self.ch[5], stride=1)}

    '''def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()'''


@SEM_SEG_HEADS_REGISTRY.register()
class DeepLabV2_head(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT

    def forward(self, logits, targets=None, masks=None):
        iter_loss = 0
        print(logits.shape)
        for logit in logits:
            # Resize labels for {100%, 75%, 50%, Max} logits
            _, H, W = logit.shape
            print(H)
            print(W)
            labels_ = self.resize_labels(targets, size=(W, H))
            iter_loss += self.losses(logit, labels_)

    def losses(self, predictions, targets, masks=None):
        print(predictions.shape)
        print(targets.shape)
        if masks is not None:
            for idx in range(len(predictions)):
                aux_mask = masks[idx].unsqueeze(0).expand(predictions[idx].size())
                predictions[idx] = predictions[idx] * aux_mask
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    def resize_labels(self, labels, size):
        """
        Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
        Other nearest methods result in misaligned labels.
        -> F.interpolate(labels, shape, mode='nearest')
        -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
        """
        new_labels = []
        for label in labels:
            label = label.cpu().float().numpy()
            label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
            new_labels.append(np.asarray(label))
        new_labels = torch.LongTensor(new_labels)
        return new_labels

if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)