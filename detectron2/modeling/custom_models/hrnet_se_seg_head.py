from typing import Callable, Dict, Optional, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec
from ..meta_arch import SEM_SEG_HEADS_REGISTRY


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegHRNetHead(nn.Module):
    """
    HRNET head adding the upsampling part from
    https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
        balance_weight: list,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight
        self._num_classes = num_classes
        self.balance_weight = balance_weight

        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "balance_weight": cfg.LOSS.BALANCE_WEIGHTS,
        }

    def forward(self, features, targets=None, masks=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = features
        if self.training:
            if masks is not None:
                return None, self.losses(x, targets, masks)
            else:
                return None, self.losses(x, targets)
        else:
            if len(x) == 2:
                x = x[1]
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            print(self.scale_heads[i](features[f]).shape)
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets, masks=None):
        if len(predictions) > 1 and len(predictions) == len(self.balance_weight):
            predictionsA = predictions[0].float()  # https://github.com/pytorch/pytorch/issues/48163
            predictionsB = predictions[1].float()
            targets[targets > (self._num_classes)] = self.ignore_value
            predictionsA = F.interpolate(
                    predictionsA, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            predictionsB = F.interpolate(
                    predictionsB, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            if masks is not None:
                for idx in range(len(predictionsA)):
                    aux_mask = masks[idx].unsqueeze(0).expand(predictionsA[idx].size())
                    predictionsA[idx] = predictionsA[idx] * aux_mask
                    aux_mask = masks[idx].unsqueeze(0).expand(predictionsB[idx].size())
                    predictionsB[idx] = predictionsB[idx] * aux_mask
            loss = self.balance_weight[0] * F.cross_entropy(
                    predictionsA, targets, reduction="mean", ignore_index=self.ignore_value
                ) + self.balance_weight[1] * F.cross_entropy(
                    predictionsB, targets, reduction="mean", ignore_index=self.ignore_value
                )
        else:
            predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
            targets[targets > (self._num_classes)] = self.ignore_value
            predictions = F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            if masks is not None:
                for idx in range(len(predictions)):
                    aux_mask = masks[idx].unsqueeze(0).expand(predictions[idx].size())
                    predictions[idx] = predictions[idx] * aux_mask
            loss = F.cross_entropy(
                predictions, targets, reduction="mean", ignore_index=self.ignore_value
            )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
