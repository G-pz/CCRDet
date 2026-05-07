# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet

from .c2former_resnet import C2FormerResNet
from .hrformer import HRFormer
from .hrfuser_hrformer_based import HRFuserHRFormerBased
from .cft_resnet import CFTResNet
from .c2former_cam_resnet import C2FormercamResNet
from .a import AMFusionResNet

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet',
    'C2FormerResNet', 'HRFormer', 'HRFuserHRFormerBased', 'CFTResNet',
    'C2FormercamResNet', 'C2FormerResNet1', 'LCAFusionResNet', 'STFFusionResNet', 'AMFusionResNet'
]
