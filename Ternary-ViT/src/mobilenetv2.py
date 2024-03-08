import imp
import sys


import torch.nn as nn

sys.path.append("/home/xts/code/pytorch-image-models")

from functools import partial
from src.mobilenetv2_tiny import mobilenet_tiny
from src.mobilenetv2_cifar import mobilenet_cifar
from src.mobilenetv2_imagenet import get_mbv2_imagenet
from src.efficientnet_lite import build_efficientnet_lite
from src.efficientnet_lite_cifar import build_efficientnet_lite_cifar
from src.myModel.MobilenetV1 import MobileNet,CirMobileNet
from src.cir_mobilenetv2_cifar import cir_mobilenet_cifar
from src.cir_mobilenetv2_imagenet import get_cir_mbv2_imagenet
from src.myModel.cir_resnet import get_cir_resnet18_cifar
from src.myModel.cir_nas_mbv2 import cir_nas_mobilenet,cir_nas_mobilenet_fix
from timm.models.layers import create_conv2d
from timm.models.registry import register_model
from timm.models.efficientnet import _create_effnet
from timm.models.efficientnet_builder import (
    decode_arch_def,
    resolve_bn_args,
    resolve_act_layer,
    round_channels,
)


__all__ = ['cifar10_mobilenetv2_100']


def _gen_cifar10_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    # normal
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s1_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    # c*1.2
    # arch_def = [
    #     ['ds_r1_k3_s1_c19'], # 24
    #     ['ir_r2_k3_s1_e6_c28'], # 32
    #     ['ir_r3_k3_s2_e6_c38'], #40
    #     ['ir_r4_k3_s2_e6_c76'], #80
    #     ['ir_r3_k3_s1_e6_c115'], #112
    #     ['ir_r3_k3_s2_e6_c192'], #192
    #     ['ir_r1_k3_s1_e6_c384'], #384
    # ]
    # each layer's latency equals
    # arch_def = [
    #     ['ds_r1_k3_s1_c16'], # 16
    #     ['ir_r2_k3_s1_e6_c24'], # 24
    #     ['ir_r3_k3_s2_e6_c36'], #36
    #     ['ir_r4_k3_s2_e6_c76'], #76
    #     ['ir_r3_k3_s1_e6_c120'], #120
    #     ['ir_r3_k3_s2_e6_c224'], #224
    #     ['ir_r1_k3_s1_e6_c448'], #448
    # ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    print(model_kwargs['block_args'])
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model

def _gen_tiny_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    # normal
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    print(model_kwargs['block_args'])
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


@register_model
def cifar10_mobilenetv2_100(pretrained=False, **kwargs):
    model = _gen_cifar10_mobilenet_v2('mobilenetv2_100', 1.4, pretrained=pretrained, **kwargs)
    model.conv_stem = create_conv2d(
        model.conv_stem.in_channels,
        model.conv_stem.out_channels,
        3,
        stride=1,
        padding=1,
    )
    return model

@register_model
def cifar10_mobilenetv2(pretrained=False, **kwargs):
    model=mobilenet_cifar(10,32,1.0)
    return model

@register_model
def cifar100_mobilenetv2(pretrained=False, **kwargs):
    model=mobilenet_cifar(100,32,1.0)
    return model

@register_model
def tinyimagenet_mobilenetv2(pretrained=False, **kwargs):
    model=mobilenet_tiny(200,64,1.0)
    return model


@register_model
def tinyimagenet_efficient_lite(pretrained=False, **kwargs):
    model_name = 'efficientnet_lite0'
    model = build_efficientnet_lite(model_name, 200, 1.0)
    return model

@register_model
def cifar10_efficient_lite(pretrained=False, **kwargs):
    model_name = 'efficientnet_lite0'
    model = build_efficientnet_lite_cifar(model_name, 10, 1.0)
    return model

@register_model
def cifar100_efficient_lite(pretrained=False, **kwargs):
    model_name = 'efficientnet_lite0'
    model = build_efficientnet_lite_cifar(model_name, 100, 1.0)
    return model

@register_model
def cifar_cir_mobilenetv1(pretrained=False, **kwargs):
    model=CirMobileNet(10)
    return model

@register_model
def cifar_mobilenetv1(pretrained=False, **kwargs):
    model=MobileNet(10)
    return model

@register_model
def cifar10_cir_resnet18(pretrained=False, progress=True, device="gpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return get_cir_resnet18_cifar(pretrained, progress, device, **kwargs)

@register_model
def cifar_cir_mobilenetv2(pretrained=False, **kwargs):
    model=cir_mobilenet_cifar(100,32,1)
    return model

@register_model
def tiny_cir_mobilenetv2(pretrained=False, **kwargs):
    model=cir_mobilenet_cifar(200,64,1)
    return model

@register_model
def imagenet_cir_mobilenetv2(pretrained=False, **kwargs):
    model=get_cir_mbv2_imagenet(1000,1.0)
    return model

@register_model
def imagenet_mobilenetv2(pretrained=False, **kwargs):
    model=get_mbv2_imagenet(1000,1.0)
    return model
# nas
@register_model
def cifar_cir_nas_mobilenetv2(pretrained=False, **kwargs):
    model=cir_nas_mobilenet(10,32,1,pretrain=False,finetune=False)
    return model

@register_model
def pretrain_cifar_cir_nas_mobilenetv2(pretrained=False, **kwargs):
    model=cir_nas_mobilenet(10,32,1,pretrain=True,finetune=False)
    return model

@register_model
def finetune_cifar_cir_nas_mobilenetv2(pretrained=False, **kwargs):
    model=cir_nas_mobilenet(10,32,1,pretrain=False,finetune=True)
    return model

@register_model
def pretrain_imagenet_cir_nas_mobilenetv2(pretrained=False, **kwargs):
    model=cir_nas_mobilenet(1000,224,1,pretrain=True,finetune=False)
    return model

@register_model
def finetune_imagenet_cir_nas_mobilenetv2(pretrained=False, **kwargs):
    model=cir_nas_mobilenet(1000,224,1,pretrain=False,finetune=True)
    return model

@register_model
def finetune_imagenet_cir_nas_mobilenetv2_fix(pretrained=False, **kwargs):
    model=cir_nas_mobilenet_fix(1000,224,1,pretrain=False,finetune=True)
    return model

@register_model
def pretrain_imagenet_cir_nas_mobilenetv2_fix(pretrained=False, **kwargs):
    model=cir_nas_mobilenet_fix(1000,224,1,pretrain=True,finetune=False)
    return model