import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from timm.models.registry import register_model
import pdb

__all__ = ["cifar10_reactnet"]

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2  + [512] * 6 + [1024] * 2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class BasicBlock(nn.Module):
    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1,
        use_relu=True,
        post_res_bn=False):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1= conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(inplanes)
        )

        if inplanes == planes:  #输入通道和输出通道相等
            self.conv2 = conv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        else:                   #输入通道和输出通道不等
            self.conv21 = conv1x1(inplanes, inplanes)
            self.conv22 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)

        self.relu2 = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(inplanes)
        )

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.post_res_bn = post_res_bn

        if self.inplanes != self.planes and self.stride == 2:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):

        #pdb.set_trace()
        out1 = self.conv1(x)
        if not self.post_res_bn:
            out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1
        if not self.post_res_bn:
            out1 = self.bn1(out1)

        out1 = self.relu1(out1)

        
        if self.inplanes == self.planes:
            out2 = self.conv2(out1)
            out2 = self.bn2(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2
            out2_1 = self.conv21(out1)
            out2_2 = self.conv22(out1)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.relu2(out2)
        return out2


class reactnet(nn.Module):
    def __init__(       
        self,
        num_classes=10,
        use_relu=True,
        post_res_bn=False,
        **kwargs,):
        super(reactnet, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.layer.append(firstconv3x3(3, stage_out_channel[i], 2))  #(inp, oup, stride)
            # elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64 and stage_out_channel[i] != 128:
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.layer.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2, use_relu, post_res_bn))
            else:
                self.layer.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1, use_relu, post_res_bn))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for i, block in enumerate(self.layer):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def _cifar10_reactnet(arch,**kwargs):
    model = reactnet(**kwargs)
    return model


@register_model
def cifar10_reactnet(**kwargs):
    return _cifar10_reactnet( "cifar10_resnet18", **kwargs)





