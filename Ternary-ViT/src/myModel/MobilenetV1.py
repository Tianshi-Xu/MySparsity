import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from ..myLayer.block_cir_matmul import BlockCirculantConv,NewBlockCirculantConv

class CirMobileNet(nn.Module):
    def __init__(self, n_class=1000):
        super(CirMobileNet, self).__init__()
        self.nclass = n_class
        self.block_size = [16 for _ in range(13)]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride,block_size):
            
            if block_size == 1:
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
        
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
        
                    NewBlockCirculantConv(inp, oup, 1, 1, block_size=block_size),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(3, 32, 1),  # for imagenet s=2
            conv_dw(32, 64, 1,block_size=self.block_size[0]),
            conv_dw(64, 128, 2,block_size=self.block_size[1]),
            conv_dw(128, 128, 1,block_size=self.block_size[2]),
            conv_dw(128, 256, 2,block_size=self.block_size[3]),
            conv_dw(256, 256, 1,block_size=self.block_size[4]),
            conv_dw(256, 512, 2,block_size=self.block_size[5]),
            conv_dw(512, 512, 1,block_size=self.block_size[6]),
            conv_dw(512, 512, 1,block_size=self.block_size[7]),
            conv_dw(512, 512, 1,block_size=self.block_size[8]),
            conv_dw(512, 512, 1,block_size=self.block_size[9]),
            conv_dw(512, 512, 1,block_size=self.block_size[10]),
            conv_dw(512, 1024, 1,block_size=self.block_size[11]), # for imagenet s=2
            conv_dw(1024, 1024, 1,block_size=self.block_size[12]),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, self.nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    
    def __str__(self):
        additional_info = "block_size: " + str(self.block_size)
        return super(CirMobileNet, self).__str__() + "\n" + additional_info

class MobileNet(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNet, self).__init__()
        self.nclass = n_class

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 1),  # for imagenet s=2
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1), # for imagenet s=2
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, self.nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
if __name__ == '__main__':
    mobilenet = MobileNet().cuda()
    print(mobilenet)