import torch.nn as nn
import math
from ..myLayer.block_cir_matmul import NewBlockCirculantConv,LearnableCir,LearnableCirBN

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class CirNasInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,feature_size,pretrain,finetune):
        super(CirNasInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.finetune = finetune
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                LearnableCirBN(inp, hidden_dim, 1, 1,feature_size,pretrain,self.finetune),
                # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(inp, hidden_dim, 1, 1, self.block_size),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                LearnableCirBN(hidden_dim, oup, 1, 1,feature_size//stride,pretrain,self.finetune),
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(hidden_dim, oup, 1, 1, self.block_size),
                # nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CirNasInvertedResidualImagenet(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,feature_size,pretrain,finetune):
        super(CirNasInvertedResidualImagenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.finetune = finetune
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                LearnableCir(inp, hidden_dim, 1, 1,feature_size,pretrain,self.finetune),
                # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(inp, hidden_dim, 1, 1, self.block_size),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                LearnableCir(hidden_dim, oup, 1, 1,feature_size//stride,pretrain,self.finetune),
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(hidden_dim, oup, 1, 1, self.block_size),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CirNasInvertedResidualFixBlockSize(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,feature_size,pretrain,finetune,block_size):
        super(CirNasInvertedResidualFixBlockSize, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.finetune = finetune
        self.pretrain = pretrain
        self.feature_size = feature_size
        self.block_size = block_size
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                LearnableCir(inp, hidden_dim, 1, 1,feature_size,pretrain,self.finetune,fix_block_size=block_size),
                # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(inp, hidden_dim, 1, 1, self.block_size),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                LearnableCir(hidden_dim, oup, 1, 1,feature_size//stride,pretrain,self.finetune,fix_block_size=block_size),
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # NewBlockCirculantConv(hidden_dim, oup, 1, 1, self.block_size),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CirNasMobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0,pretrain=False,finetune=False):
        super(CirNasMobileNetV2, self).__init__()
        if n_class == 1000:
            block = CirNasInvertedResidualImagenet
        else:
            block = CirNasInvertedResidual
        input_channel = 32
        last_channel = 1280
        self.feature_size = input_size
        self.pretrain = pretrain
        self.finetune = finetune
        if input_size == 224:
            self.feature_size = input_size/2
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        if n_class != 10:
            self.features = [conv_bn(3, input_channel, 2)]
        else:
            self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        idx=0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,feature_size=self.feature_size,pretrain=self.pretrain,finetune=self.finetune))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,feature_size=self.feature_size,pretrain=self.pretrain,finetune=self.finetune))
                input_channel = output_channel
            self.feature_size = self.feature_size//s
            idx+=1
        # building last several layers
        self.features = nn.Sequential(*self.features)
        self.conv = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __str__(self):
        additional_info = "pretrain: " + str(self.pretrain)+"\n"+"fine-tune: "+str(self.finetune)
        return super(CirNasMobileNetV2, self).__str__() + "\n" + additional_info


class CirNasMobileNetV2FixBlockSize(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0,pretrain=False,finetune=False,block_size=1):
        super(CirNasMobileNetV2FixBlockSize, self).__init__()
        block = CirNasInvertedResidualFixBlockSize
        input_channel = 32
        last_channel = 1280
        self.feature_size = input_size
        self.pretrain = pretrain
        self.finetune = finetune
        if input_size == 224:
            self.feature_size = input_size/2
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        if n_class != 10:
            self.features = [conv_bn(3, input_channel, 2)]
        else:
            self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        idx=0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,feature_size=self.feature_size,pretrain=self.pretrain,finetune=self.finetune,block_size=block_size))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,feature_size=self.feature_size,pretrain=self.pretrain,finetune=self.finetune,block_size=block_size))
                input_channel = output_channel
            self.feature_size = self.feature_size//s
            idx+=1

        self.features = nn.Sequential(*self.features)
        self.conv = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __str__(self):
        additional_info = "pretrain: " + str(self.pretrain)+"\n"+"fine-tune: "+str(self.finetune)
        return super(CirNasMobileNetV2FixBlockSize, self).__str__() + "\n" + additional_info

    
def cir_nas_mobilenet(n_class, input_size, width_mult,pretrain,finetune) -> CirNasMobileNetV2:
    model = CirNasMobileNetV2(n_class=n_class, input_size=input_size, width_mult=width_mult,pretrain=pretrain,finetune=finetune)
    return model

def cir_nas_mobilenet_fix(n_class, input_size, width_mult,pretrain,finetune,block_size=-1) -> CirNasMobileNetV2FixBlockSize:
    model = CirNasMobileNetV2FixBlockSize(n_class=n_class, input_size=input_size, width_mult=width_mult,pretrain=pretrain,finetune=finetune,block_size=block_size)
    return model