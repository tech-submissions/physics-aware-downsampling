# Lint as: python3

import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils import checkpoint

batch_norm = nn.BatchNorm2d
group_norm = nn.GroupNorm
_use_group_norm = False


def get_norm_layer(num_channels):
    if _use_group_norm:
        num_channels_per_group = 16
        return nn.GroupNorm(num_channels // num_channels_per_group,
                            num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, padding_mode='replicate', groups=groups,
                     bias=bias)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=1, downsample=None,
                 groups=1, residual_block=None,
                 tag=None):
        super(BasicBlock, self).__init__()
        self.bn1 = get_norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn2 = get_norm_layer(expansion * planes)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.tag = tag
        self.sub_layer1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sub_layer2 = nn.Sequential(self.conv2, self.bn2)

    def init_block(self):
        self.conv1.weight.data.div_(100)
        if self.downsample:
            self.downsample[0].weight.data.fill_(
                1 / self.downsample[0].weight.shape[1])

    def forward(self, x):
        residual = x
        out = checkpoint.checkpoint(self.sub_layer1, x)
        out = checkpoint.checkpoint(self.sub_layer2, out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1,
                    groups=1, residual_block=None, tag=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1,
                          stride=stride, bias=False),
                get_norm_layer(out_planes),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        name = tag + '.0' if tag else None
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups,
                            residual_block=residual_block, tag=name))
        self.inplanes = out_planes
        for i in range(1, blocks):
            name = tag + '.{}'.format(i) if tag else None
            layers.append(
                block(self.inplanes, planes, expansion=expansion, groups=groups,
                      residual_block=residual_block, tag=name))

        return nn.Sequential(*layers)

    def features(self, x):
        x.requires_grad = True
        detour = self.detour(x)

        x = checkpoint.checkpoint(self.layer0, x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x += detour
        return x

    def forward(self, x):
        return self.features(x)

    def regularization_pre_step(self, model):
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.weight.grad.add_(self.weight_decay * m.weight)
        return 0


class ResNetDEM(ResNet):

    def __init__(self, inplanes=64, block=BasicBlock, residual_block=None,
                 layers=(3, 4, 6, 3), width=(64, 128, 256, 512), expansion=4,
                 groups=(1, 1, 1, 1), kernel_size=7):
        super(ResNetDEM, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=kernel_size,
                               stride=2, padding=int((kernel_size - 1) / 2),
                               padding_mode='replicate', bias=False)
        self.bn1 = get_norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.detour = nn.AvgPool2d(kernel_size=16)

        for i in range(len(layers)):
            setattr(self, 'layer%s' % str(i + 1),
                    self._make_layer(block=block, planes=width[i],
                                     blocks=layers[i], expansion=expansion,
                                     stride=1 if i in {0, 3} else 2,
                                     residual_block=residual_block,
                                     groups=groups[i]))

        self.avgpool = lambda x: x
        self.conv2 = nn.Conv2d(width[-1], 1, kernel_size=1, bias=False)
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)

    def init_model(self):
        self.conv1.weight.data.fill_(
            1 / (self.conv1.kernel_size[0] * self.conv1.kernel_size[1]))
        self.conv2.weight.data.fill_(1 / self.conv2.weight.shape[1])


def resnet(use_group_norm: bool = False) -> ResNetDEM:
    global _use_group_norm
    _use_group_norm = use_group_norm
    return ResNetDEM(inplanes=64, block=BasicBlock, layers=(2, 2, 2, 2),
                     width=(64, 128, 256, 512), expansion=1, kernel_size=3)
