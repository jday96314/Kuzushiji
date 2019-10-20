import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate = 0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0 and dropout_rate <= .5:
            self.MidBlockDropout = nn.Dropout2d(p = dropout_rate)
        else:
            self.MidBlockDropout = nn.Dropout2d(p = dropout_rate//2)

        if dropout_rate > .5:
            self.InputDropout = nn.Dropout2d(p = dropout_rate//2)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        if self.dropout_rate <= .5:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(self.InputDropout(x))))

        if self.dropout_rate > 0:
            out = self.MidBlockDropout(out)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_rate = 0): # TODO: Hook up dropout.
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.dropout_rate = dropout_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        live_prob = .5
        if self.training:
            if random.random() < live_prob:
                out = F.elu(self.bn1(self.conv1(x)))
                out = F.dropout2d(out, p = self.dropout_rate/2, training = self.training)
                out = F.elu(self.bn2(self.conv2(out)))
                out = F.dropout2d(out, p = self.dropout_rate/2, training = self.training)
                out = self.bn3(self.conv3(out))
                out += self.shortcut(x)
                out = F.elu(out)
            else:
                out = self.shortcut(x)
                out = F.elu(out)
        else:
            out = F.elu(self.bn1(self.conv1(x)))
            out = F.dropout2d(out, p = self.dropout_rate/2, training = self.training)
            out = F.elu(self.bn2(self.conv2(out)))
            out = F.dropout2d(out, p = self.dropout_rate/2, training = self.training)
            out = self.bn3(self.conv3(out)) * live_prob
            out += self.shortcut(x)
            out = F.elu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, image_channels, filter_count_coef, dropout_rate):
        super(ResNet, self).__init__()
        self.in_planes = filter_count_coef

        self.conv1 = nn.Conv2d(image_channels, filter_count_coef, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_count_coef)
        self.layer1 = self._make_layer(block, filter_count_coef, num_blocks[0], stride=1, dropout_rate = dropout_rate)
        self.layer2 = self._make_layer(block, filter_count_coef*2, num_blocks[1], stride=2, dropout_rate = dropout_rate)
        self.layer3 = self._make_layer(block, filter_count_coef*4, num_blocks[2], stride=2, dropout_rate = dropout_rate)
        self.layer4 = self._make_layer(block, filter_count_coef*8, num_blocks[3], stride=2, dropout_rate = dropout_rate)
        self.FinalChannelsCount = filter_count_coef * 8 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # return out


def ResNet18(image_channels, filter_count_coef = 64, dropout_rate = 0):
    return ResNet(BasicBlock, [2,2,2,2], image_channels, filter_count_coef, dropout_rate)

def ResNet34(image_channels, filter_count_coef = 64, dropout_rate = 0):
    return ResNet(BasicBlock, [3,4,6,3], image_channels, filter_count_coef, dropout_rate)

def ResNet50(image_channels, filter_count_coef = 64, dropout_rate = 0):
    return ResNet(Bottleneck, [3,4,6,3], image_channels, filter_count_coef, dropout_rate)

def ResNet101(image_channels, filter_count_coef = 64, dropout_rate = 0):
    return ResNet(Bottleneck, [3,4,23,3], image_channels, filter_count_coef, dropout_rate)

def ResNet152(image_channels, filter_count_coef = 64, dropout_rate = 0):
    return ResNet(Bottleneck, [3,8,36,3], image_channels, filter_count_coef, dropout_rate)
