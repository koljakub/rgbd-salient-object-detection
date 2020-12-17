# RGB-D CNN for SOD task
# ----------------------
# Source:
# * https://github.com/DengPingFan/D3NetBenchmark/blob/master/model/vgg_new.py
# * https://github.com/DengPingFan/D3NetBenchmark/blob/master/model/RgbdNet.py

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_backbone(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self, in_channel=3, pre_train_path=None):
        super(VGG_backbone, self).__init__()
        self.in_channel = in_channel
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(in_channel, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))

        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3', nn.MaxPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        if pre_train_path is not None and os.path.exists(pre_train_path):
            self._initialize_weights(torch.load(pre_train_path))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())

        torch.nn.init.kaiming_normal_(self.conv1.conv1_1.weight)
        self.conv1.conv1_1.weight.data[:, :3, :, :].copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
        self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
        self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
        self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
        self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
        self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
        self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, in_channel=32, side_channel=512):
        super(Decoder, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(side_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        init_weight(self)

    def forward(self, x, side):
        x = F.interpolate(x, size=side.size()[2:], mode='bilinear', align_corners=True)
        side = self.reduce_conv(side)
        x = torch.cat((x, side), 1)
        x = self.decoder(x)
        return x


class SingleStream(nn.Module):
    def __init__(self, in_channel=3):
        super(SingleStream, self).__init__()
        self.backbone = VGG_backbone(in_channel=in_channel, pre_train_path='./model/vgg16_feat.pth')
        self.toplayer = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        channels = [64, 128, 256, 512, 512, 32]
        # Decoders
        decoders = []
        for idx in range(5):
            decoders.append(Decoder(in_channel=32, side_channel=channels[idx]))
        self.decoders = nn.ModuleList(decoders)
        init_weight(self.toplayer)

    def forward(self, input):
        l1 = self.backbone.conv1(input)
        l2 = self.backbone.conv2(l1)
        l3 = self.backbone.conv3(l2)
        l4 = self.backbone.conv4(l3)
        l5 = self.backbone.conv5(l4)
        l6 = self.toplayer(l5)
        feats = [l1, l2, l3, l4, l5, l6]

        x = feats[5]
        for idx in [4, 3, 2, 1, 0]:
            x = self.decoders[idx](x, feats[idx])

        return x


class PredLayer(nn.Module):
    def __init__(self, in_channel=32):
        super(PredLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        init_weight(self)

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x
