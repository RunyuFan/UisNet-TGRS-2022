import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import MiT, ResNet, PVTv2
from backbones.layers import trunc_normal_
from heads import SegFormerHead
# from heads import SFHead
# from Deformable_ConvNet import DeformConv2D
from baseline_models import FCN, deeplabv3
from Unet import UNet
# class ConvModule(nn.Sequential):
#     def __init__(self, c1, c2, k, s=1, p=0):
#         super().__init__(
#             nn.Conv2d(c1, c2, k, s, p, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.ReLU(),
#         )


class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

segformer_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}


class SegFormer(nn.Module):
    def __init__(self, variant: str = 'B0', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(variant)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, segformer_settings[variant], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        # for i in y:
        #     print(i.shape)
        y = self.decode_head(y)   # 4x reduction in image size
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

class SOST(nn.Module):
    def __init__(self, n_class):
        super(SOST,self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.Channel_attention = ChannelAttentionModule()

        self.astr_conv0=nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2,out_channels=64,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=1,dilation=1),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5to3 = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(5, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(64*4, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)
        feature_mmdata = self.conv_block_5to3(mmdata)
        feature_mmdata = self.semantic_img_model(feature_mmdata)
        # print(features.shape)

        fuse_cat = torch.cat([features, feature_mmdata], 1)

        feature0=self.astr_conv0(fuse_cat)
        feature1=self.astr_conv1(fuse_cat)
        feature2=self.astr_conv2(fuse_cat)
        feature3=self.astr_conv3(fuse_cat)

        fuse = torch.cat([feature0, feature1, feature2, feature3], 1)

        C_out = self.Channel_attention(fuse)
        # g.pow(2).mean(1)
        C_out = self.conv_block_fc(C_out)
        out = F.interpolate(C_out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return fuse_cat.pow(2).mean(1), fuse.pow(2).mean(1), C_out.pow(2).mean(1), out

class SOST_RGB(nn.Module):
    def __init__(self, n_class):
        super(SOST_RGB,self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        print(self.semantic_img_model)
        self.semantic_img_model.decode_head.linear_pred = nn.Conv2d(768, self.out_channels*2, kernel_size=(1, 1), stride=(1, 1))
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.Channel_attention = ChannelAttentionModule()

        self.astr_conv0=nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2,out_channels=64,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.astr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=1,dilation=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.astr_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.astr_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # self.conv_block_6to3 = nn.Sequential(
        #     # FCViewer(),
        #     nn.Conv2d(6, 3, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.ReLU(inplace=True)
        #     # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        # )

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(64*4, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)
        # feature_mmdata = self.conv_block_6to3(mmdata)
        # feature_mmdata = self.semantic_mmdata_model(feature_mmdata)

        # fuse_cat = torch.cat([features, feature_mmdata], 1)

        feature0=self.astr_conv0(features)
        feature1=self.astr_conv1(features)
        feature2=self.astr_conv2(features)
        feature3=self.astr_conv3(features)

        fuse = torch.cat([feature0, feature1, feature2, feature3], 1)

        out = self.Channel_attention(fuse)

        out = self.conv_block_fc(out)
        # print(out.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class SOST_ASPP(nn.Module):
    def __init__(self, n_class):
        super(SOST_ASPP,self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        # self.Channel_attention = ChannelAttentionModule()

        self.astr_conv0=nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2,out_channels=64,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=1,dilation=1),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5to3 = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(5, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(64*4, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)
        feature_mmdata = self.conv_block_5to3(mmdata)
        feature_mmdata = self.semantic_img_model(feature_mmdata)
        # print(features.shape)

        fuse_cat = torch.cat([features, feature_mmdata], 1)

        feature0=self.astr_conv0(fuse_cat)
        feature1=self.astr_conv1(fuse_cat)
        feature2=self.astr_conv2(fuse_cat)
        feature3=self.astr_conv3(fuse_cat)

        fuse = torch.cat([feature0, feature1, feature2, feature3], 1)

        # out = self.Channel_attention(fuse)

        out = self.conv_block_fc(fuse)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class SOST_CAtt(nn.Module):
    def __init__(self, n_class):
        super(SOST_CAtt,self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.Channel_attention = ChannelAttentionModule()

        # self.astr_conv0=nn.Sequential(
        #     nn.Conv2d(in_channels=self.out_channels*2,out_channels=64,kernel_size=1,stride=1,padding=0),
        #     # nn.BatchNorm2d(self.dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.astr_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=1,dilation=1),
        #     # nn.BatchNorm2d(self.dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.astr_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
        #     # nn.BatchNorm2d(self.dim),
        #     nn.ReLU(inplace=True),
        # )
        # self.astr_conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
        #     # nn.BatchNorm2d(self.dim),
        #     nn.ReLU(inplace=True),
        # )
        self.conv_block_5to3 = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(5, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(300, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)
        feature_mmdata = self.conv_block_5to3(mmdata)
        feature_mmdata = self.semantic_img_model(feature_mmdata)
        # print(features.shape)

        fuse_cat = torch.cat([features, feature_mmdata], 1)

        # feature0=self.astr_conv0(fuse_cat)
        # feature1=self.astr_conv1(fuse_cat)
        # feature2=self.astr_conv2(fuse_cat)
        # feature3=self.astr_conv3(fuse_cat)
        #
        # fuse = torch.cat([feature0, feature1, feature2, feature3], 1)

        out = self.Channel_attention(fuse_cat)

        out = self.conv_block_fc(out)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class Segformer_baseline(nn.Module):
    def __init__(self, n_class):
        super(Segformer_baseline, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(features)
        # print(out.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class SOST_ALL(nn.Module):
    def __init__(self, n_class):
        super(SOST_ALL,self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B2', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b2.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.Channel_attention = ChannelAttentionModule()

        self.astr_conv0=nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2,out_channels=64,kernel_size=1,stride=1,padding=0),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=1,dilation=1),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.astr_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels*2, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            # nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5to3 = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(5, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        # self.head_instance = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.PPM = PPM(self.n_class*2, self.n_class)

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(self.out_channels*2, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace=True)
            # # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )
    def forward(self, h_rs, mmdata):
        # print(h_rs.shape, x_floor.shape)
        mmdata = torch.cat([h_rs, mmdata], 1)
        features = self.semantic_img_model(h_rs)
        feature_mmdata = self.conv_block_5to3(mmdata)
        feature_mmdata = self.semantic_img_model(feature_mmdata)
        # print(features.shape)

        fuse_cat = torch.cat([features, feature_mmdata], 1)

        # feature0=self.astr_conv0(fuse_cat)
        # feature1=self.astr_conv1(fuse_cat)
        # feature2=self.astr_conv2(fuse_cat)
        # feature3=self.astr_conv3(fuse_cat)
        #
        # fuse = torch.cat([feature0, feature1, feature2, feature3], 1)
        #
        # out = self.Channel_attention(fuse)

        out = self.conv_block_fc(fuse_cat)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class FPN(nn.Module):
    def __init__(self, n_class):
        super(FPN,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.backbone = ResNet('50')
        # self.backbone.load_state_dict(torch.load('resnet50.pth', map_location='cpu'))
        self.head = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.semantic_img_model = SegFormer('B3', self.out_channels)
        # self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))
        # # print(self.semantic_img_model)

    def forward(self, h_rs, mmdata):
        # x = torch.randn(2, 3, 96, 96)
        features = self.backbone(h_rs)
        out = self.head(features)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # print(out.shape)
        return out

class fcn_resnet50(nn.Module):
    def __init__(self, n_class):
        super(fcn_resnet50,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.FCN = FCN(self.n_class)

    def forward(self, h_rs, mmdata):
        out = self.FCN(h_rs)
        # print(out.shape)
        return out

class deeplabv3_resnet50(nn.Module):
    def __init__(self, n_class):
        super(deeplabv3_resnet50,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.deeplabv3 = deeplabv3(self.n_class)

    def forward(self, h_rs, mmdata):
        out = self.deeplabv3(h_rs)
        # print(out.shape)
        return out

class UNet_model(nn.Module):
    def __init__(self, n_class):
        super(UNet_model,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.UNet = UNet(3, self.n_class)

    def forward(self, h_rs, mmdata):
        out = self.UNet(h_rs)
        # print(out.shape)
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out
