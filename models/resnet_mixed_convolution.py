import os
import warnings
import sys

from models.GA import GridAttentionBlock3D

sys.path.append(r"models")
try:
    from attentions import slice_spatial_attention_3d, se_block_3d
except:
    from .attentions import slice_spatial_attention_3d, se_block_3d
else:
    pass

warnings.filterwarnings("ignore")
import torch
import torchvision
import torch.nn as nn
from torch.nn import Sequential, Linear
from torchsummary import summary


# https://github.com/farazahmeds/Classification-of-brain-tumor-using-Spatiotemporal-models

class modifybasicstem(nn.Sequential):
    """
    用于 resnet_mixed_convolution
    """

    def __init__(self, in_channel):
        super(modifybasicstem, self).__init__(
            nn.Conv3d(in_channel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem4MRI(nn.Sequential):
    """
    用于 resnet2p1
    """

    def __init__(self, in_channel):
        super(R2Plus1dStem4MRI, self).__init__(
            nn.Conv3d(in_channel, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),

            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class basicstem(nn.Sequential):
    """
    用于 r3d_18
    """

    def __init__(self, in_channel):
        super(basicstem, self).__init__(
            nn.Conv3d(in_channel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class big_small_conv_stem(nn.Sequential):
    def __init__(self, in_channel):
        super(big_small_conv_stem, self).__init__(
            nn.Conv3d(in_channel, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )


def resnet_mixed_convolution(in_channel, num_classes, pretrained=False):
    stem = modifybasicstem(in_channel=in_channel)
    model = torchvision.models.video.mc3_18(pretrained=pretrained)
    model.stem = stem
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def resnet_mixed_convolution_big_small_conv(in_channel, num_classes, pretrained=False):
    model = torchvision.models.video.mc3_18(pretrained=pretrained)
    model.stem = big_small_conv_stem(in_channel=in_channel)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def resnet2p1(in_channel, num_classes, pretrained=False):
    stem = R2Plus1dStem4MRI(in_channel=in_channel)
    model = torchvision.models.video.r2plus1d_18(pretrained=pretrained)
    model.stem = stem
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def r3d_18(in_channel, num_classes, pretrained=False):
    stem = basicstem(in_channel=in_channel)
    model = torchvision.models.video.r3d_18(pretrained=pretrained)
    model.stem = stem
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


class resnet_mixed_conv_SSA(nn.Module):
    def __init__(self, pretrained=False):
        super(resnet_mixed_conv_SSA, self).__init__()

        # 导入模型
        model = resnet_mixed_convolution(in_channel=4, pretrained=pretrained)

        self.attention = slice_spatial_attention_3d(4, 127)  # ⭐️
        self.model = model

    def forward(self, x):
        x = self.attention(x)
        x = self.model(x)

        return x


class resnet_mixed_conv_SE(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False):
        super(resnet_mixed_conv_SE, self).__init__()

        # 导入模型
        model = resnet_mixed_convolution(in_channel=in_channel, num_classes=num_classes, pretrained=pretrained)

        # ⭐️插入注意力机制
        layer_to_insert = model.layer4
        attention = se_block_3d(channel=512, ratio=16)
        layer_to_insert.append(attention)

        self.model = model

    def forward(self, x):
        x = self.model(x)

        return x


class resnet_mixed_conv_GA(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False, is_get_att=False, need_fc=True):
        super(resnet_mixed_conv_GA, self).__init__()
        self.is_get_att = is_get_att
        self.need_fc = need_fc
        # 导入模型
        model = resnet_mixed_convolution(in_channel=in_channel, num_classes=num_classes, pretrained=pretrained)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        if need_fc:
            model.fc[1] = Linear(in_features=512 + 256, out_features=2, bias=True)
            self.fc = model.fc

        # ⭐️插入注意力机制
        self.GA = GridAttentionBlock3D(in_channels=256, gating_channels=512,
                                       inter_channels=256, sub_sample_factor=(2, 2, 2),
                                       mode='concatenation')

    def forward(self, x):
        # x = self.stem(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x_layer3 = x#.clone()
        # x = self.layer4(x)
        # # print('layer4_out=', x.shape)
        # GA_out, att = self.GA(x_layer3, x)
        # # print('GA_out=', GA_out.shape, att.shape)
        # x = torch.cat([self.avgpool(x), self.avgpool(GA_out)], 1)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = self.stem(x)
        x = self.layer1(x)  # x_layer1
        x = self.layer2(x)  # x_layer2
        x_layer3 = self.layer3(x)
        x_layer4 = self.layer4(x_layer3)
        # print('!!!!!', x_layer3.shape, x_layer4.shape)
        GA_out, att = self.GA(x_layer3, x_layer4)
        # print('222222', GA_out.shape, att.shape)
        x = torch.cat([self.avgpool(x_layer4), self.avgpool(GA_out)], 1)
        x = torch.flatten(x, 1)

        if self.need_fc:
            x = self.fc(x)

        if self.is_get_att:
            return x, att
        else:
            return x


class resnet_mixed_conv_GA2(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False):
        super(resnet_mixed_conv_GA2, self).__init__()

        # 导入模型
        model = resnet_mixed_convolution(in_channel=in_channel, num_classes=num_classes, pretrained=pretrained)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        model.fc[1] = Linear(in_features=512 + 256 + 128, out_features=2, bias=True)
        self.fc = model.fc

        # ⭐️插入注意力机制
        self.GA23 = GridAttentionBlock3D(in_channels=128, gating_channels=256,
                                         inter_channels=128, sub_sample_factor=(2, 2, 2),
                                         mode='concatenation')
        self.GA34 = GridAttentionBlock3D(in_channels=256, gating_channels=512,
                                         inter_channels=256, sub_sample_factor=(2, 2, 2),
                                         mode='concatenation')

    def forward(self, x):
        x_stem = self.stem(x)
        x_layer1 = self.layer1(x_stem)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        GA_out23, att23 = self.GA23(x_layer2, x_layer3)
        GA_out34, att34 = self.GA34(x_layer3, x_layer4)
        x = torch.cat([self.avgpool(x_layer4), self.avgpool(GA_out23), self.avgpool(GA_out34)], 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 仅进行多尺度特征融合，不进行GA
class MCMF(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False, is_get_att=False):
        super(MCMF, self).__init__()
        self.is_get_att = is_get_att
        # 导入模型
        model = resnet_mixed_convolution(in_channel=in_channel, num_classes=num_classes, pretrained=pretrained)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        model.fc[1] = Linear(in_features=512 + 256, out_features=2, bias=True)
        self.fc = model.fc

    def forward(self, x):
        x_stem = self.stem(x)
        x_layer1 = self.layer1(x_stem)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x = torch.cat([self.avgpool(x_layer4), self.avgpool(x_layer3)], 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # 模型初始化
    # model = resnet_mixed_conv_GA(in_channel=4, num_classes=2, pretrained=False, need_fc=False).to(device)
    # model = resnet_mixed_convolution(in_channel=4, num_classes=2, pretrained=False).to(device)
    # model = resnet2p1Ω(in_channel=4, num_classes=2, pretrained=False).to(device)
    model = resnet_mixed_conv_GA(in_channel=4, num_classes=2, pretrained=False).to(device)
    print(model)
    # print(model.layer4[1].conv2)
    # model = resnet2p1(in_channel=4, pretrained=False).to(device)
    # model = r3d_18(in_channel=1, pretrained=False).to(device)
    # model = resnet_mixed_conv_SSA().to(device)

    # 输出模型架构
    summary(model, (4, 128, 128, 128))  # 打印模型结构

    input = torch.rand(1, 4, 128, 128, 128).to(device)
    output = model(input)
    print('output=', output.shape)

    # ----------------------------------------------------------------
    # Total params: 12, 223, 363
    # Trainable params: 12, 223, 363
    # Non - trainable params: 0
    # ----------------------------------------------------------------
    # Input size(MB): 32.00
    # Forward / backward pass size(MB): 2089063.96
    # Params size(MB): 46.63
    # Estimated Total Size(MB): 2089142.59
    # ----------------------------------------------------------------
