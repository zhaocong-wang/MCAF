import os

import torch
from torch import nn
from torch.nn import Linear
from torchsummary import summary

from models.GA import GridAttentionBlock3D
from models.resnet_mixed_convolution import resnet_mixed_convolution


class MCL3(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False):
        super(MCL3, self).__init__()
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
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_layer3 = self.layer3(x)
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
    model = MCL3(in_channel=4, num_classes=2, pretrained=False).to(device)
    print(model)

    # 输出模型架构
    summary(model, (4, 128, 128, 128))  # 打印模型结构

    input = torch.rand(1, 4, 128, 128, 128).to(device)
    print('input=', input.shape)

    output = model(input)
    print('output=', output.shape)
