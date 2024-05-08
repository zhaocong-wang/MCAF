import torch
import torchvision
import torch.nn as nn
from torch.nn import Linear
from torchsummary import summary
import os
import warnings
import sys
from models.GA import GridAttentionBlock3D

sys.path.append(r"models")
warnings.filterwarnings("ignore")


class Stem(nn.Sequential):
    def __init__(self, in_channel):
        super(Stem, self).__init__(
            nn.Conv3d(in_channel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


def MC(in_channel, num_classes, pretrained=False):
    stem = Stem(in_channel=in_channel)
    model = torchvision.models.video.mc3_18(pretrained=pretrained)
    model.stem = stem
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


class MCAF(nn.Module):
    def __init__(self, in_channel, num_classes, pretrained=False, is_get_att=False, need_fc=True):
        super(MCAF, self).__init__()
        self.is_get_att = is_get_att
        self.need_fc = need_fc
        # 导入模型
        model = MC(in_channel=in_channel, num_classes=num_classes, pretrained=pretrained)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        if need_fc:
            model.fc[1] = Linear(in_features=512 + 256, out_features=2, bias=True)
            self.fc = model.fc

        self.GA = GridAttentionBlock3D(in_channels=256, gating_channels=512,
                                       inter_channels=256, sub_sample_factor=(2, 2, 2),
                                       mode='concatenation')

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # x_layer1
        x = self.layer2(x)  # x_layer2
        x_layer3 = self.layer3(x)
        x_layer4 = self.layer4(x_layer3)
        GA_out, att = self.GA(x_layer3, x_layer4)
        x = torch.cat([self.avgpool(x_layer4), self.avgpool(GA_out)], 1)
        x = torch.flatten(x, 1)

        if self.need_fc:
            x = self.fc(x)

        if self.is_get_att:
            return x, att
        else:
            return x


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device：', device)

    model = MCAF(in_channel=4, num_classes=2, pretrained=False).to(device)
    summary(model, (4, 128, 128, 128))

    input = torch.rand(1, 4, 128, 128, 128).to(device)
    output = model(input)
    print('output=', output.shape)
