import torch
import torch.nn as nn
from torchsummary import summary

from models.attentions import se_block_3d


class Net3d(nn.Module):
    def __init__(self, in_channel, num_classes, is_leakyrelu=False):
        super(Net3d, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        if is_leakyrelu:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=2),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=2),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )

    def forward(self, x):  # (4,128,128,128)
        x = self.encoder(x)

        output = self.classifier(x)  # (4,128,128,128)

        return output


class Net3d_BigSmall(nn.Module):
    def __init__(self, in_channel, num_classes, is_leakyrelu=False):
        super(Net3d_BigSmall, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes

        if is_leakyrelu:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )

    def forward(self, x):  # (4,128,128,128)
        x = self.encoder(x)

        output = self.classifier(x)  # (4,128,128,128)

        return output


class Net3d_BigSmall_avgpool(nn.Module):
    def __init__(self, in_channel, num_classes, is_leakyrelu=False):
        super(Net3d_BigSmall_avgpool, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        if is_leakyrelu:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.AvgPool3d(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )

    def forward(self, x):  # (4,128,128,128)
        x = self.encoder(x)

        output = self.classifier(x)  # (4,128,128,128)

        return output


class Net3d_BigSmall_se(nn.Module):
    def __init__(self, in_channel, num_classes, is_leakyrelu=False):
        super(Net3d_BigSmall_se, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        # 注意力机制 初始化
        # self.se1 = se_block_3d(64)
        # self.se2 = se_block_3d(128)
        self.se3 = se_block_3d(256)

        if is_leakyrelu:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                # self.se1,
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                # self.se2,
                nn.BatchNorm3d(128),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                self.se3,
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channel, 64, kernel_size=9, stride=4, padding=1),
                # self.se1,
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(64, 128, kernel_size=7, stride=1, padding=2),
                # self.se2,
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2),

                nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
                self.se3,
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(3, 2)
            )
            # self.classifier = nn.Conv3d(in_channels=32, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, self.num_classes),
            )

    def forward(self, x):  # (4,128,128,128)
        x = self.encoder(x)

        output = self.classifier(x)  # (4,128,128,128)

        return output


# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # 模型初始化
    model = Net3d_BigSmall_avgpool(in_channel=4, num_classes=2, is_leakyrelu=True).to(device)
    # model = Net3d_relu(in_channel=4, num_classes=2).to(device)
    # 输出模型架构
    summary(model, (4, 128, 128, 128))  # 打印模型结构
