import torch
import torch.nn as nn
import math
from torchsummary import summary

# https://github.com/YOLOonMe/EMA-attention-module/blob/main/EMA_attention_module
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class se_block_3d(nn.Module):
    def __init__(self, channel, ratio=2):  # 16
        super(se_block_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)  # 每个通道得到一个权重f
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class se_block_2_5d(nn.Module):
    def __init__(self, channel, z_num, ratio=16, group=8):  # 16
        super(se_block_2_5d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.group = group
        self.fc = nn.Sequential(
            nn.Linear(channel * z_num // group, channel * z_num // group // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * z_num // group // ratio, channel * z_num // group, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, z, h, w = x.size()
        # print(c * z / self.group)
        x2_5 = x.view(b, int(c * z / self.group), self.group, h, w)
        # print(x2_5.shape)
        weight = self.avg_pool(x2_5).view(b, int(c * z / self.group))  # 每个通道得到一个权重f
        # print(weight.shape)
        weight = self.fc(weight).view(b, int(c * z / self.group), 1, 1, 1)
        # print(weight.shape)
        return (x2_5 * weight).view(b, c, z, h, w)


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 用于slice_spatial_attention_3d(SSA) 我发明的
class slice_attention_block(nn.Module):
    def __init__(self, channel, z_num, r=0.5, group=8):
        super(slice_attention_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.group = group
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(channel * (z_num + 1) // group, int((channel * (z_num + 1) // group) * r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int((channel * (z_num + 1) // group) * r), channel * (z_num + 1) // group, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)

        # 将127的z轴填充为128
        pad_dims = (0, 0, 0, 0, 0, 1, 0, 0)
        pad_value = 0
        x = torch.nn.functional.pad(x, pad_dims, mode='constant', value=pad_value)

        b, c, z, h, w = x.size()
        # print(c * z / self.group)
        x2_5 = x.view(b, int(c * z / self.group), self.group, h, w)
        # print(x2_5.shape)

        weight_avg = self.avg_pool(x2_5).view(b, int(c * z / self.group))  # 每个通道得到一个权重f
        # print(weight_avg.shape)
        weight_avg = self.fc(weight_avg).view(b, int(c * z / self.group), 1, 1, 1)
        # print(weight_avg.shape)

        weight_max = self.max_pool(x2_5).view(b, int(c * z / self.group))  # 每个通道得到一个权重f
        # print(weight_max.shape)
        weight_max = self.fc(weight_max).view(b, int(c * z / self.group), 1, 1, 1)

        weight = weight_max + weight_avg
        weight = self.sigmoid(weight)
        return (x2_5 * weight).view(b, c, z, h, w)


# 用于CBAM3d
class ChannelAttentionModul3d(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul3d, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool3d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool3d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1, 1))
        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul3d(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul3d, self).__init__()
        self.conv = nn.Conv3d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('!!!', x.shape)
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, C, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, C, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


class CBAM3d(nn.Module):
    def __init__(self, in_channel):
        super(CBAM3d, self).__init__()
        self.Cam = ChannelAttentionModul3d(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul3d(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class slice_spatial_attention_3d(nn.Module):
    def __init__(self, in_channel, z_num):
        super(slice_spatial_attention_3d, self).__init__()
        self.Cam = slice_attention_block(channel=in_channel, z_num=z_num)  # 通道注意力模块
        self.Sam = SpatialAttentionModul3d(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


# -------------------------------------------------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 坐标注意力机制
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # block = ChannelAttentionModul3d(4).to(device)
    # input = torch.rand((1, 4, 128, 128, 128)).to(device)
    # output = block(input)
    # print('output',output.shape)
    # summary(block, (4, 128, 128, 128))

    block = slice_spatial_attention_3d(4, 127).to(device)
    input = torch.rand((1, 4, 127, 128, 128)).to(device)
    output = block(input)
    print('output', output.shape)
    summary(block, (4, 127, 128, 128))

    # se_block = se_block(128).to(device)
    # summary(se_block, (128, 32, 32))
