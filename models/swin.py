import os
import torchvision
import torch
from torch import nn
from torchsummary import summary


# def swin3d(in_channels):
#     model = torchvision.models.video.swin_transformer.SwinTransformer3d(patch_size=[4, 4, 4], embed_dim=96,
#                                                                         depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                                                                         window_size=[2, 7, 7], num_classes=2)
#
#     embed_conv3d = nn.Conv3d(in_channels, 96, kernel_size=(4, 4, 4), stride=(4, 4, 4))
#     model.patch_embed.proj = embed_conv3d
#     return model


def swin3d_T(in_channels, num_classes):
    model = torchvision.models.video.swin_transformer.swin3d_t(num_classes=num_classes)
    # print(model)
    embed_conv3d = nn.Conv3d(in_channels, 96, kernel_size=(4, 4, 4), stride=(4, 4, 4))  # k s 都从(2, 4, 4)->(4, 4, 4)
    model.patch_embed.proj = embed_conv3d                                               # 否则爆显存
    return model


def swin3d_S(in_channels, num_classes):
    model = torchvision.models.video.swin_transformer.swin3d_s(num_classes=num_classes)
    embed_conv3d = nn.Conv3d(in_channels, 96, kernel_size=(4, 4, 4), stride=(4, 4, 4))  # k s 都从(2, 4, 4)->(4, 4, 4)
    model.patch_embed.proj = embed_conv3d                                               # 否则爆显存
    return model


def swin3d_B(in_channels, num_classes):
    model = torchvision.models.video.swin_transformer.swin3d_b(num_classes=num_classes)
    embed_conv3d = nn.Conv3d(in_channels, 128, kernel_size=(4, 4, 4), stride=(4, 4, 4)) # k s 都从(2, 4, 4)->(4, 4, 4)
    model.patch_embed.proj = embed_conv3d                                               # 否则爆显存
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # S:Total params:33,xxx,xxx T:Total params: 18,873,698  B爆显存
    model = swin3d_B(in_channels=4,num_classes=2).to(device)
    summary(model, (4, 128, 128, 128))  # 打印模型结构
