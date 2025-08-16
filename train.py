import os
import random
import sys
import time

import numpy as np
import torch
import torchio
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchio as tio
from torchio import RescaleIntensity, Compose
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from dataset import BraTS2019
from models.MCL3 import MCL3
from models.babynet import BabyNet
from models.microsoft_i3d import InceptionI3d
from models.mobilenetv3_3d import MobileNetV3
from models.mvit import get_mvit_v2_s, get_mvit_v1_b
from models.resnet_mixed_convolution import resnet_mixed_convolution, resnet_mixed_conv_SE, \
    resnet_mixed_convolution_big_small_conv, r3d_18, resnet_mixed_conv_GA, MCMF
from models.squeezenet3d import SqueezeNet3D
from models.swin import swin3d_T
from models.vit import ViT
from models.vivit import ViViT
from utils.train import train, train_with_FGSM
from utils.tools import show_one_slice, seed_torch, get_forzen

criterion_list = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "BCEWithLogits": nn.BCEWithLogitsLoss(),
}


def get_model(model_name, in_channels, num_classes):
    model_list = {
        "MC": resnet_mixed_convolution(in_channel=in_channels, num_classes=num_classes, pretrained=False),
        "MCGA": resnet_mixed_conv_GA(in_channel=in_channels, num_classes=num_classes, pretrained=False),
        "R3D": r3d_18(in_channel=in_channels, num_classes=num_classes, pretrained=False),
        "MCL3": MCL3(in_channel=in_channels, num_classes=num_classes, pretrained=False),
        "swin3d-t": swin3d_T(in_channels=in_channels, num_classes=num_classes),
        "BabyNet": BabyNet(msha=True, n_frames=128, input_size=(128, 128), in_channels=in_channels,
                           num_classes=num_classes),
        "mvit-v2-s": get_mvit_v2_s(in_channel=in_channels, num_classes=num_classes),
        "mvit-v1-b": get_mvit_v1_b(in_channel=in_channels, num_classes=num_classes),
        "I3D": InceptionI3d(in_channels=in_channels, num_classes=num_classes),
        "ViT": ViT(in_channels=in_channels, num_classes=num_classes, img_size=(128, 128, 128), patch_size=16,
                   classification=True, spatial_dims=3),  # , post_activation=False, dropout_rate=0.2 有空可以试试
        "ViViT": ViViT(image_size=128, patch_size=16, num_classes=num_classes, num_frames=128, in_channels=in_channels,
                       heads=12, dim=768, dropout=0.1, emb_dropout=0.1),
        "MobileNetV3-L-3D": MobileNetV3(_mobilenet_v3_conf("mobilenet_v3_large")[0],
                                        _mobilenet_v3_conf("mobilenet_v3_large")[1],
                                        num_classes=num_classes, in_channels=in_channels),
        "MobileNetV3-S-3D": MobileNetV3(_mobilenet_v3_conf("mobilenet_v3_small")[0],
                                        _mobilenet_v3_conf("mobilenet_v3_small")[1],
                                        num_classes=num_classes, in_channels=in_channels, dropout=0.2),
        "SqueezeNet3D": SqueezeNet3D(in_channels, 128, 128, num_classes)

    }
    return model_list[model_name]


if __name__ == '__main__':

    # 🟡获取命令行参数，用于指定是K折交叉验证的第几折
    k_fold = int(sys.argv[1])

    # 🟡指定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(k_fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    seed_torch(42)  # 固定随机种子

    # 🟢训练参数设置-----------
    model_name = "MCL3"
    batch_size = 4
    epochs = 200
    specified_modal = 'noUse'  # 't1', 't1ce', 't2', 'flair' ｜ "noUse"
    loss_name = "CrossEntropy"
    aug_type = "BA"
    use_FGSM = True
    lr = 1e-5  # most:1e-5 | MobileNetV3-S-3D:5e-3
    # -----------------------

    loss = criterion_list[loss_name]
    data_folder_name = 'BraTS2019_zscoreNo0_by_me'
    model = get_model(model_name, 4 if specified_modal == "noUse" else 1, 2).to(device)
    num_workers = 2
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr,momentum=0.9)

    if aug_type:
        if aug_type == "BA":
            # 🟡0410数据增强定义（BA）
            LGG_weight = (335 - 76) / 335
            HGG_weight = 1 - LGG_weight
            print('数据增强概率：LGG={}，HGG={}'.format(LGG_weight, HGG_weight))
            spatial1 = tio.OneOf({
                tio.RandomAffine(scales=(0.95, 1.05), degrees=5, isotropic=True,
                                 image_interpolation='nearest'): 0.33,
                tio.RandomFlip(axes=2): 0.33,
                tio.RandomBlur(std=0.015): 0.33,
            },
                p=LGG_weight,
            )
            transforms1 = [spatial1]
            transform1 = tio.Compose(transforms1)

            spatial2 = tio.OneOf({
                tio.RandomAffine(scales=(0.95, 1.05), degrees=5, isotropic=True,
                                 image_interpolation='nearest'): 0.33,
                tio.RandomFlip(axes=2): 0.33,
                tio.RandomBlur(std=0.015): 0.33,
            },
                p=HGG_weight,
            )
            transforms2 = [spatial2]
            transform2 = tio.Compose(transforms2)

            transform = [transform1, transform2]

        elif aug_type == "AUG":
            # 🟡0330数据增强定义（aug）
            spatial = tio.OneOf({
                tio.RandomAffine(scales=(0.95, 1.05), degrees=5, isotropic=True,
                                 image_interpolation='nearest'): 0.33,
                tio.RandomFlip(axes=2): 0.33,
                tio.RandomBlur(std=0.015): 0.33,
            },
                p=0.5,
            )
            transforms = [spatial]
            transform = tio.Compose(transforms)
        else:
            print("数据增强模式设置错误：{}".format(aug_type))
    else:
        transform = None

    # 🟡初始化数据集
    train_set = BraTS2019(mode='train', data_folder_name=data_folder_name, cut='center', transform=transform,
                          k_fold=k_fold,
                          specified_modal=[specified_modal] if specified_modal != "noUse" else ['t1', 't1ce', 't2',
                                                                                                'flair'],
                          is_slice_channel_h_w=True if model_name == "ViViT" else False)
    val_set = BraTS2019(mode='val', data_folder_name=data_folder_name, cut='center', transform=None,
                        k_fold=k_fold,
                        specified_modal=[specified_modal] if specified_modal != "noUse" else ['t1', 't1ce', 't2',
                                                                                              'flair'],
                        is_slice_channel_h_w=True if model_name == "ViViT" else False)

    print('训练集：', len(train_set))
    print('验证集：', len(val_set))

    # 🟡dataloader定义
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 🟡指定存储路径
    save_path = 'runs2024/{}_myZscore_5Fold'.format(model_name) + str(
        k_fold) + '_bs{}_epochs{}_loss{}_cos_centercut_specifiedModal{}_augType{}_{}/'.format(
        batch_size,
        epochs,
        loss_name,
        specified_modal,
        aug_type if aug_type else "NoUse",
        "useFGSM" if use_FGSM else "noUseFGSM",
        # lr
    )
    print('存储路径：', save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    writer = SummaryWriter(log_dir=save_path)

    if use_FGSM:
        train_with_FGSM(save_path, model, device, train_dataloader, val_dataloader, loss, optimizer, epochs, writer,
                        fgsm_mode='fixed', epsilon=0.015,
                        unfreeze=0)  # specified_one_modal_num=['t1', 't1ce', 't2', 'flair'].index(specified_modal)
    else:
        train(save_path, model, device, train_dataloader, val_dataloader, loss, optimizer, epochs, writer)

    writer.close()
