import gc
import os
import random
import sys
import time
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BraTS2019
from ..models.net3d import Net3d, Net3d_BigSmall, Net3d_BigSmall_se
from ..models.resnet18 import resnext50_32x4d
import torchio as tio

from models.resnet_mixed_convolution import resnet_mixed_convolution
from utils.tools import seed_torch


def train_with_FGSM(save_path, model, device, train_loader, val_loader, loss, optimizer, num_epochs, writer, fgsm_mode,
                    epsilon='rand', unfreeze=0, specified_one_modal_num='no use'):
    if epsilon == 'rand':
        print('FGSM_mode={}'.format(epsilon))
    if fgsm_mode == 'old':
        from utils.FGSM import fgsm_attack_old as fgsm_attack
    elif fgsm_mode == 'fixed':
        from utils.FGSM import fgsm_attack_fixed as fgsm_attack
    else:
        print('FGSM 模式设置错误！')
        return -1

    loss_history = []
    train_history = []
    val_history = []
    val_loss_hist = []
    metric_y_val = metric_p_val = None
    best_acc = 0  # 用于选择最佳权重1
    best_loss = 999  # 用于选择最佳权重2

    # 学习率调度器
    # https://blog.csdn.net/weixin_44682222/article/details/122218046
    # 1. CosineAnnealingWarmRestarts：在达到指定的epoch后，它会重新开始新的周期，从而避免了CosineAnnealingLR可能陷入的局部最优问题。该调度器的缺点是计算量较大，可能需要更多的训练时间来收敛。
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                      T_0=40,
    #                                                      T_mult=2,
    #                                                      eta_min=1e-9)
    # 2. CosineAnnealingLR：在每个epoch结束时根据余弦函数调整学习率，从而使得学习率逐渐降低，直到达到最小值。该调度器的优点是计算简单，对于大多数模型都能有效地提高性能。但是它有一个缺点，就是在学习率降低到最小时，可能会陷入局部最优，从而导致模型性能下降。
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):

        print(epoch)

        # 解冻
        if unfreeze != 0:
            if epoch == unfreeze:
                print('模型所有层解冻')
                for param in model.parameters():
                    param.requires_grad = True
                # optimizer.add_param_group({'params': model.fc1.parameters()})
                # optimizer = torch.optim.Adam(model.parameters(), scheduler.get_last_lr()[0])
                optimizer = torch.optim.Adam(model.parameters(), 1e-5)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - unfreeze, eta_min=0)

        model.train()

        correct_samples = 0
        total_samples = 0
        loss_accum = 0

        for i_step, (x, y) in tqdm(enumerate(train_loader)):

            # train on simple sample
            if specified_one_modal_num != 'no use':
                # print('specified_one_modal_num:',specified_one_modal_num)
                x_gpu = x[:, specified_one_modal_num, :, :, :].unsqueeze(1).to(device=device)
                # print(x_gpu.shape)
            else:
                x_gpu = x.to(device=device)

            y_gpu = y.to(device=device)

            # Set requires_grad attribute of tensor. Important for Attack
            x_gpu.requires_grad = True

            prediction = model(x_gpu)
            loss_value = loss(prediction, y_gpu.reshape((-1,)))
            _, preds = torch.max(prediction, 1)
            preds = preds.cpu()

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # traning on FGSM sample

            # Collect datagrad
            data_grad = x_gpu.grad.data.cpu()

            # Call FGSM Attack
            if epsilon != 'rand':
                perturbed_data = fgsm_attack(x_gpu.cpu(), epsilon, data_grad)
            else:
                print('采用了随机FGSM（在train.py）')
                perturbed_data = fgsm_attack(x_gpu.cpu(), random.uniform(0.005, 0.015), data_grad)  # ⭐

            del x_gpu

            prediction_fgsm = model(perturbed_data.to(device=device))
            loss_value = loss(prediction_fgsm, y_gpu.reshape((-1,)))
            _, preds_fgsm = torch.max(prediction_fgsm, 1)
            preds_fgsm = preds_fgsm.cpu()

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            y_cpu = y_gpu.cpu()

            preds = np.concatenate((preds.numpy(), preds_fgsm.numpy()))
            labels = np.concatenate((y_cpu.reshape((-1,)).numpy(), y_cpu.reshape((-1,)).numpy()))

            if i_step == 0 and epoch == 0:
                metric_y = labels
                metric_p = preds
            else:
                metric_y = np.concatenate((metric_y, labels))
                metric_p = np.concatenate((metric_p, preds))

            correct_samples += torch.sum(torch.tensor(preds) == torch.tensor(labels))
            loss_accum += float(loss_value.cpu())

            total_samples += y_cpu.shape[0] * 2
            del y_gpu
            del prediction_fgsm
            del perturbed_data
            gc.collect()

        ave_loss = loss_accum / ((i_step + 1) * 2)
        train_accuracy = correct_samples / total_samples
        writer.add_scalar("Loss/train", ave_loss, epoch)
        writer.add_scalar("Acc/train", train_accuracy, epoch)
        writer.add_scalar("F1/train", f1_score(metric_y, metric_p), epoch)

        val_accuracy, loss_val, metric_y_val, metric_p_val = compute_valid(model, device, val_loader, loss, epoch,
                                                                           metric_y_val,
                                                                           metric_p_val, specified_one_modal_num)
        writer.add_scalar("Loss/valid", loss_val, epoch)
        writer.add_scalar("Acc/valid", val_accuracy, epoch)
        writer.add_scalar("F1/valid", f1_score(metric_y_val, metric_p_val), epoch)

        writer.add_scalar("Lr/epoch", scheduler.get_last_lr()[-1], epoch)

        # scheduler.step(epoch)
        if unfreeze != 0:
            if epoch >= unfreeze:
                scheduler.step(epoch - unfreeze)
            else:
                scheduler.step(epoch)
        else:
            scheduler.step(epoch)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        val_loss_hist.append(loss_val)

        print("Average loss: %f, Val loss: %f, Train accuracy: %f, Val accuracy: %f, Train AP: %f,Val AP: %f" % (
            ave_loss, loss_val, train_accuracy, val_accuracy, average_precision_score(metric_y, metric_p),
            average_precision_score(metric_y_val, metric_p_val)))

        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())

        # if best_acc < val_accuracy:
        #     best_acc = val_accuracy
        #     torch.save(model.state_dict(), save_path + 'best.pth')

        if best_loss > loss_val:
            best_loss = loss_val
            torch.save(model.state_dict(), save_path + 'best_loss.pth')

        # if epoch % 5 == 0:  # ⭐
        #     torch.save(model.state_dict(), save_path + 'epoch{}.pth'.format(epoch))
        # torch.save(model.state_dict(), save_path + 'last.pth')

    writer.close()


def train(save_path, model, device, train_loader, val_loader, loss, optimizer, num_epochs, writer):
    # loss_history = []
    # train_history = []
    # val_history = []
    # val_loss_hist = []
    # metric_y_val = metric_p_val = None
    # best_acc = 0  # 用于选择最佳权重1
    best_loss = 999  # 用于选择最佳权重2

    # 学习率调度器
    # 1. CosineAnnealingWarmRestarts：在达到指定的epoch后，它会重新开始新的周期，从而避免了CosineAnnealingLR可能陷入的局部最优问题。该调度器的缺点是计算量较大，可能需要更多的训练时间来收敛。
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                      T_0=40,
    #                                                      T_mult=2,
    #                                                      eta_min=1e-9)
    # 2. CosineAnnealingLR：在每个epoch结束时根据余弦函数调整学习率，从而使得学习率逐渐降低，直到达到最小值。该调度器的优点是计算简单，对于大多数模型都能有效地提高性能。但是它有一个缺点，就是在学习率降低到最小时，可能会陷入局部最优，从而导致模型性能下降。
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    for epoch in range(num_epochs):

        print('epoch=', epoch)
        model.train()

        correct_samples = 0
        total_samples = 0
        loss_accum = 0

        for i_step, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device=device)
            y = y.to(device=device)

            prediction = model(x)
            loss_value = loss(prediction, y)  # y.reshape((-1,))

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, preds = torch.max(prediction, 1)

            if i_step == 0:
                metric_y = y.cpu().numpy()
                metric_p = preds.cpu().numpy()
            else:
                metric_y = np.concatenate((metric_y, y.cpu().numpy()))
                metric_p = np.concatenate((metric_p, preds.cpu().numpy()))

            correct_samples += torch.sum(preds == y)

            loss_accum += loss_value.item()

            total_samples += y.shape[0]

            # del y
            # del loss_value
            # gc.collect()

        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = correct_samples / total_samples
        writer.add_scalar("Loss/train", ave_loss, epoch)
        writer.add_scalar("Acc/train", train_accuracy, epoch)
        # writer.add_scalar("F1/train", f1_score(metric_y, metric_p), epoch)

        val_accuracy, loss_val = compute_valid(model, device, val_loader, loss)  # metric_y_val, metric_p_val
        writer.add_scalar("Loss/valid", loss_val, epoch)
        writer.add_scalar("Acc/valid", val_accuracy, epoch)
        # writer.add_scalar("F1/valid", f1_score(metric_y_val, metric_p_val), epoch)

        writer.add_scalar("Lr/epoch", scheduler.get_last_lr()[-1], epoch)
        scheduler.step(epoch)

        # loss_history.append(float(ave_loss))
        # train_history.append(train_accuracy)
        # val_history.append(val_accuracy)
        # val_loss_hist.append(loss_val)

        # print("Train Loss: %f, Val Loss: %f, Train Accuracy: %f, Val Accuracy: %f, Train AP: %f,Val AP: %f" % (
        #     ave_loss, loss_val, train_accuracy, val_accuracy, average_precision_score(metric_y, metric_p),
        #     average_precision_score(metric_y_val, metric_p_val)))
        print("Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}, LR: {}".format(
            ave_loss, loss_val, train_accuracy, val_accuracy, scheduler.get_last_lr()))

        # print('Epoch:', epoch, 'LR:', )

        # if best_acc < val_accuracy:
        #     best_acc = val_accuracy
        #     torch.save(model.state_dict(), save_path + 'best.pth')

        if best_loss > loss_val:
            best_loss = loss_val
            torch.save(model.state_dict(), save_path + 'best_loss.pth')

        # torch.save(model.state_dict(), save_path + 'last.pth')
        # if epoch % 5 == 0:  # ⭐
        #     torch.save(model.state_dict(), save_path + 'epoch{}.pth'.format(epoch))

    writer.close()


def compute_valid(model, device, loader, loss, specified_one_modal_num='no use'):
    model.eval()
    with torch.no_grad():
        correct_samples = 0
        total_samples = 0
        loss_accum = 0

        for i_step, (x, y) in enumerate(loader):
            # # x_gpu = x.to(device=device, dtype=torch.float)
            # x_gpu = x.to(device=device)

            if specified_one_modal_num != 'no use':
                # print('specified_one_modal_num:',specified_one_modal_num)
                x = x[:, specified_one_modal_num, :, :, :].unsqueeze(1).to(device=device)
                # print(x_gpu.shape)
            else:
                x = x.to(device=device)

            y = y.to(device=device)

            prediction = model(x)
            loss_value = loss(prediction, y)
            _, preds = torch.max(prediction, 1)

            if i_step == 0:
                metric_y = y.cpu().numpy()
                metric_p = preds.cpu().numpy()
            else:
                metric_y = np.concatenate((metric_y, y.cpu().numpy()))
                metric_p = np.concatenate((metric_p, preds.cpu().numpy()))

            correct_samples += torch.sum(preds == y)
            total_samples += y.shape[0]
            loss_accum += loss_value.item()

            # del x
            # del y
            # del loss_value

        loss_val = loss_accum / (i_step + 1)
        val_accuracy = correct_samples / total_samples
        return val_accuracy, loss_val  # , metric_y, metric_p


if __name__ == '__main__':
    # 🟡获取命令行参数，用于指定gpu
    gpu_num = int(sys.argv[1])

    # 🟡指定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # 🟡获取开始时间，用于命名文件夹
    start_time_str = str(time.localtime(time.time())[0]) + '_' + str(time.localtime(time.time())[1]) + '_' + \
                     str(time.localtime(time.time())[2]) + '_' + str(time.localtime(time.time())[3]) + '_' + \
                     str(time.localtime(time.time())[4]) + '_' + str(time.localtime(time.time())[5])
    print("开始时间", start_time_str)
    save_path = 'runs/' + start_time_str + '_Net3d_BigSmall_se3_relu_myZscore_bs32_lr4_cos_centercut/'
    print('存储路径：', save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    seed_torch(42)  # 固定随机种子

    # 🟢训练参数设置
    # model = resnet_mixed_convolution(in_channel=4, num_classes=3, pretrained=False).to(device)
    # model = swin3d(in_channels=4).to(device)
    # model = Net3d(in_channel=4, num_classes=2, is_leakyrelu=True).to(device)
    # model = Net3d_BigSmall(in_channel=4, num_classes=2, is_leakyrelu=False).to(device)
    model = Net3d_BigSmall_se(in_channel=4, num_classes=2, is_leakyrelu=False).to(device)
    batch_size = 32
    loss = nn.CrossEntropyLoss()  # FocalLoss(gamma=2, weight=[0.77, 0.23], to_onehot_y=True)
    # loss = nn.MSELoss()  # ⭐
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)  # 1e-5
    # optimizer = torch.optim.SGD(model.parameters(), 1e-5, momentum=0.9, nesterov=True, weight_decay=1e-4)
    data_folder_name = 'BraTS2019_zscoreNo0_by_me'

    # 🟢0326数据增强定义
    # spatial = tio.OneOf({
    #     tio.RandomAffine(scales=(0.95, 1.05), degrees=(-5, 5, 0, 0, 0, 0), isotropic=True,
    #                      image_interpolation='nearest'): 0.33,
    #     tio.RandomFlip(axes=2): 0.33,
    #     tio.RandomBlur(std=0.015): 0.33,
    # },
    #     p=0.5,
    # )
    # transforms = [spatial]
    # transform = tio.Compose(transforms)

    # 🟢0327数据增强定义
    # spatial = tio.OneOf({
    #     tio.RandomAffine(scales=(0.95, 1.05), degrees=(-2, 2, -2, 2, -5, 5), isotropic=True,
    #                      image_interpolation='nearest'): 0.33,
    #     tio.RandomFlip(axes=2): 0.33,
    #     tio.RandomBlur(std=0.015): 0.33,
    # },
    #     p=0.5,
    # )
    # transforms = [spatial]
    # transform = tio.Compose(transforms)

    # 🟢0328数据增强定义
    # rescale = RescaleIntensity((0.05, 99.5))
    # randaffine = torchio.RandomAffine(scales=(0.9, 1.2), degrees=10, isotropic=True, image_interpolation='nearest')
    # flip = torchio.RandomFlip(axes=2, p=0.5)
    # transforms = [rescale, flip, randaffine]
    # transform = Compose(transforms)

    # 2023.3.22 版数据增强
    # randaffine = tio.RandomAffine(scales=(0.95, 1.05), degrees=[-2, 2, -2, 2, -5, 5], isotropic=True,
    #                               image_interpolation='nearest',
    #                               p=0.5)
    # flip = tio.RandomFlip(axes=(0), p=0.5)  # 0:左右
    # Gamma = tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    # blur = tio.RandomBlur(std=0.3, p=0.5)
    # transforms = [randaffine, flip, blur]
    # transform = torchio.Compose(transforms)

    # train_set = BraTS2019_old(mode='train', rand_cut=True, transform=None, k_fold=k_fold)  # ⭐
    # val_set = BraTS2019_old(mode='val', rand_cut=True, transform=None, k_fold=k_fold)

    # 🟢0330数据增强定义
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

    # 🟢数据集定义
    train_set = BraTS2019(mode='MLISE_train', data_folder_name=data_folder_name, cut='center', transform=None, )
    val_set = BraTS2019(mode='MLISE_val', data_folder_name=data_folder_name, cut='center', transform=None)

    print('训练集：', len(train_set))
    print('验证集：', len(val_set))

    # 🟡训练开始
    writer = SummaryWriter(log_dir=save_path)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # train_with_FGSM(save_path, model, device, train_dataloader, val_dataloader, loss, optimizer, 200, writer,
    #                 fgsm_mode='fixed', epsilon=0.015)

    train(save_path, model, device, train_dataloader, val_dataloader, loss, optimizer, 200, writer)

    writer.close()
