import os
import torchio as tio
import numpy as np
import random
from tqdm import tqdm
import SimpleITK as sitk

from utils.tools import get_brats2019_data_path, z_score_no_0, show_one_slice, get_brats2021_data_path, rintersection


def preprocessing_2019(data_id, save_dir, z_score_mode):
    # 🟡分别对4个模态进行z-score
    z_score_torchio = tio.ZNormalization(masking_method=lambda x: x != 0)
    modals = ['t1', 't1ce', 't2', 'flair']
    for i, modal in enumerate(modals):
        modal = tio.ScalarImage(get_brats2019_data_path(data_id, modal)[0])  # (1, 240, 240, 155)
        if z_score_mode == 'torchio':
            modal = z_score_torchio(modal)
        elif z_score_mode == 'me':
            modal = z_score_no_0(modal.numpy()[0])
            modal = np.expand_dims(modal, axis=0)
        else:
            print('preprocessing_2019模式设置错误！')
            return 0
        if i == 0:
            data = modal
        else:
            data = np.concatenate((data, modal), axis=0)

    # 🟡堆叠为 (4, 240, 240, 155)
    data = np.array(data)

    # 🟡CXYZ -> CZYX
    data = data.transpose(0, 3, 2, 1)

    # 🟡保存文件
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    data = data.astype(np.float32)
    np.save(save_dir + '/{}.npy'.format(data_id), data)


def preprocessing_2021(data_id, save_dir, z_score_mode='me'):
    # 🟡分别对4个模态进行z-score
    z_score_torchio = tio.ZNormalization(masking_method=lambda x: x != 0)
    modals = ['t1', 't1ce', 't2', 'flair']
    for i, modal in enumerate(modals):
        # print(get_brats2021_data_path(data_id, modal, root_adjust='')[0])
        modal = tio.ScalarImage(get_brats2021_data_path(data_id, modal, root_adjust='')[0])  # (1, 240, 240, 155)
        if z_score_mode == 'torchio':
            print('z_score_mode:', z_score_mode)
            modal = z_score_torchio(modal)
        elif z_score_mode == 'me':
            modal = z_score_no_0(modal.numpy()[0])
            modal = np.expand_dims(modal, axis=0)
        else:
            print('preprocessing_2021模式设置错误！')
            return 0
        if i == 0:
            data = modal
        else:
            data = np.concatenate((data, modal), axis=0)

    # 🟡堆叠为 (5, 240, 240, 155)
    seg = tio.ScalarImage(get_brats2021_data_path(data_id, 't1')[1])  # (1, 240, 240, 155)
    data = np.concatenate((data, seg), axis=0)

    # 🟡CXYZ -> CZYX
    data = data.transpose(0, 3, 2, 1)

    # 🟡保存文件
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    data = data.astype(np.float32)
    np.save(save_dir + '/{}.npy'.format(data_id), data)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # # 🟢参数设置
    # save_dir = 'preprocessed_data/BraTS2019_zscoreNo0_by_me'
    # is_run = 1  # 用于控制 预处理模式/测试模式
    # z_score_mode = 'me'  # me / torchio
    #
    # # ⚪️状态信息展示
    # print('任务名：BraTS2019数据预处理')
    # print('目标文件夹：', save_dir)
    # print('z-score模式：', z_score_mode)
    #
    # # 🟡拿到id列表
    # with open('runtime_file/all_2019.txt', encoding='utf-8') as f:  # ⭐
    #     data_infos = f.readlines()
    #
    # # 🟡进行批量数据预处理
    # if is_run:
    #     bar = tqdm(enumerate(data_infos), total=len(data_infos))  # 初始化进度条
    #     for i, (item) in bar:
    #         item = item.split(';')[0]
    #         bar.set_description('正在处理{}'.format(item))  # 设置进度条开头
    #         preprocessing_2019(item, save_dir, z_score_mode)
    #
    # # 🔴测试处理后的数据
    # else:
    #     data = np.load(save_dir + '/' + data_infos[0].split(';')[0] + '.npy')
    #     print(data.shape)
    #     print(data.dtype)
    #     show_one_slice(data[0], 70)

    # -------------------------------------
    # 🟡拿到id列表
    # data_infos = []
    # with open('runtime_file/train_2021.txt', encoding='utf-8') as f:
    #     temp = f.readlines()
    # data_infos.extend(temp)
    # with open('runtime_file/val_2021.txt', encoding='utf-8') as f:
    #     temp = f.readlines()
    # data_infos.extend(temp)

    # 🟡进行批量数据预处理
    # bar = tqdm(enumerate(data_infos), total=len(data_infos))  # 初始化进度条
    # for i, (item) in bar:
    #     item = item.replace('\n', '')
    #     bar.set_description('正在处理{}'.format(item))  # 设置进度条开头
    #     preprocessing_2021(item, save_dir='preprocessed_data/BraTS2021_zscoreNo0_by_me')
    # -------------------------------------

    # 读取需要去除的id
    done = os.listdir('preprocessed_data/BraTS2021_zscoreNo0_by_me')
    done = [name.split('.')[0] for name in done]
    print(len(done), done)
    # 读取所有2021
    all = os.listdir('../datasets/BraTS/2021/BraTS2021_TrainingData')
    print(len(all))
    # 去除已完成的
    todo = rintersection(all, done)
    print(len(todo), todo)
    # 进行批量数据预处理
    bar = tqdm(enumerate(todo), total=len(todo))  # 初始化进度条
    for i, (item) in bar:
        bar.set_description('正在处理{}'.format(item))  # 设置进度条开头
        preprocessing_2021(item, save_dir='preprocessed_data/BraTS2021_zscoreNo0_by_me')
