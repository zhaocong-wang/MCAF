import csv
import os
import random
import math
import imageio as iio
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import skimage.io as io
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight


def read_nii(path):
    data = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(data)
    return data


def seg_label_to_0_1(id, save_path):
    """
    将所有子区域全部组合，成为肿瘤整体，并打包为nii.gz
    :return:
    """
    # 生成相对路径
    isHgg = lgg_0_hgg_1(id)
    if isHgg:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/'
    else:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/'
    path = root_path + id + '/' + id + '_seg.nii.gz'
    # 读取为numpy

    data = sitk.ReadImage(path)
    # 💗获取原始文件的坐标和位置空间
    origin = data.GetOrigin()  # 这三句是获取的原始图像文件的位置和方向吧。
    spacing = data.GetSpacing()
    direction = data.GetDirection()

    data = sitk.GetArrayFromImage(data)  # uint8 or int16
    # data.astype(np.uint8)

    # 处理
    data[data != 0] = 1

    # 💗将自己的文件处理成和官方一致的位置坐标系
    data = sitk.GetImageFromArray(data)
    data.SetOrigin(origin)
    data.SetSpacing(spacing)
    data.SetDirection(direction)
    # 保存
    sitk.WriteImage(data, save_path + id + '_seg_full_tumor.nii.gz')


def recode(id, modal, save_path):
    """
    本文件用于将nii.gz文件重新编码
    简而言之就是，使用simpleitk读取nii -> 转为numpy -> 重新打包回 nii
    :return:
    """
    # 生成相对路径
    isHgg = lgg_0_hgg_1(id)
    if isHgg:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/'
    else:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/'
    path = root_path + id + '/' + id + '_' + modal + '.nii.gz'

    # 读取为numpy
    data = sitk.ReadImage(path)
    # 💗获取原始文件的坐标和位置空间
    origin = data.GetOrigin()  # 这三句是获取的原始图像文件的位置和方向吧。
    spacing = data.GetSpacing()
    direction = data.GetDirection()

    data = sitk.GetArrayFromImage(data)  # uint8 or int16
    print(data.dtype)
    # data.astype(np.uint8)

    # 💗将自己的文件处理成和原始文件一致的位置坐标系
    data = sitk.GetImageFromArray(data)
    data.SetOrigin(origin)
    data.SetSpacing(spacing)
    data.SetDirection(direction)
    # 保存
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    sitk.WriteImage(data, save_path + id + '_' + modal + '.nii.gz')


def to_one_hot(label):
    """
    转onehot
    :param label:
    :return:
    """
    label[label == 4] = 3
    one_modal = F.one_hot(torch.tensor(label.astype(np.int64)), 4)
    one_modal = one_modal.permute(3, 0, 1, 2)
    one_modal = one_modal.numpy()
    return one_modal.astype(np.int16)


def r_one_hot(label):
    """
    onehot转初始
    :param label:
    :return:
    """
    bg = label[0].copy()
    bg[bg == 1] = 0

    label1 = label[1].copy()

    label2 = label[2].copy()
    label2[label2 == 1] = 2

    label4 = label[3].copy()
    label4[label4 == 1] = 3

    original_label = bg + label1 + label2 + label4

    return original_label


def cut_to_128(data):
    """
    数据裁剪 (155, 240, 240) -> (128, 128, 128) or (4, 155, 240, 240) -> (4, 128, 128, 128)
    :param data:
    :return:
    """
    modal_num = len(data.shape)  # 判断是one_modal还是4个modal打包送入
    if modal_num == 3:
        data = data[15:143, 8:232, 16:240]  # 确保不裁剪掉肿瘤区域的情况下,进行非对称裁切
        data = data[0:128, 48:176, 48:176]  # 再进行中心裁切（损伤了边缘的肿瘤，但受制于显存，无可奈何）
    elif modal_num == 4:
        data = data[:, 15:143, 8:232, 16:240]
        data = data[:, 0:128, 48:176, 48:176]
    return data


def r_cut_to_128(data):
    map1 = np.zeros((128, 224, 224))
    map1[0:128, 48:176, 48:176] = data
    map2 = np.zeros((155, 240, 240))
    map2[15:143, 8:232, 16:240] = map1  # 此时的map2已经是处理完的(155,240,240)的pred
    return map2


def cut_to_lossless_minimum(data):
    """
    数据裁剪到无损肿瘤区域的最小 (155, 240, 240) -> (128, 186, 194) or (4, 155, 240, 240) -> (4, 128, 186, 194)
    :param data:
    :return:
    """
    modal_num = len(data.shape)  # 判断是one_modal还是4个modal打包送入
    if modal_num == 3:
        data = data[15:143, 29:215, 45:239]  # 确保不裁剪掉肿瘤区域的情况下,进行非对称裁切
    elif modal_num == 4:
        data = data[:, 15:143, 29:215, 45:239]
    return data


def show_one_slice(data, num, title=None, need_save=False, save_path=''):
    if need_save:
        io.imsave(save_path, data[num])
    else:
        io.imshow(data[num], cmap='gray')
        if title:
            plt.suptitle(title, y=1, horizontalalignment='center')
        io.show()


def intersection(a, b):
    """
    求交集
    :param a:
    :param b:
    :return:
    """
    intersection = [v for v in a if v in b]
    return intersection


def rintersection(a, b):
    """
    求差集
    :param a:
    :param b:
    :return:
    """
    rintersection = [v for v in a if v not in b]
    return rintersection


def get_brats2021_data_path(img_id, modal, root_adjust=''):
    root_path = root_adjust + '../datasets/BraTS/2021/BraTS2021_TrainingData/'

    img_path = os.path.join(root_path, img_id) + '/' + img_id + '_' + modal + '.nii.gz'
    lab_path = os.path.join(root_path, img_id) + '/' + img_id + '_' + 'seg.nii.gz'

    return img_path.replace('\n', ''), lab_path.replace('\n', '')


def brats2020_data_reader(id, need_onehot):
    """
    :param subject_path: 某患者id
    :return:
    """

    four_modal = []
    root_path = '../datasets/BraTS/MICCAI_BraTS2020_TrainingData/'
    modal_path = [root_path + id + '/' + id + '_flair.nii',
                  root_path + id + '/' + id + '_t1.nii',
                  root_path + id + '/' + id + '_t1ce.nii',
                  root_path + id + '/' + id + '_t2.nii',
                  root_path + id + '/' + id + '_seg.nii']
    for one_modal_path in modal_path:
        one_modal = sitk.ReadImage(one_modal_path)  # <class 'SimpleITK.SimpleITK.Image'>
        one_modal = sitk.GetArrayFromImage(one_modal)  # 转为数组 <class 'numpy.ndarray'>

        if one_modal_path[-5] != 'g':
            # 处理input
            four_modal.append(one_modal)
        else:
            # 处理标签
            if need_onehot:
                seg = to_one_hot(one_modal)
            else:
                seg = np.expand_dims(one_modal, axis=0)

    four_modal = np.array(four_modal)
    return four_modal, seg


def brats2019_data_reader(id, need_onehot=False):
    four_modal = []
    isHgg = lgg_0_hgg_1(id)
    if isHgg:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/'
    else:
        root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/'
    modal_path = [root_path + id + '/' + id + '_flair.nii.gz',
                  root_path + id + '/' + id + '_t1.nii.gz',
                  root_path + id + '/' + id + '_t1ce.nii.gz',
                  root_path + id + '/' + id + '_t2.nii.gz',
                  root_path + id + '/' + id + '_seg.nii.gz']
    for one_modal_path in modal_path:
        one_modal = sitk.ReadImage(one_modal_path)  # <class 'SimpleITK.SimpleITK.Image'>
        one_modal = sitk.GetArrayFromImage(one_modal)  # 转为数组 <class 'numpy.ndarray'>
        one_modal = z_score(one_modal)  # ⚠️z_score处理

        if one_modal_path[-8] != 'g':
            # 处理input
            four_modal.append(one_modal)
        else:
            # 处理标签
            if need_onehot:
                seg = to_one_hot(one_modal)
            else:
                seg = np.expand_dims(one_modal, axis=0)

    four_modal = np.array(four_modal)

    return four_modal, seg


def divide_8_2(name_list, random_seed=37):
    np.random.seed(random_seed)
    length = len(name_list)
    part = length // 10
    np.random.shuffle(name_list)

    train_list = name_list[0:part * 8]
    test_list = name_list[part * 8:]

    return train_list, test_list


def divide_2_1(name_list, random_seed=37):
    np.random.seed(random_seed)
    length = len(name_list)
    part = length // 3
    np.random.shuffle(name_list)

    train_list = name_list[0:part * 2]
    test_list = name_list[part * 2:]

    return train_list, test_list


def divide_n_1(name_list, n, random_seed=37):
    np.random.seed(random_seed)
    length = len(name_list)
    part = length // (n + 1)
    np.random.shuffle(name_list)

    train_list = name_list[0:part * n]
    test_list = name_list[part * n:]

    return train_list, test_list


# 均分成n份
def divide_list(l, n):
    # 计算每一份的大小
    size = len(l) // n
    # 将列表分割成 n 个子列表
    result = [l[i:i + size] for i in range(0, len(l), size)]
    # 如果列表无法均分，则将剩余的元素添加到最后一个子列表
    if len(result) > n:
        result[-2] = result[-2] + result[-1]
        result.pop()
    return result


def divide_8_1_1(name_list, random_seed=37):
    np.random.seed(random_seed)
    length = len(name_list)
    part = length // 10
    np.random.shuffle(name_list)

    train_num = part * 8
    val_num = (length - train_num) // 2
    train_list = name_list[0:train_num]
    val_list = name_list[train_num:train_num + val_num]
    test_list = name_list[train_num + val_num:]

    return train_list, val_list, test_list


def z_score_no_0(one_model):
    """
    :param one_model: numpy array 类型的 (155,240,240) 数据
    :return: 处理后的numpy
    """
    # z-score标准化
    # 1.求均值avg
    nozero_num = np.count_nonzero(one_model)  # 非0元素个数  1455282
    sum = np.sum(one_model)  # 所有元素之和
    avg = sum / nozero_num  # 均值
    # 2.求标准差
    # nozero_numpy = one_modal.copy()
    nozero_numpy = one_model[one_model != 0]  # 删除Numpy数组中所有的0元素  (1455282,)
    std = np.std(nozero_numpy)  # 标准差 79.27162262
    # 3.制作shape=(155,240,240)的array，准备相减
    mask = one_model.copy()
    mask[mask > 0] = 1
    mask = mask * avg
    # 4.完成计算
    output = (one_model - mask) / std
    return output


def z_score(one_model):
    # 计算均值和标准差
    mean = np.mean(one_model)
    std = np.std(one_model)

    # 对图像进行z-score标准化
    return (one_model - mean) / std


def Dice(output, target, eps=1e-3):
    inter = np.sum(output * target) + eps
    union = np.sum(output) + np.sum(target) + eps * 2
    x = 2 * inter / union
    dice = np.mean(x)
    return dice


def data_list_split_label(data_list):
    """
    :param data_list: 从txt文件中读取的train_lines
    :return: 文件路径，标签
    """
    path = []
    label = []
    for item in data_list:
        path.append(item.split(';')[0])
        label.append(int(item.split(';')[1][:-1]))
    return path, label


def get_sampler_weights_list(data_lines_path):
    """
    :param data_lines_path: 如 'val_2019.txt'
    :return: WeightedRandomSampler 所需的 weights_list
    """
    with open('runtime_file/' + data_lines_path, encoding='utf-8') as f:
        train_list = f.readlines()
    train_path, train_label = data_list_split_label(train_list)  # 🌟1.获取训练集标签列表

    class_0 = 0
    class_1 = 0
    # print(train_label)
    for item in train_label:  # 🌟2.计算每一类各有多少个样本
        if item == 0:
            class_0 += 1
        elif item == 1:
            class_1 += 1
        else:
            print('1出错了')
    class_sample_count = [class_0, class_1]
    # print(class_sample_count)

    weights = 1 / torch.Tensor(class_sample_count)  # 🌟3.计算类别权重
    # print(weights)

    weights_list = []  # 🌟4.处理为len=样本个数的权重list
    for item in train_label:
        if item == 0:
            weights_list.append(weights[0])
        elif item == 1:
            weights_list.append(weights[1])
        else:
            print('2出错了')
    # print(weights_list)
    return weights_list


def get_class_punishment_weight(data_lines_path):
    with open('runtime_file/' + data_lines_path, encoding='utf-8') as f:
        train_list = f.readlines()
    train_path, train_label = data_list_split_label(train_list)  # 🌟1.获取训练集标签列表
    class_0 = 0
    class_1 = 0
    # print(train_label)
    for item in train_label:  # 🌟2.计算每一类各有多少个样本
        if item == 0:
            class_0 += 1
        elif item == 1:
            class_1 += 1
        else:
            print('1出错了')
    return torch.tensor([1 - (class_0 / (class_0 + class_1)), 1 - (class_1 / (class_0 + class_1))])


def get_class_punishment_weight_sklearn(data_lines_path):
    with open('runtime_file/' + data_lines_path, encoding='utf-8') as f:
        train_list = f.readlines()
    train_path, train_label = data_list_split_label(train_list)  # 🌟1.获取训练集标签列表
    weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_label)
    return torch.tensor(weight.astype(np.float32))


def lgg_0_hgg_1(id, root_adjust=None):
    if root_adjust:
        hgg_list = os.listdir(root_adjust + '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG')  # 259人
        lgg_list = os.listdir(root_adjust + '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG')  # 76人
    else:
        hgg_list = os.listdir('../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG')  # 259人
        lgg_list = os.listdir('../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG')  # 76人
    if id in lgg_list:
        return 0
    elif id in hgg_list:
        return 1
    else:
        print('lgg_0_hgg_1函数被传入了错误的患者id')
        return 'lgg_0_hgg_1函数被传入了错误的患者id'


# 生成文件路径
def get_brats2019_data_path(img_id, modal, root_adjust=None):
    isHgg = lgg_0_hgg_1(img_id, root_adjust=root_adjust)
    if root_adjust:
        if isHgg:
            root_path = root_adjust + '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/'
        else:
            root_path = root_adjust + '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/'
    else:
        if isHgg:
            root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/'
        else:
            root_path = '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/'

    img_path = root_path + img_id + '/' + img_id + '_' + modal + '.nii.gz'
    lab_path = root_path + img_id + '/' + img_id + '_seg.nii.gz'
    # lab_path = root_path + img_id + '/' + img_id + '_seg.nii.gz'

    return img_path, lab_path


def gen_all_csv():
    """
    为py-radiomics制作所需的目标文件
    :return:
    """
    ids = os.listdir('../datasets/BraTS/BraTS_2019_seg_full_tumor/')
    print('共{}位患者'.format(len(ids)))

    f = open('runtime_file/all.csv', mode='a', encoding='utf-8', newline='')
    isFirst = True
    for id in ids:
        img_path, lab_path = get_brats2019_data_path(id[:-22], modal='flair')
        row = {
            'Image': img_path,
            'Mask': lab_path
        }
        print(row)
        # 若是第一行数据，则先添加表头
        if isFirst:
            csv_writer = csv.DictWriter(f, ['Image', 'Mask'])
            csv_writer.writeheader()
            isFirst = False

        # 将提取到的特征存储到csv
        csv_writer.writerow(row)

    f.close()


def imgs2gif(image_dir_list, output_dir, gif_name, duration=0.05):
    """
    :param image_dir_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    """
    frames = []
    for image_name in image_dir_list:
        frames.append(iio.imread(image_name))

    iio.mimsave(output_dir + gif_name + '.gif', frames, 'GIF', duration=duration)
    # 删除单帧的png图片
    for file_path in image_dir_list:
        os.remove(file_path)


def nii_to_pngs(nii_id, modal, output_path):
    nii = sitk.ReadImage(get_brats2019_data_path(nii_id, modal)[0])  # 读取为sitk的图片类型 <class 'SimpleITK.SimpleITK.Image'>
    nii = sitk.GetArrayFromImage(nii)  # 转为数组 <class 'numpy.ndarray'>
    for i in range(nii.shape[0]):
        io.imsave(fname=output_path + str(i) + '.png', arr=nii[i, :, :])


def seg_to_pngs(nii_id, modal, output_path):
    nii = sitk.ReadImage(get_brats2019_data_path(nii_id, modal)[1])  # 读取为sitk的图片类型 <class 'SimpleITK.SimpleITK.Image'>
    nii = sitk.GetArrayFromImage(nii)  # 转为数组 <class 'numpy.ndarray'>

    for i in range(nii.shape[0]):
        # （240，240）-> (3,240,240)
        img_rgb = np.stack([nii[i, :, :], nii[i, :, :], nii[i, :, :]], axis=0)
        print(img_rgb.shape)
        # 标签赋予颜色 1,2,4
        img_rgb[0][img_rgb[0] == 1] = 255
        img_rgb[1][img_rgb[1] == 1] = 0
        img_rgb[2][img_rgb[2] == 1] = 0

        img_rgb[0][img_rgb[0] == 2] = 0
        img_rgb[1][img_rgb[1] == 2] = 255
        img_rgb[2][img_rgb[2] == 2] = 0

        img_rgb[0][img_rgb[0] == 4] = 0
        img_rgb[1][img_rgb[1] == 4] = 0
        img_rgb[2][img_rgb[2] == 4] = 255

        # ZYX -> XYZ
        img_rgb = img_rgb.transpose(2, 1, 0)
        io.imsave(fname=output_path + str(i) + '.png', arr=img_rgb)


def nii_seg_to_pngs(nii_id, modal, output_path):
    nii = sitk.ReadImage(get_brats2019_data_path(nii_id, modal)[0])  # 读取为sitk的图片类型 <class 'SimpleITK.SimpleITK.Image'>
    nii = sitk.GetArrayFromImage(nii)  # 转为数组 <class 'numpy.ndarray'>
    seg = sitk.ReadImage(get_brats2019_data_path(nii_id, modal)[1])
    seg = sitk.GetArrayFromImage(seg)

    for i in range(seg.shape[0]):
        # （240，240）-> (3,240,240)
        seg_rgb = np.stack([seg[i, :, :], seg[i, :, :], seg[i, :, :]], axis=0)
        print(seg_rgb.shape)
        # 标签赋予颜色 1,2,4
        seg_rgb[0][seg_rgb[0] == 1] = 255
        seg_rgb[1][seg_rgb[1] == 1] = 0
        seg_rgb[2][seg_rgb[2] == 1] = 0

        seg_rgb[0][seg_rgb[0] == 2] = 0
        seg_rgb[1][seg_rgb[1] == 2] = 255
        seg_rgb[2][seg_rgb[2] == 2] = 0

        seg_rgb[0][seg_rgb[0] == 4] = 0
        seg_rgb[1][seg_rgb[1] == 4] = 0
        seg_rgb[2][seg_rgb[2] == 4] = 255

        # 将分割标签放到原图上
        nii_rgb = np.stack([nii[i, :, :], nii[i, :, :], nii[i, :, :]], axis=0)
        img_rgb = nii_rgb + seg_rgb * 0.9

        # ZYX -> XYZ
        img_rgb = img_rgb.transpose(1, 2, 0)
        io.imsave(fname=output_path + str(i) + '.png', arr=img_rgb)


def nii_to_gif(nii_id, modal, output_path, mode):
    if mode == 'img':
        nii_to_pngs(nii_id, modal, output_path)
    elif mode == 'seg':
        seg_to_pngs(nii_id, modal, output_path)
    elif mode == 'img_seg':
        nii_seg_to_pngs(nii_id, modal, output_path)
    else:
        print('模式设置错误为：', mode)
        exit()

    image_path_list = []
    for i in range(155):
        image_path_list.append(output_path + str(i) + '.png')

    imgs2gif(image_path_list, output_path, nii_id + '_' + modal + '_' + mode, duration=0.05)


def to_name_list(alist):
    out = []
    for item in alist:
        out.append(item.split(';')[0])
    return out


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_standard_deviation(np_array):
    """
    计算标准差
    :param list: np.array([0.9394, 0.9091, 0.9697, 0.9394, 0.9155])
    :return:标准差
    """
    return np.std(np_array)


def get_forzen(model):
    frozen_layers = []
    trainable_layers = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_layers.append(name)
        else:
            trainable_layers.append(name)

    print("被冻结的层:")
    for layer in frozen_layers:
        print(layer)


def get_all_2021_ids():
    map_df = pd.read_csv('../../datasets/BraTS/BraTS21-17_Mapping.csv')
    map_df = map_df[['BraTS2021', 'BraTS2019']]
    map_dict = map_df.set_index('BraTS2021')['BraTS2019'].to_dict()  # key是2021，value是2019
    map_dict = {k: v for k, v in map_dict.items() if not isinstance(k, float) or not math.isnan(k)}  # 去除nan 1251
    all_2021 = list(map_dict.keys())
    return all_2021


def get_2021_by_2019(id_2021):
    map_df = pd.read_csv('../datasets/BraTS/BraTS21-17_Mapping.csv')
    map_df = map_df[['BraTS2021', 'BraTS2019']]
    map_dict = map_df.set_index('BraTS2019')['BraTS2021'].to_dict()  # key是2021，value是2019
    map_dict = {k: v for k, v in map_dict.items() if not isinstance(k, float) or not math.isnan(k)}  # 去除nan 335
    return map_dict[id_2021]


# ⭐️将结果存入csv
def save_to_csv(save_path, columns, data):
    '''
    :param save_path: 路径 + csv文件名
    :param columns: 列名数组，如['A', 'B', 'C']
    :param data: 需要追加的行（数组形式），如：[10, 11, 12]
    '''
    # 判断文件是否存在
    if not os.path.isfile(save_path):
        # CSV 文件不存在，创建新文件并指定列名
        df = pd.DataFrame(columns=columns)
        df.to_csv(save_path, index=False)
        # 调用自己，完成第一行数据插入
        save_to_csv(save_path, columns, data)
    else:
        # 读取已有的 CSV 文件
        df = pd.read_csv(save_path)
        # 将新行数据添加到 DataFrame 对象中
        df.loc[len(df)] = data
        # 将更新后的 DataFrame 保存为 CSV 文件
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # pass
    print(get_2021_by_2019("BraTS19_CBICA_ANZ_1"))

    # input, seg = brats2019_data_reader("BraTS19_2013_16_1", need_onehot=False)
    # print(input.shape)
    # print(input.dtype)
    # print(np.unique(input))
    # show_one_slice(input[0], 60)

    # out = np.ones((1, 4, 128, 128,show 128))
    # tar = np.zeros((1, 4, 128, 128, 128))
    # print('{:.6f}'.format(Dice(out,tar)))

    # print(get_class_punishment_weight("val_2019.txt"))

    # -----------2022.12.3-------------
    # seg_label_to_0_1('BraTS19_TCIA12_249_1', save_path='../datasets/BraTS/BraTS_2019_seg_full_tumor/')

    # data_full = read_nii('../datasets/BraTS/BraTS_2019_seg_full_tumor/BraTS19_TMC_30014_1_seg_full_tumor.nii.gz')
    # data_ori = read_nii(
    #     '../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_TMC_30014_1/BraTS19_TMC_30014_1_seg.nii.gz')
    # print(data_full.dtype, data_ori.dtype)
    # print(data_full.shape, data_ori.shape)
    # print(np.unique(data_full), np.unique(data_ori))
    # show_one_slice(data_full, 120)
    # show_one_slice(data_ori, 120)

    # with open('runtime_file/test_2019.txt', encoding='utf-8') as f:
    #     train_line = f.readlines()
    # ids, _ = data_list_split_label(train_line)
    # print(ids)
    #
    # for id in ids:
    #     seg_label_to_0_1(id, save_path='../datasets/BraTS/BraTS_2019_seg_full_tumor/')

    # gen_all_csv()

    # acc = [
    #     0.9394,
    #     0.9091,
    #     0.9697,
    #     0.9394,
    #     0.9155
    # ]
    # auc = [
    #     0.8519,
    #     0.8696,
    #     0.8704,
    #     0.8870,
    #     0.8224
    # ]
    #
    # print(get_standard_deviation(np.array(auc)))

    # import matplotlib.pyplot as plt
    #
    # # 数据
    # data = [[0.9343, 0.8971, 0.8289, 0.9653, 0.8750, 0.9506, 0.8514, 0.9579],
    #         [0.7701, 0.5910, 0.2632, 0.9189, 0.4878, 0.8095, 0.3419, 0.8608],
    #         [0.8806, 0.8438, 0.7763, 0.9112, 0.7195, 0.9328, 0.7468, 0.9219],
    #         [0.7851, 0.6100, 0.2895, 0.9305, 0.5500, 0.8169, 0.3793, 0.8700],
    #         [0.7851, 0.6053, 0.2763, 0.9344, 0.5526, 0.8148, 0.3684, 0.8705]]
    #
    # # 横坐标标注
    # x_labels = ['Accuracy', 'AUC', 'Recall_LGG', 'Recall_HGG', 'Precision_LGG', 'Precision_HGG', 'F1_LGG', 'F1_HGG']
    # y_labels = ['4 modals', 'T1', 'T1ce', 'T2', 'Flair']
    # # 绘制折线图
    # for i in range(5):
    #     plt.plot(x_labels, data[i], label=y_labels[i])
    #
    # # 设置纵坐标刻度范围
    # plt.ylim(0, 1)
    #
    # # 添加图例
    # plt.legend()
    #
    # # 显示网格线
    # plt.grid(True)
    #
    # # 倾斜 x 坐标轴标签
    # plt.xticks(rotation=45)
    # # 展示图形
    # plt.show()

    # ----------------------------------
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # # 数据
    # data = [[0.9343, 0.8971, 0.8289, 0.9653, 0.8750, 0.9506, 0.8514, 0.9579],
    #         [0.7701, 0.5910, 0.2632, 0.9189, 0.4878, 0.8095, 0.3419, 0.8608],
    #         [0.8806, 0.8438, 0.7763, 0.9112, 0.7195, 0.9328, 0.7468, 0.9219],
    #         [0.7851, 0.6100, 0.2895, 0.9305, 0.5500, 0.8169, 0.3793, 0.8700],
    #         [0.7851, 0.6053, 0.2763, 0.9344, 0.5526, 0.8148, 0.3684, 0.8705]]
    #
    # # 横坐标标注
    # x_labels = ['Accuracy', 'AUC', 'Recall_LGG', 'Recall_HGG', 'Precision_LGG', 'Precision_HGG', 'F1_LGG', 'F1_HGG']
    # y_labels = ['4 modals', 'T1', 'T1ce', 'T2', 'Flair']
    #
    # # 设置 Seaborn 样式
    # sns.set(style="whitegrid")
    #
    # # 绘制折线图
    # for i in range(5):
    #     plt.plot(x_labels, data[i], label=y_labels[i])
    #
    # # 设置纵坐标刻度范围
    # plt.ylim(0, 1)
    #
    # # 添加图例
    # plt.legend()
    #
    # # 倾斜 x 坐标轴标签
    # plt.xticks(rotation=45)
    #
    # # 显示网格线
    # plt.grid(True)
    #
    # # 调整图像布局，防止字符超出图像最大范围
    # plt.tight_layout()
    #
    # # 展示图形
    # plt.show()
    # ----------------------------------
