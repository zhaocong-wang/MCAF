import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio


def z_score_no_0(one_model):
    nozero_num = np.count_nonzero(one_model)
    sum = np.sum(one_model)
    avg = sum / nozero_num

    nozero_numpy = one_model[one_model != 0]
    std = np.std(nozero_numpy)

    mask = one_model.copy()
    mask[mask > 0] = 1
    mask = mask * avg

    output = (one_model - mask) / std
    return output


class BraTS2019(Dataset):
    def __init__(self, data_path, ids_txt_path):
        super(BraTS2019, self).__init__()

        with open(ids_txt_path, encoding='utf-8') as f:
            data_infos = f.readlines()
        self.ids = []
        self.labels = []
        for item in data_infos:
            id = item[0:-1].split(';')[0]
            label = item[0:-1].split(';')[1]
            self.ids.append(id)
            self.labels.append(int(label))

        self.data_path = data_path
        self.center_cut_128 = tio.CropOrPad(128)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        label = self.labels[index]

        modals = ['t1', 't1ce', 't2', 'flair']
        for i, modal in enumerate(modals):
            path = self.data_path + ('HGG' if int(label) else "LGG") + '/' + id + '/' + id + "_" + modal + ".nii.gz"
            modal = tio.ScalarImage(path)  # (1, 240, 240, 155)
            modal = z_score_no_0(modal.numpy()[0])
            modal = np.expand_dims(modal, axis=0)

            if i == 0:
                data = modal
            else:
                data = np.concatenate((data, modal), axis=0)

        data = np.array(data)

        # CXYZ -> CZYX
        data = data.transpose(0, 3, 2, 1)

        data = self.center_cut_128(data)  # (4, 155, 240, 240) -> (4, 128, 128, 128)

        return torch.tensor(data).type(torch.FloatTensor), torch.tensor(int(label))


if __name__ == '__main__':
    dataset = BraTS2019("../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/", "runtime_files/5_fold_1_test_2019.txt")
    print(len(dataset))

    data, label = dataset[0]
    print(data.shape, label)
