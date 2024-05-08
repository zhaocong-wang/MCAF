import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BraTS2019
from models.MCAF import MCAF


def get_y_true_y_pred_y_prob(model, device, loader):
    y_true = []
    y_pred = []
    y_prob = []

    softmax = nn.Softmax()
    model.eval()
    with torch.no_grad():
        for i_step, (x, y) in enumerate(loader):
            prediction = model(x.to(device=device))

            _, preds = torch.max(prediction, 1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(softmax(prediction)[:, 1].cpu().numpy())

    return y_true, y_pred, y_prob


def get_eval_info(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)[1]
    precision = precision_score(y_true, y_pred, average=None)[1]
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print('Accuracy=', accuracy)
    print('F1=', f1)
    print('AUC:', auc)
    print('recall=', recall)
    print('precision=', precision)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device：', device)

    model = MCAF(in_channel=4, num_classes=2, pretrained=False).to(device)

    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    for k in range(5):
        weight_path = "weights/fold_{}_MCAF.pth".format(k + 1)
        weights = torch.load(weight_path)
        model.load_state_dict(weights)
        print("fold:", k + 1)
        print("weight_path:", weight_path)

        test_dataloader = DataLoader(
            BraTS2019("../datasets/BraTS/MICCAI_BraTS_2019_Data_Training/",
                      "runtime_files/5_fold_{}_test_2019.txt".format(k + 1)),
            num_workers=2)

        y_true, y_pred, y_prob = get_y_true_y_pred_y_prob(model, device, test_dataloader)
        get_eval_info(y_true, y_pred)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    print('----------------------------------------------------------')
    print('                                                          ')
    print('----------------------------------------------------------')
    get_eval_info(y_true_all, y_pred_all)
    print('----------------------------------------------------------')
    print('                                                          ')
    print('----------------------------------------------------------')
