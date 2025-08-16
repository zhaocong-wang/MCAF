import torch

# 若数据被归一化到[0,1]
def fgsm_attack_old(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 噪声越来越大，机器越来越难以识别，但人眼可以看出差别
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

# 若数据值域未被归一化，开启自适应调整
def fgsm_attack_fixed(image, epsilon, data_grad):
    epsilon = torch.max(image) * epsilon
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image
