import os
from typing import Optional, Callable, TypeVar, Any, Dict, List
import torch
import torchvision
from torch import nn
from torch.nn import Conv3d, Dropout, Linear
from torchinfo import summary
from torchvision.models._api import register_model
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.video import MViT_V2_S_Weights, MViT
from torchvision.models.video.mvit import MSBlockConfig, _mvit, mvit_v2_s, MViT_V1_B_Weights

from models.resnet_mixed_convolution import r3d_18


# 重写原始函数
@register_model()
@handle_legacy_interface(weights=("pretrained", MViT_V2_S_Weights.KINETICS400_V1))
def my_mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> MViT:
    """
    Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
        :members:
    """
    weights = MViT_V2_S_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(128, 128),  # 🟢 spatial_size: The spacial size of the input as (H, W) 修改为自己数据的尺寸
        temporal_size=128,  # 🟢 temporal_size: The temporal size ``T`` of the input. 修改为自己数据的尺寸
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )


@register_model()
@handle_legacy_interface(weights=("pretrained", MViT_V1_B_Weights.KINETICS400_V1))
def my_mvit_v1_b(*, weights: Optional[MViT_V1_B_Weights] = None, progress: bool = True, **kwargs: Any) -> MViT:
    """
    Constructs a base MViTV1 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V1_B_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V1_B_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V1_B_Weights
        :members:
    """
    weights = MViT_V1_B_Weights.verify(weights)

    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "output_channels": [192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768, 768],
        "kernel_q": [[], [3, 3, 3], [], [3, 3, 3], [], [], [], [], [], [], [], [], [], [], [3, 3, 3], []],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=(128, 128),
        temporal_size=128,
        block_setting=block_setting,
        residual_pool=False,
        residual_with_cls_embed=False,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )


def get_mvit_v2_s(in_channel, num_classes):
    model = my_mvit_v2_s(num_classes=num_classes)
    model.conv_proj = Conv3d(in_channel, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
    return model


def get_mvit_v1_b(in_channel, num_classes):
    model = my_mvit_v1_b(num_classes=num_classes)
    model.conv_proj = Conv3d(in_channel, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备：', device)

    # 模型初始化
    # model = get_mvit_v2_s(in_channel=4, num_classes=2).to(device)
    model = get_mvit_v1_b(in_channel=4, num_classes=2).to(device)
    print(model)

    input = torch.rand(2, 4, 128, 128, 128).to(device)
    output = model(input)
    print('output=', output.shape, type(output), output.dtype, output)

    # model = r3d_18(in_channel=4, num_classes=2, pretrained=False).to(device)
    # output = model(input)
    # print('output=', output.shape, type(output), output.dtype, output)

    # 输出模型架构
    summary(model, (4, 128, 128, 128))  # 打印模型结构
