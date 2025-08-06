#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Function.py
# @Time      :2023/6/11 8:16 PM
# @Author    :Oliver
# from FuzzyInferenceSystem import *
import torch
from torch.nn import init
import math
import torch.nn.functional as F

# from Frox import Layer
# from Frox.config import cfg_instance as cfg
# from config.model_config.Configuration import TopConfig as cfg

_slope_core_HardTanh = torch.nn.Hardtanh(0, 1)


def _slope_core_tanh(x_):
    return (torch.nn.Tanh()((x_ * 2 - 1)) + 1) / 2


def _slope_core_sigmoid(x_):
    return torch.nn.Sigmoid()((x_ * 4 - 2))


_slope_core_dict = {
    # "HardTanh": slope_core_HardTanh,
    "Tanh": _slope_core_tanh,
    "Sigmoid": _slope_core_sigmoid
}


class _BasicFunction(torch.nn.Module):

    def __init__(self, shape, slope_core="Tanh"):
        super().__init__()
        self.factory_kwargs = {}
        self._default_shape = shape
        self.slope_core = _slope_core_dict[slope_core]


    def forward(self, x):
        return x


# region 一元函数
#
class GaussianFunction(_BasicFunction):
    """
    一元实值函数
    [...,*shape]->[...,*shape]
    """

    def __init__(self, shape, sigma_min=1, sigma_max=10):
        super().__init__(shape)
        # self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.mean = torch.nn.Parameter(torch.randn(shape, **self.factory_kwargs))
        self.sigma = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs)  * (sigma_max-sigma_min) + sigma_min)
        # self.reset_parameters()

    def forward(self, x):
        return torch.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2))
#
#
# class TrapezoidFunction(_BasicFunction):
#     """
#     一元实值函数
#     [...,*shape]->[...,*shape]
#     """
#
#     def __init__(self, shape, *, slope_core="Tanh"):
#         super().__init__(shape,slope_core)
#         # self.factory_kwargs = {'device': device, 'dtype': dtype}
#         a, b, c, d = torch.msort(torch.randn([4, *shape], **self.factory_kwargs))
#         self.a = torch.nn.Parameter(a)
#         self.b = torch.nn.Parameter(b)
#         self.c = torch.nn.Parameter(c)
#         self.d = torch.nn.Parameter(d)
#         # self.reset_parameters()
#
#     def forward(self, x):
#         m = self.slope_core((x - self.a) / (self.b - self.a))
#         n = self.slope_core((x - self.d) / (self.c - self.d))
#         return m * n
#
#
# class StepFunction(_BasicFunction):
#     """
#     一元实值函数
#     [...,*shape]->[...,*shape]
#     """
#
#     def __init__(self, shape, *, slope_core="Tanh"):
#         super().__init__(shape, slope_core)
#         a, b = torch.msort(torch.randn([2, *shape], **self.factory_kwargs))
#         self.a = torch.nn.Parameter(a)
#         self.b = torch.nn.Parameter(b)
#         # self.slope_core = _slope_core_dict[slope_core]
#         # self.reset_parameters()
#
#     def forward(self, x):
#         m = self.slope_core((x - self.a) / (self.b - self.a))
#         return m
#
#
# class StrictlyTrapFunction(_BasicFunction):
#     """
#     一元实值函数
#     [...,*shape]->[...,*shape]
#     """
#
#     def __init__(self, shape, *, slope_core="Tanh"):
#         super().__init__(shape, slope_core)
#
#         self.center = torch.nn.Parameter(torch.randn(shape, **self.factory_kwargs))
#         self.slope_up = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))
#         self.topPlat_len = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))
#         self.slope_down = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))
#
#         # self.slope_core = _slope_core_dict[slope_core]
#         # self.reset_parameters()
#
#     def forward(self, x):
#         slope_up = torch.exp(self.slope_up)
#         slope_down = -torch.exp(self.slope_down)
#         center = self.center
#         topPlat_len = torch.nn.LeakyReLU()(self.topPlat_len)
#         m = self.slope_core(1 + slope_up * (x - center + topPlat_len / 2))
#         n = self.slope_core(1 + slope_down * (x - center - topPlat_len / 2))
#         return m * n


# endregion

# region 多元函数
class ExponentialWeightedMean(torch.nn.Module):
    def __init__(self, k, sum_dim=-1):
        super().__init__()
        self.k = k
        self.sum_dim = sum_dim

    def forward(self, x):
        weights = torch.exp(self.k * x)
        weighted_mean = torch.sum(weights * x, dim=self.sum_dim) / torch.sum(weights,dim=self.sum_dim)
        return weighted_mean

class PatternConv1d(torch.nn.Module):
    """
    ([Batch], 'in_channels', in_length)   ->  ([Batch], 'out_channels', out_length)

    沿着时间轴方向对进行卷积，
    1. 沿着时间轴的方向，每次卷积使用的权重相同（提取的特征相同）
    2. kernel_size 可以决定卷积核跨越时间的长度
    3. 无论输入有几个channel，总是被卷积核直接全部跨过
    4. out_channels 的数量间接决定了pattern的数量
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_core = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,
                                         stride=stride,padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def out_length(self, input_length):
        return (input_length - self.kernel_size + 2 * self.padding) // self.stride + 1

    def forward(self, x):
        conv_out = self.conv_core(x)
        return conv_out

class MultRulePatternConv1d(torch.nn.Module):
    """
    ([Batch], 'in_channels', in_length)   ->  ([Batch], 'rule_num', 'pattern_num', out_length)

    沿着时间轴方向对进行卷积，但是输出额外分离出rule这一维度，其他与PatternConv1d相同
    """
    def __init__(self, in_channels, pattern_num, rule_num, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        out_channels = pattern_num * rule_num
        self.conv_core = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,
                                         stride=stride,padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pattern_num = pattern_num

    def out_length(self, input_length):
        return (input_length - self.kernel_size + 2 * self.padding) // self.stride + 1

    def forward(self, x):
        conv_out = self.conv_core(x)
        return torch.stack(torch.split(conv_out, self.pattern_num, dim=-2), dim=-3)



class TimeConv1d(torch.nn.Module):
    """
    ([Batch], in_state, time_dim)   ->  ([Batch], channel(rule), ~=time_dim/kernel_size/stride)
     channel_features 对应一个fls的所有规则
    沿着时间轴方向对进行卷积，kernel_size 可以决定卷积核跨越时间的长度
    """
    def __init__(self, in_state: int, channel: int,kernel_size=1, bias: bool = True) -> None:
        super().__init__()
        self.conv_core = torch.nn.Conv1d(in_channels=in_state, out_channels=channel,
                                         kernel_size=kernel_size, stride=1,bias=bias)
        # self.out_shape = (channel_features, out_features)

    def forward(self, x):
        conv_out = self.conv_core(x)
        return conv_out

class ParallelLinear(torch.nn.Module):
    """
    (Batch, in_features, 1 (or out_feature))   ->  (Batch, channel_features, out_features)
    ([Batch], in_state, time_dim)   ->  ([Batch], channel(rule), time_dim)
    warning: Batch是必选项
    channel_features 对应一个fls的所有规则
    可以对输入做线性变换,每个channel线性变换的矩阵不一样（区别于Linear）
    """
    def __init__(self, in_features: int, out_features: int, channel_features=1, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channel_features = channel_features

        self.conv_core = torch.nn.Conv1d(in_channels=in_features, out_channels=channel_features,
                                         kernel_size=1, stride=1,bias=bias)
        self.out_shape = (channel_features, out_features)
    def forward(self, x):
        conv_out = self.conv_core(x.expand(-1,-1,self.in_features))
        return conv_out

# class ParallelLinear(torch.nn.Module):
#     """
#     ([Batch], 1, in_features)   ->  ([Batch], channel_features, out_features)
#     ([Batch], in_state, time_dim)   ->  ([Batch], channel(rule), time_dim)
#      channel_features 对应一个fls的所有规则
#     可以对输入做线性变换,每个channel线性变换的矩阵不一样（区别于Linear）
#     """
#     def __init__(self, in_features: int, out_features: int, channel_features=1, bias: bool = True) -> None:
#         super().__init__()
#         self.conv_core = torch.nn.Conv1d(in_channels=1, out_channels=out_features * channel_features,
#                                          kernel_size=in_features, stride=(1),bias=bias)
#         self.out_shape = (channel_features, out_features)
#     def forward(self, x):
#         conv_out = self.conv_core(x).reshape(-1, *self.out_shape)
#         return conv_out


class GaussianFunctionMultiple(_BasicFunction):
    """
    一阶多元实值函数
    [...,1, last_dim(shape_in)] ->[...,channel, 1]
    channel为rule的数量, shape_in shape_out 控制线性映射维度，通常是相等的
    head_num并行
    """

    def __init__(self, shape_in: int, channel=1, shape_out=None):
        """
        :param shape_in: 最后一维的形状
        :param channel: -
        :param shape_out: 变换后的形状
        """
        shape_out = shape_in if shape_out is None else shape_out
        super().__init__(shape_out)
        # self.linear_axis = linear_axis
        self.linear = ParallelLinear(shape_in, shape_out, channel, bias=True)
        self.base = GaussianFunction([channel, shape_out])

    def forward(self, x):
        # y = self.linear(x)
        y = self.linear(x.transpose(-1,-2))
        rtn = self.base(torch.sum(y ** 2, dim=-1, keepdim=True))
        # rtn = torch.exp(-(torch.sum(y ** 2, dim=-1, keepdim=True)) / 2)
        return rtn

class TimeGaussianFunctionMultiple(_BasicFunction):
    """
    一阶多元实值函数
    [..., time_dim, last_dim(shape_in)] ->[...,channel, time_dim]
    channel为rule的数量, shape_in shape_out 控制线性映射维度，通常是相等的
    head_num并行
    """

    def __init__(self, shape_in: int, channel=1, shape_out=None):
        """
        :param shape_in: 最后一维的形状
        :param channel: -
        :param shape_out: 变换后的形状
        """
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.channel = channel

        shape_out = shape_in if shape_out is None else shape_out
        super().__init__(shape_out)
        # self.linear_axis = linear_axis
        self.linear = TimeConv1d(shape_in, channel, bias=True)
        self.base = GaussianFunction([channel,   1])

    def forward(self, x):
        y = self.linear(x.transpose(-1, -2))
        rtn= self.base(torch.sum(y ** 2, dim=-1, keepdim=True))
        return rtn
#
# class TrapezoidFunctionMultiple(TrapezoidFunction):
#     def __init__(self, shape_in: int, channel=1, shape_out=None,slope_core="Tanh"):
#         """
#         :param shape_in: 最后一维的形状
#         :param channel: -
#         :param shape_out: 变换后的形状
#         """
#         shape_out = shape_in if shape_out is None else shape_out
#         super().__init__(shape_out)
#         # self.linear_axis = linear_axis
#         self.linear = ParallelLinear(shape_in, shape_out, channel, bias=True)
#
#         self.center = torch.nn.Parameter(torch.randn(shape_out, **self.factory_kwargs))
#         self.slope_up = torch.nn.Parameter(torch.rand(shape_out, **self.factory_kwargs))
#         self.topPlat_len = torch.nn.Parameter(torch.rand(shape_out, **self.factory_kwargs))
#         self.slope_down = torch.nn.Parameter(torch.rand(shape_out, **self.factory_kwargs))
#
#         self.slope_core = _slope_core_dict[slope_core]
#
#
#     def forward(self, x):
#         y = self.linear(x)
#         m = self.slope_core((y - self.para_A) / (self.para_B - self.para_A))
#         n = self.slope_core((y - self.para_D) / (self.para_C - self.para_D))
#         return m * n

#
# class StepFunctionMultiple(StepFunction):
#     def __init__(self, shape_in, shape_out=None, AB=None, FixedA=False, FixedB=False,
#                  slope_core="Tanh",linear_axis=-1):
#         shape_out = shape_in if shape_out is None else shape_out
#         super().__init__(shape_out, AB, FixedA, FixedB, slope_core)
#         self.linear = _LinearTransition(shape_in, shape_out, linear_axis)
#
#     def forward(self, x):
#         y = self.linear(x)
#         m = self.slope_core((y - self.para_A) / (self.para_B - self.para_A))
#         return m
#
#
# class StrictlyTrapFunctionMultiple(StrictlyTrapFunction):
#     def __init__(self, shape_in, shape_out, center, slope_up, topPlat_len, slope_down,
#                  Fixed_center=False, Fixed_slope_up=False, Fixed_topPlat_len=False, Fixed_slope_down=False,
#                  slope_core="Tanh",linear_axis=-1):
#         shape_out = shape_in if shape_out is None else shape_out
#         super().__init__(shape_out, center, slope_up, topPlat_len, slope_down,
#                          Fixed_center, Fixed_slope_up, Fixed_topPlat_len, Fixed_slope_down, slope_core)
#         self.linear = _LinearTransition(shape_in, shape_out, linear_axis)
#
#     def forward(self, x):
#         y = self.linear(x)
#         slope_up = torch.exp(self.para_slope_up)
#         slope_down = -torch.exp(self.para_slope_down)
#         center = self.para_center
#         topPlat_len = torch.nn.LeakyReLU()(self.para_topPlat_len)
#         m = self.slope_core(1 + slope_up * (y - center + topPlat_len / 2))
#         n = self.slope_core(1 + slope_down * (y - center - topPlat_len / 2))
#         return m * n

# endregion

if __name__ == '__main__' and True:
    # -*- coding: utf-8 -*-

    func_3 = GaussianFunctionMultiple(2,1,3)
    from matplotlib import cm
    from matplotlib import pyplot as plt
    import numpy as np
    sample_band = 5
    sample_rate = 0.1
    X_raw = np.arange(-sample_band, sample_band, sample_rate)
    Y_raw = np.arange(-sample_band, sample_band, sample_rate)
    X, Y = np.meshgrid(X_raw, Y_raw)
    XY = torch.from_numpy(np.stack([X,Y])).to(dtype=torch.float32).permute(1, 2, 0).unsqueeze(-2)
    Z2 = func_3(XY.view(-1, 1, 2)).reshape(*X.shape).squeeze().detach().numpy()

    fig, ax = plt.subplots()

    # 绘制热度图
    im = ax.imshow(Z2,vmin=0,vmax=1, cmap='rainbow', label='Temperature (°C)')

    # 添加颜色条
    plt.colorbar(im,label="membership degree")

    # 设置 x 轴和 y 轴的刻度
    tick_num = 11
    ax.set_xticks(np.linspace(0, 2*sample_band/sample_rate-1,tick_num))
    ax.set_xticklabels(np.linspace(-sample_band,sample_band,tick_num)*10//1/10)
    ax.set_yticks(np.linspace(0, 2*sample_band/sample_rate-1,tick_num))
    ax.set_yticklabels(np.linspace(-sample_band,sample_band,tick_num)*10//1/10)

    # 设置坐标轴标签
    ax.set_xlabel('X1-axis')
    ax.set_ylabel('X2-axis')

    # 设置标题
    ax.set_title("")

    plt.savefig("../2d_MF.svg",bbox_inches='tight', format='svg', transparent=True)
    plt.savefig("../2d_MF.pdf",bbox_inches='tight', format='pdf', transparent=True)
    plt.savefig("../2d_MF.png",bbox_inches='tight', format='png', transparent=True)

    pass


if __name__ == '__main__' and False:
    from matplotlib import pyplot as plt
    import numpy as np

    xx = np.linspace(-16, 16, 1600)


    def cut_zero2one(x):
        return np.clip(xx, 0, 1)


    x2 = torch.linspace(-20, 20, 100)
    slope_core_dict = {
        # "HardTanh": _slope_core_HardTanh,
        # "Tanh": _slope_core_Tanh,
        # "Sigmoid": _slope_core_Sigmoid,
        "Gaussian": GaussianFunction(1),
    }

    for name_, func in slope_core_dict.items():
        # plt.figure()
        plt.plot(x2, func(x2).detach().numpy() if isinstance(func(x2),torch.Tensor) else func(x2), label=name_)
        plt.ylim(-0.05, 1.2)
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
    # plt.legend()      # 标签
    plt.savefig(f"../funcs.png", transparent=True)
    plt.savefig(f"../funcs.svg", transparent=True)
    plt.savefig(f"../funcs.pdf", transparent=True)
    plt.show()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x, zero2one(x), label="u_2")
    # plt.ylim(-0.05, 1.2)
    # # plt.legend()
    # ax=plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # plt.savefig("../u2.png",transparent=True)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(x, zero2one(u_raw(x)), label="t")
    # plt.ylim(-0.05, 1.2)
    # # plt.legend()
    # ax=plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # plt.savefig("../t.png",transparent=True)
    # plt.show()
