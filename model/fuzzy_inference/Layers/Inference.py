#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Inference.py
# @Time      :2023/6/14 6:53 PM
# @Author    :Oliver
import torch
# from Frox.BasicType import Layer
# from Frox.config import cfg_instance as cfg
# from config.model_config.Configuration import TopConfig as cfg
from .. import Function as Func
from ..Function import MultRulePatternConv1d, ExponentialWeightedMean, GaussianFunction

"""
推理层: 按定义--根据x计算针对不同规则的隶属度，然后在defuzzifier层将这个隶属度与输入的“初始”隶属度进行tnorm运算（乘<default>或取最小）
       每次计算的结果需要拼接在mu_f后方
       对于x输入和输出的tensor形状是相同的
       [!]由于隶属度总是在0-1之间，如果进行多层的推理容易由于精度有限而归零，建议使用放大器对每个batch的隶属度进行放缩(但是效果不算明显)

x_, [..., rule_dim, *[in_shape]]
Output:
mu_f: [..., rule_dim, mu_f_dim]
mapping: 输出形状与输入一致 函数和输入之间是一一对应关系
"""



class _BasicInferenceLayer(torch.nn.Module):

    def __init__(self, factory_kwargs):
        super().__init__()
        self.factory_kwargs = factory_kwargs

    @staticmethod
    def slope(x_):
        # return torch.sigmoid(x_ * 4 - 2)
        return (torch.tanh((x_ * 2 - 1)) + 1) / 2


class GaussianMappingInferenceLayer(_BasicInferenceLayer):
    """
    一元实值函数
    [...,*shape]->[...,*shape]
    sigma range: U~[sigma_min, sigma_max]
    """
    def __init__(self, shape, sigma_min=1, sigma_max=10,  **kwargs):
        super().__init__(kwargs)
        self.mean = torch.nn.Parameter(torch.randn(shape, **self.factory_kwargs))
        self.sigma = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs) * (sigma_max-sigma_min) + sigma_min)

    def forward(self, x):
        return torch.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2))

class TrapezoidalMappingInferenceLayer(_BasicInferenceLayer):

    def __init__(self, shape, sigma=1, mean=0, **kwargs):
        super().__init__(kwargs)
        a, b, c, d = torch.msort(torch.randn([4, *shape], **self.factory_kwargs) * sigma + mean)
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.c = torch.nn.Parameter(c)
        self.d = torch.nn.Parameter(d)

    def forward(self, x):
        m = self.slope((x - self.a) / (self.b - self.a))
        n = self.slope((x - self.d) / (self.c - self.d))
        return m * n

class StrictlyTrapezoidalMappingInferenceLayer(_BasicInferenceLayer):
    def __init__(self, shape, **kwargs):
        super().__init__(kwargs)
        self.center = torch.nn.Parameter(torch.randn(shape, **self.factory_kwargs))
        self.slope_up = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))
        self.topPlat_len = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))
        self.slope_down = torch.nn.Parameter(torch.rand(shape, **self.factory_kwargs))

        # self.slope = _slope_dict[slope]
        # self.reset_parameters()

    def forward(self, x):
        slope_up = torch.exp(self.slope_up)
        slope_down = -torch.exp(self.slope_down)
        center = self.center
        top_plat_len = torch.nn.LeakyReLU()(self.topPlat_len)
        m = self.slope(1 + slope_up * (x - center + top_plat_len / 2))
        n = self.slope(1 + slope_down * (x - center - top_plat_len / 2))
        return m * n


class SlopeMappingInferenceLayer(_BasicInferenceLayer):

    def __init__(self, shape, a_sigma=1, a_mean=0,b_sigma=1, b_mean=0, **kwargs):
        super().__init__(kwargs)
        # a, b = torch.msort(torch.randn([2, *shape], **self.factory_kwargs))
        a = torch.randn(shape, **self.factory_kwargs)* a_sigma + a_mean
        b = torch.randn(shape, **self.factory_kwargs)* b_sigma + b_mean
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        # self.slope = _slope_dict[slope]
        # self.reset_parameters()

    def forward(self, x):
        m = self.slope((x - self.a) / (self.b - self.a))
        return m


class TimeGaussianInferenceLayer(_BasicInferenceLayer):
    """
    input:  [batch, time_dim, x_dim]
    output: [batch, rule_num, membership]

    """
    def __init__(self, in_channels, time_length, pattern_num, rule_num, time_horizon=1, ewm_k=5,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(kwargs)
        self.rule_pattern_conv = MultRulePatternConv1d(in_channels, pattern_num, rule_num,time_horizon,
                                                       stride, padding, dilation, groups, bias)
        self.time_length = time_length
        self.pattern_out_length = self.rule_pattern_conv.out_length(time_length)
        self.membership_func = GaussianFunction([rule_num, pattern_num, self.pattern_out_length])
        # self.softmax = torch.nn.Softmax(dim=-2)
        self.ewm = ExponentialWeightedMean(ewm_k, sum_dim=-2)

    def forward(self, x):
        pattern_graph = self.rule_pattern_conv(x.transpose(-2, -1))
        membership = self.membership_func(pattern_graph)

        # membership = self.softmax(membership)
        rule_membership = self.ewm(membership)
        return rule_membership












class BasicInferenceLayer(torch.nn.Module):
    """
    x:      [...,   rule_dim==1,        x_dim==x_in_dim]
    mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
    mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
    """

    def __init__(self, interface_function):
        super().__init__()
        # self.rule_num = rule_num
        self.antecedent_membership_function = interface_function

    def forward(self, x_):
        return self.antecedent_membership_function(x_)

# 
# 
# class GaussianInferenceLayer(BasicInferenceLayer):
#     """
#     x:      [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
#     """
# 
#     def __init__(self, x_dim, rule_num):
#         shape = [rule_num] + [x_dim] if isinstance(x_dim, int) else [rule_num] + x_dim
#         super().__init__(Func.GaussianFunction(shape))
#         # self.setAntFunc(Func.GaussianFunction(shape, Mean, Sigma, FixedMean, FixedSigma))
# 
#         # self.setAntFunc()
# 
# 
# class TrapInferenceLayer(BasicInferenceLayer):
#     """
#     x:      [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
#     """
# 
#     def __init__(self, x_dim, rule_num, slope="Tanh"):
#         shape = [rule_num] + [x_dim] if isinstance(x_dim, int) else [rule_num] + x_dim
#         super().__init__(Func.TrapezoidFunction(shape, slope))
#         self.setAntFunc()
# 
# 
# class HalfTrapInferenceLayer(BasicInferenceLayer):
#     """
#     x:      [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
#     """
# 
#     def __init__(self, x_dim, rule_num, slope="Tanh"):
#         shape = [rule_num] + [x_dim] if isinstance(x_dim, int) else [rule_num] + x_dim
#         super().__init__(Func.StepFunction(shape, slope))
# 
# 
# class StrictlyTrapInferenceLayer(BasicInferenceLayer):
#     """
#     x:      [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
#     mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
#     """
# 
#     def __init__(self, x_dim, rule_num, slope="Tanh"):
#         shape = [rule_num] + [x_dim] if isinstance(x_dim, int) else [rule_num] + x_dim
#         super().__init__(Func.StrictlyTrapFunction(shape, slope))


class MultGaussianInferenceLayer(BasicInferenceLayer):
    def __init__(self, x_dim, rule_num):
        super().__init__(Func.GaussianFunctionMultiple(x_dim, rule_num, x_dim))


class MultChannelGaussianInferenceLayer(BasicInferenceLayer):
    def __init__(self, x_dim, rule_num, channel_num):
        super().__init__(Func.GaussianFunctionMultiple(x_dim, [channel_num,rule_num], x_dim))

class MultiTimeGaussianInferenceLayer(BasicInferenceLayer):
    def __init__(self, x_dim, rule_num):
        super().__init__(Func.TimeGaussianFunctionMultiple(x_dim, rule_num))