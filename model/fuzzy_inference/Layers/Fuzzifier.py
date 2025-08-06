#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Fuzzifier.py
# @Time      :2023/6/14 6:16 PM
# @Author    :Oliver
# from Frox.BasicType import Layer
# from Frox.config import cfg_instance as cfg
from .. import Function as Func
import torch

"""
将确定的输入x进行拆解，获得x与其对应的隶属度的初始值(x为其本身及周边的采样，隶属度根据采样产生相应的变化)
对于singletonFuzzifier来说，x即为输入值，而隶属度初始均为1
Output:
x_, mu_x: [..., rule_dim, *[in_shape]]
mu_f: [..., rule_dim, membership==0]
"""


class SingletonFuzzifierLayer(torch.nn.Module):
    """
    def forward:
    x_in:       <-  [...,   x_dim]
    x_:         ->  [...,   1,    x_dim]
    mu_x:       ->  [...,   1,    x_dim]
    mu_f:       ->  [...,   'rule_num', 0]
    主要起到一个格式转化的作用 不一定要使用
    """

    def __init__(self, rule_num):
        super().__init__()
        self.rule_num = rule_num
        # self.xShape = xShape

    def forward(self, x_in):
        x_ = x_in[..., None, :]
        # mu_x = torch.ones(size=x_.shape, device=x_.device)
        mu_f = torch.empty(size=[*x_.shape[:-2], self.rule_num, 0], device=x_.device)
        return x_, mu_f

class TimeFuzzifierLayer(torch.nn.Module):
    """
    def forward:
    x_in:       <-  [...,    x_time_dim,    'x_in_dim']
    x_:         ->  [...,    1,    x_time_dim,   hidden_dim]
    mu_x:       ->  [...,    1,    x_time_dim,   hidden_dim]    # singleton省略
    mu_f:       ->  [...,    'rule_num', 0]
    顺便做了一个将输入线性组合扩展成高维度的功能。
    主要起到一个格式转化的作用 不一定要使用
    """

    def __init__(self, rule_num, x_in_dim, hidden_dim):
        super().__init__()
        self.rule_num = rule_num
        self.x_in_dim = x_in_dim
        self.hidden_dim = hidden_dim
        self.linear_layer = torch.nn.Linear(x_in_dim, hidden_dim)
        # self.xShape = xShape

    def forward(self, x_in):
        x_ = self.linear_layer(x_in)
        # mu_x = torch.ones(size=x_.shape, device=x_.device)
        mu_f = torch.empty(size=[*x_.shape[:-2], self.rule_num, 0], device=x_.device)
        return x_, mu_f

