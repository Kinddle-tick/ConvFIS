#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Defuzzifier.py
# @Time      :2023/6/14 10:23 PM
# @Author    :Oliver
# from Frox.BasicType import Layer
# from Frox.config import cfg_instance as cfg
# from config.model_config.Configuration import TopConfig as cfg
import torch
import math
from torch.nn import init

"""
解模糊化
将前文用x和各个规则的隶属函数计算获得的隶属度进行整合，获得输出y
"""

def QTnormMin(Mu_Q):
    return torch.min(Mu_Q, dim=-1)[0]


def QTnormProd(Mu_Q):
    return torch.prod(Mu_Q, dim=-1)

class BasicSingletonDefuzzifierLayer(torch.nn.Module):
    """
    x:      [...,   rule_dim==1,        x_dim==x_in_dim]
    mu_x:   [...,   rule_dim==1,        x_dim==x_in_dim]
    mu_f:   [...,   rule_dim==rule_num, mapping_dim] -> [...,ruleDim, mapping_dim]
    """
    height: torch.nn.Parameter
    tsk_c: torch.nn.Parameter

    def __init__(self, rule_num, out_put_shape: int or list, flag_QTnormMin=False):
        super().__init__()
        self.rule_num = rule_num
        out_put_shape = torch.tensor([out_put_shape] if isinstance(out_put_shape, int) else out_put_shape)
        self.output_shape = [-1, *out_put_shape]
        self.out_put_len = torch.prod(out_put_shape)

        if flag_QTnormMin:
            self.QTnorm = QTnormMin
        else:
            self.QTnorm = QTnormProd

    def shaping_y(self, x):
        return torch.reshape(x, self.output_shape)

    def calculate_phi(self, mu_f):
        # ->[(batch),rule,1]
        Prod_Mu_Q = self.QTnorm(mu_f)
        Phi = (Prod_Mu_Q / torch.sum(Prod_Mu_Q, dim=-1, keepdim=True))  # 每个规则的隶属度 Prod_Mu_Q==0报错
        return Phi[..., None]

    # def forward(self, x_, mu_f,*args):
    #     Phi = self.calculate_phi(mu_f)
    #     y = self.height + torch.sum(self.tsk_c * x_[..., None], dim=-2)
    #     rtn = torch.sum(Phi * y, dim=-2)
    #     return self.shaping_y(rtn)


class SingletonHeightDefuzzifierLayer(BasicSingletonDefuzzifierLayer):
    """
    因为预测y的rank不同对应的操作差异比较大，而在fuzzy的预测中y的位置不是很重要，所以此处选择的方法是将y压扁，最后在输出的时候改成要求的形状
    """
    def __init__(self, rule_num, y_shape: int or list, flag_QTnormMin=False):
        super().__init__(rule_num,y_shape, flag_QTnormMin)
        self.height = torch.nn.Parameter(torch.randn([rule_num, self.out_put_len]))
        # self.tsk_c = torch.nn.Parameter(torch.zeros([1], device=cfg.default_device))

    def forward(self, x_, mu_f):
        Phi = self.calculate_phi(mu_f)
        y = self.height
        rtn = torch.sum(Phi * y, dim=-2)
        return self.shaping_y(rtn)


class SingletonTskDefuzzifierLayer(BasicSingletonDefuzzifierLayer):
    def __init__(self, rule_num, x_dim, y_shape, flag_QTnormMin=False):
        super().__init__(rule_num,y_shape, flag_QTnormMin)
        self.height = torch.nn.Parameter(torch.randn([rule_num, self.out_put_len, ]))
        self.tsk_c = torch.nn.Parameter(torch.randn([rule_num, x_dim, self.out_put_len, ]))

    def forward(self, x_, mu_f):
        Phi = self.calculate_phi(mu_f)
        y = self.height + torch.sum(self.tsk_c * x_[..., None], dim=-2)
        rtn = torch.sum(Phi * y, dim=-2)
        return self.shaping_y(rtn)

