#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/21 13:46
# @Author  : oliver
# @File    : data_transform.py
# @Software: PyCharm
"""
与dataprocessor类似，但是transform是可逆的变换，甚至可以在模型中直接写出。
transform只对数据整体生效，一般不具有根据情况处理的逻辑
一般不会在训练过程中应用transform（应用也很难控制）最后选择numpy作为标准
"""
import torch
import pandas as pd
import numpy as np


class Transform(object):
    name = "DftTransform"
    def __init__(self):
        self.eps = None
        self.shape = None

    def fit(self, x, **kwargs):
        if isinstance(x, np.ndarray):
            self.eps = np.finfo(x.dtype).eps
        else:
            self.eps = np.finfo(np.dtype('float64')).eps
        return self

    def set_para(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def transform(self, x, **kwargs):
        y = x
        return y

    def inverse_transform(self, y, y_slice=slice(None), **kwargs):
        x = y
        return x

    def fit_trans(self, x, **kwargs):
        para = self.fit(x, **kwargs)
        rtn = self.transform(x, **kwargs)
        return para,rtn

    def get_shape(self):
        rtn = list(self.shape)
        # rtn[self.dim] = "*"
        return rtn

    def to_dict(self):
        return {self.name:{}}

class StandardScaler(Transform):
    name = "StandardScaler"
    # eps = torch.finfo(dft_cfg.default_dtype).eps

    def __init__(self):
        super().__init__()
        self.mean = 0
        self.std = 1
        self.dim = None

        # self.shape = None
        # self.noise_std = 1e-2
        # self.noise_std = 1e-1

    def fit(self, x:np.array, dim=1, **kwargs):
        """
        计算关键参数以备转换，参数会保存在内部
        """
        view = x.reshape([-1, x.shape[-1]])
        super().fit(view, **kwargs)
        self.mean = view.mean(axis=0)
        self.std = view.std(axis=0).clip(min=self.eps)
        # self.dim = dim
        return {"mean": self.mean, "std": self.std}

    def transform(self, x, **kwargs):
        """
        对输入数据进行标准化转换
        """
        view = x.reshape([-1, x.shape[-1]])
        view[:] = (view - self.mean) / self.std
        return x

    def inverse_transform(self, y, y_slice=slice(None), **kwargs):
        """
        对标准化后的数据进行逆转换
        """
        view = y.reshape([-1, y.shape[-1]])
        view[:] = (view * self.std) + self.mean
        # y = y.to(self.device)
        return y

    def get_shape(self):
        rtn = list(self.shape)
        # rtn[self.dim] = "*"
        return rtn

class MaxMin0to1Scaler(Transform):
    name = "MaxMin0to1Scaler"
    # eps = torch.finfo(dft_cfg.default_dtype).eps

    def __init__(self):
        super().__init__()
        self.min = 0
        self.max = 1
        self.dim = None
        # self.shape = None
        # self.eps = 0
        self.scaler = 0.999
        # self.device = torch.device("cpu")
        # self.noise_std = 1e-2
        # self.noise_std = 1e-1

    def fit(self, x:np.array, dim=1, **kwargs):
        """
        计算关键参数以备转换，参数会保存在内部
        """
        view = x.reshape([-1, x.shape[-1]])
        super().fit(x, **kwargs)
        self.min = view.min(0)
        self.max = view.max(0)
        # self.shape = x.shape
        # self.dim = dim
        # self.device = x.device
        return {"min": self.min, "max": self.max}

    def transform(self, x, **kwargs):
        """
        对输入数据进行标准化转换
        """
        view = x.reshape([-1, x.shape[-1]])
        view[:] = (view - self.min) / (self.max - self.min) * self.scaler
        return x

    def inverse_transform(self, y:np.array, y_slice=slice(None), **kwargs):
        """
        对标准化后的数据进行逆转换
        """
        view = y.reshape([-1, y.shape[-1]])
        view[:] = (self.max - self.min) * view/self.scaler + self.min
        # y = y.to(self.device)
        return y

    def get_shape(self):
        rtn = list(self.shape)
        # rtn[self.dim] = "*"
        return rtn


# class Differential(Transform):
#     # eps = torch.finfo(dft_cfg.default_dtype).eps
#
#     def __init__(self):
#         super().__init__()
#         self.origin = None
#         # self.mean = 0
#         # self.std = 1
#         self.dim = None
#         # self.shape = None
#         self.slice_origin = None
#         # self.device = torch.device("cpu")
#         # self.slice_left = None
#         # self.slice_right = None
#
#     def fit(self, x:pd.DataFrame, dim=1, **kwargs):
#         """
#         计算关键参数以备转换，参数会保存在内部
#          这里只是确定原点的位置
#
#         """
#         super().fit(x, **kwargs)
#         # self.device = x.device
#         # self.dim = dim
#         # if dim > 0:
#         #     self.slice_origin = [slice(None)] * (dim) + [[-1]]
#         # else:
#         #     self.slice_origin = [Ellipsis] + [[-1]] + [slice(None)] * (-1 - dim)
#         #
#         # self.origin = x[self.slice_origin]
#         self.origin = x.iloc[0,:]
#         return self
#
#     def transform(self, x_in, flag_out=False, **kwargs):
#         """
#         对输入数据进行标准化转换
#         """
#         if flag_out:
#             return torch.diff(torch.concat([self.origin, x_in], dim=self.dim), dim=self.dim)
#         else:
#             return torch.diff(x_in, dim=self.dim)
#         # relative_x = x_in - self.origin
#         # return relative_x[self.slice_left] - relative_x[self.slice_right]
#         # return (x - self.mean) / self.std
#
#     def inverse_transform(self, y,y_slice=slice(None), flag_out=True, **kwargs):
#         """
#         对标准化后的数据进行逆转换
#         """
#         y = y.to(self.device)
#         if flag_out:
#             # 如果是输出数据 直接累加y 然后总体加上origin即可
#             return torch.cumsum(y, dim=self.dim) + self.origin[y_slice]
#         else:
#             sum_y = torch.sum(y, dim=self.dim, keepdim=True)
#             reversed_sum = torch.cumsum(y, dim=self.dim) - sum_y
#             out = torch.concat([-sum_y, reversed_sum], dim=self.dim)
#             return out + self.origin[y_slice]
#         # return y * self.std + self.mean
#
#     def get_shape(self):
#         rtn = list(self.shape)
#         rtn[self.dim] = "*"
#         return rtn

# class GlobalStandardScaler(StandardScaler):
#     def fit(self, x, **kwargs):
#         """
#         只保留最后一个维度
#         """
#         # super().fit(x, **kwargs)
#         Transform.fit(self, x, **kwargs)
#         tmp_x = x.contiguous().view(-1, x.shape[-1])
#         self.mean = torch.mean(tmp_x, dim=-2, keepdim=True)
#         self.std = torch.sqrt(torch.var(tmp_x, dim=-2, keepdim=True) + self.eps)
#         # self.shape = x.shape
#         self.dim = -1
#         # self.device = x.device
#         return self
#
#     def inverse_transform(self, y, y_slice=slice(None), **kwargs):
#         """
#         对标准化后的数据进行逆转换
#         """
#         y=y.to(self.device)
#         return y * self.std + self.mean
#
# class GlobalMaxMin0to1Scaler(MaxMin0to1Scaler):
#
#     def fit(self, x, **kwargs):
#         # super(Transform).fit(x, **kwargs)
#         Transform.fit(self, x, **kwargs)
#         tmp_x = x.contiguous().view(-1, x.shape[-1])
#         self.min = torch.min(tmp_x, dim=-2, keepdim=True)[0]
#         self.max = torch.max(tmp_x, dim=-2, keepdim=True)[0] + self.eps
#         # self.shape = x.shape
#         # self.device = x.device
#         return self
#
#

class TransformFactory(object):
    def __init__(self):
        self.member_dict = {StandardScaler.name: StandardScaler,
                            MaxMin0to1Scaler.name: MaxMin0to1Scaler}

    def create(self, name, **kwargs):
        if name in self.member_dict:
            return self.member_dict[name](**kwargs)
        else:
            return None



