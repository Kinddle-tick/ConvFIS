#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/5 14:06
# @Author  : Oliver
# @File    : error_fuction.py
# @Software: PyCharm
import numpy as np

class _ErrorCalculator:
    formal_name = "used_when_draw"
    nick_name = "used_everywhere_sample"
    def __init__(self, batch_axis=0, quan_q=0.025):
        self.buffer_error = None
        self.batch_axis = batch_axis
        self.quan_q = quan_q

        self.n=None
        self.mean = None
        self.std = None
        self.quan_up = None
        self.quan_down = None


    def __call__(self,x, y, *args, **kwargs):
        self.calculate(x, y, *args, **kwargs)
        self.update()
        self.buffer_error = None
        return self

    def calculate(self,x,y, *args, **kwargs):
        return self

    def update(self):
        if self.buffer_error is None:
            return
        else:
            self.n = len(self.buffer_error)
            self.mean = np.mean(self.buffer_error, axis=self.batch_axis)
            self.std = np.std(self.buffer_error, axis=self.batch_axis)
            self.quan_up = self.quan(1-self.quan_q, "higher")
            self.quan_down = self.quan(self.quan_q, "lower")


    def quan(self, q, method=None, axis=None):
        if self.buffer_error is None:
            return None
        else:
            return np.quantile(self.buffer_error, q,
                               method="linear" if method is None else method,
                               axis=self.batch_axis if axis is None else axis)

    # @property
    # def quan_up(self):
    #     return self.quan(1-self.quan_q, "higher")
    #
    # @property
    # def quan_down(self):
    #     return self.quan(self.quan_q, "lower")
    #
    # @property
    # def mean(self):
    #     if self.buffer_error is None:
    #         return None
    #     else:
    #         return np.mean(self.buffer_error, axis=self.batch_axis)
    #
    # @property
    # def std(self):
    #     if self.buffer_error is None:
    #         return None
    #     else:
    #         return np.std(self.buffer_error, axis=self.batch_axis)
    #
    # @property
    # def n(self):
    #     if self.buffer_error is None:
    #         return None
    #     else:
    #         return len(self.buffer_error)

class ErrorMse(_ErrorCalculator):
    formal_name = "MSE"
    nick_name = "mse"
    def calculate(self, x, y, *args, **kwargs):
        self.buffer_error = (x - y) ** 2
        return self
        # return self.mean, self.quan_up, self.quan_down


class ErrorMae(_ErrorCalculator):
    formal_name = "MAE"
    nick_name = "mae"
    def calculate(self, x, y, *args, **kwargs):
        self.buffer_error = np.abs(x - y)
        return self.mean, self.quan_up, self.quan_down

class ErrorRMse(_ErrorCalculator):
    formal_name = "RMSE"
    nick_name = "rmse"
    def calculate(self, x, y, *args, **kwargs):
        error = np.sqrt(np.mean((x - y) ** 2, axis=self.batch_axis, keepdims=True))
        self.buffer_error = error
        return error,error,error

class ErrorEuclidean(_ErrorCalculator):
    formal_name = "Euclidean distance"
    nick_name = "euclidean"
    def __init__(self,batch_axis=0, euclid_axis=-1, quan_q=0.025):
        super().__init__(batch_axis,quan_q)
        self.euclid_axis = euclid_axis

    def calculate(self, x, y, *args, **kwargs):
        self.buffer_error = np.sqrt(np.sum(((x - y) ** 2), axis=self.euclid_axis, keepdims=True))
        return self.mean, self.quan_up, self.quan_down

class ErrorMape(_ErrorCalculator):
    formal_name = "MAPE(%)"
    nick_name = "mape"
    def calculate(self, x:np.array, y, *args, **kwargs):
        non_zero = (x != 0)
        # non_zero_mask = non_zero.all(axis = tuple([i for i in range(1,x.ndim)]))
        non_zero_mask = np.all(non_zero, axis = tuple([i for i in range(1,x.ndim)]))
        x = x[non_zero_mask]
        y = y[non_zero_mask]

        self.buffer_error = np.abs((x - y) / x) * 100
        return self.mean, self.quan_up, self.quan_down

