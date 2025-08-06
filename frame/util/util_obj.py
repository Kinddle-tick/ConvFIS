#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/27 13:57
# @Author  : oliver
# @File    : util_obj.py
# @Software: PyCharm
import random
from enum import Enum
import datetime

class StrNumberIdIterator(object):
    def __init__(self, start, end, shuffle=True):
        str_length = len(str(end-1))
        self.id_series = [f"{i:0{str_length}d}" for i in range(start,end)]
        if shuffle:
            random.shuffle(self.id_series)

    def pop(self):
        if len(self.id_series) == 0:
            raise StopIteration(f"Used all of the defined ID in instance {self}. "
                                f"Try to release used ID or extend the ID range")
        return self.id_series.pop(0)

    def release(self, obj):
        self.id_series.append(obj)

def get_str_time(Date=True, Time=True, dateDiv="-", timeDiv="-", datetimeDiv="_"):
    format_str = ""
    if Date:
        format_str += f"%Y{dateDiv}%m{dateDiv}%d"
    if Date and Time:
        format_str += datetimeDiv
    if Time:
        format_str += f"%H{timeDiv}%M{timeDiv}%S"
    return datetime.datetime.now().strftime(format_str)



class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

class MetricStrategy(ExplicitEnum):
    NO = "no"
    TRAIN = "train"
    VALID = "valid"

    EPOCH = "epoch"
    STEP = "step"

class EnableSituationStrategy(ExplicitEnum):
    NONE = 0b000        # 未知
    TRAIN = 0b001       # 训练时
    VALID = 0b010       # 验证时
    TEST = 0b100        # 测试时
    TRAIN_VALID= 0b011
    TRAIN_TEST = 0b101
    VALID_TEST = 0b110
    ALL = 0b111



class EpochProperty(ExplicitEnum):
    NONE = "none"       # 未知
    TRAIN = "train" # 训练时
    VALID = "valid" # 验证时
    TEST = "test"   # 测试时

class SaveStrategy(ExplicitEnum):
    NO = "no"
    # STEPS = "steps"
    EPOCH = "epoch"             # 每个epoch都存
    BEST = "best"               # 最佳的时候存
    BEST_VALID = "best_valid"     # 最佳valid时候存
    BEST_TRAIN = "best_train"   # 最佳train时候存
    CUSTOM = "custom"           # 自定义保存
    # COVER = "cover"             # 覆盖式保存
    # DIVIDE ="divide"            # 分离式保存

class StateEnum(Enum):
    CLOSED = 0              # 意为没有开始过 没有初始化
    INITIALIZING = 1
    ACTIVE = 2
    ANALYZING = 3
    FINISHED = 4            # 意思是至少结束了一次 当前非活跃状态