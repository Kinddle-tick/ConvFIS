#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/11 15:40
# @Author  : Oliver
# @File    : basic_protocol.py
# @Software: PyCharm
import json
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import pandas as pd
import datetime

# from frame.data_process.data_processor import ProcessorFactory
# from frame.data_process.data_processor import Processor

FILENAME_data_group_track_pt = "tracklist.pt"
FILENAME_data_group_info = "info.json"
FILENAME_data_group_csv = "all_track.csv"
COLUMN_divide_data = "sample_id"

class _SortIdMap:
    def __init__(self):
        self.id_map=[0]

    def update(self, length):
        self.id_map.append(self.id_map[-1]+length)
        return self

    def __add__(self, other):
        self_len = self.id_map[-1]
        self.id_map.extend([cut_point+self_len for cut_point in other.id_map])
        return self

    def cut_percent(self, percent=0.):
        return self.id_map[int(len(self.id_map)*percent)]

    def __getitem__(self, item):
        if isinstance(item, int):
            return slice(self.id_map[item], self.id_map[item+1])

    def __len__(self):
        return len(self.id_map) -1

class Data(object):
    time_dtype="<M8[ns]"
    def __init__(self, father=None):
        self.data_count=0
        self.build_buffer={}
        self._raw_data:pd.DataFrame = pd.DataFrame()
        self.id_map=_SortIdMap()

        self.gene_time = datetime.datetime.now().strftime(f"%Y-%m-%d_%H:%M:%S")
        self.process_step = ProcessStep().link_data(self)
        self.transform_step = TransformStep().link_data(self)

        # self.columns = None
        if isinstance(father, Data):
            self.inherits(father)

    def inherits(self, other):
        # 只会同步信息，不会同步数据
        self.gene_time = other.gene_time
        self.process_step = other.process_step
        self.transform_step = other.transform_step
        return self

    def _add_data(self, data:pd.DataFrame,build=False):
        self.build_buffer.update({self.data_count: data})
        self.id_map.update(len(data))
        self.data_count += 1
        if build:
            self._build()

    def append(self, data:pd.DataFrame):
        self._add_data(data,True)

    def extend(self, data_list):
        for data in data_list:
            self._add_data(data)
        self._build()

    def _build(self):
        self._raw_data = pd.concat({-1:self._raw_data, **self.build_buffer}, ignore_index=True)
        self.build_buffer.clear()

    def cut_dataset(self, start=0., end=1., with_id=True):
        start = start if isinstance(start,int) else int(len(self.id_map)*start)
        end = end if isinstance(end,int) else int(len(self.id_map)*end)
        i = start
        while i < end:
            if with_id:
                yield self._raw_data.iloc[self.id_map[i]], i
            else:
                yield self._raw_data.iloc[self.id_map[i]]
            i+=1

    def __getitem__(self, item):
        if isinstance(item, (int, np.int64)):
            if item >= self.data_count:
                return None
            else:
                return self._raw_data.iloc[self.id_map[item]]
        else:
            print(f"invalid Data index {item}, type：{type(item)}. need type 'int' or 'np.int64'")
            return None

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self._raw_data.iloc[self.id_map[i]]
            i+=1

    def __len__(self):
        return self.id_map.__len__()

    def __repr__(self):
        return self._raw_data.__repr__()

    def get_total_data(self):
        return self._raw_data

    def replace_total_data(self, data:pd.DataFrame):
        if len(data) != len(self._raw_data):
            raise IndexError("The data length is not equal to the original data length. The Replace mission cannot done")
        else:
            self._raw_data = deepcopy(data)

    def copy(self):
        return deepcopy(self)

    def shuffle(self, seed=42):
        data = Data(self)
        index = list(range(len(self)))
        np.random.seed(seed)
        np.random.shuffle(index)
        for i in index:
            data.append(self[i])
        return data

    def update_info_by_json(self, json_path):
        with open(json_path, 'r') as f:
            info = json.load(f)
        # self.update_info(info)
        # self.data_num = info["data_num"]
        # self.process_step = info["process_step"]
        self.process_step = ProcessStep(info["process_step"]).link_data(self)
        self.transform_step = TransformStep(info["transform_step"]).link_data(self)

    def gene_info_dict(self):
        info = dict()
        info["gene_time"] = self.gene_time
        info["data_num"] = self.data_count
        info["process_step"] = self.process_step
        info["transform_step"] = self.transform_step
        return info

    @property
    def columns(self):
        return self._raw_data.columns


from frame.data_process.data_processor import ProcessorFactory
from frame.data_process.data_transform import TransformFactory

class ProcessStep(dict):
    def __init__(self, history=None, **kwargs):
        super().__init__()
        self.dft_data = None
        self.step = []
        self.reconstruct(history)

    def reconstruct(self,history):
        if history is None:
            return
        else:
            # data = self.dft_data.copy()
            process_factory=ProcessorFactory()
            for i, process_info in history.items():
                for name,paras in process_info.items():
                    processor = process_factory.create(name,**paras)
                    self.record_process(processor)
                    # data = data.process_step.apply(processor)

    def link_data(self, data):
        self.dft_data = data
        return self

    def record_process(self, process):
        self.step.append(process)
        self.clear()
        self.update(self.to_dict())
        # return self

    def apply(self, process):
        data_ = self.dft_data.copy()
        data = process(data_)
        data.process_step.dft_data = data
        data.process_step.record_process(process)
        return data

    def to_dict(self):
        return {i: process.to_dict() for i, process in enumerate(self.step)}

    def __repr__(self):
        rtn = self.to_dict()
        return rtn.__repr__()

class TransformStep(dict):
    def __init__(self, history=None, **kwargs):
        super().__init__()
        # self.dft_data = data
        self.dft_data = None
        self.step = []
        self.para = []
        self.reconstruct(history)
        # self.column_name=[]

    def reconstruct(self,history):
        if history is None:
            return
        else:
            # data = self.dft_data.copy()
            transform_factory=TransformFactory()
            for i, transform_info in history.items():
                for name,paras in transform_info.items():
                    transform = transform_factory.create(name,**paras)
                    self.record_transform(transform, None)  # 理论上可以保存para然后读取，先不做这个工作
                    # data = data.process_step.apply(processor)
    def link_data(self,data):
        self.dft_data=data
        return self

    def record_transform(self, transform, para):
        self.step.append(transform)
        self.para.append(para)
        self.clear()
        self.update(self.to_dict())

    def  apply_overall(self, transform, ignore_columns=("time",)):
        data = self.dft_data.copy()
        column = [i for i in data.columns if i not in ignore_columns]
        overall_data = data.get_total_data()[column]
        para, overall_data = transform.fit_trans(overall_data.to_numpy())
        data.replace_total_data(pd.DataFrame(overall_data, columns=column))

        data.transform_step.dft_data = data
        data.transform_step.record_transform(transform, para)
        return data

    def apply_individual(self, transform, ignore_columns=("time",)):
        data = self.dft_data.copy()
        column = [i for i in data.columns if i not in ignore_columns]
        # seed_df = pd.DataFrame(columns=column)
        paras = {}
        samples = []
        # samples = [seed_df]
        for i, sample in enumerate(data):
            # new_sample = deepcopy(sample)
            para, new_sample = transform.fit_trans(sample[column].to_numpy())
            paras.update({i: para})
            samples.append(pd.DataFrame(new_sample, columns=column))
            # data.raw_data.loc[data.id_map[i], column] = new_sample

        data = Data(data)
        data.extend(samples)
        data.transform_step.dft_data = data
        data.transform_step.record_transform(transform, paras)
        return data

    def apply(self, transform, ignore_columns=("time",), mode="overall"):
        if mode == "overall":
            return self.apply_overall(transform, ignore_columns)
        elif mode == "individual":
            return self.apply_individual(transform, ignore_columns)
        elif mode == "pass":
            return self.dft_data
        else:
            print(f"[-] These is no transform mode {mode},")
            return None


    def _inv_apply(self, data:np.array, i=None):
        if isinstance(data, pd.DataFrame):
            rtn = deepcopy(data)
            use_data = data.to_numpy()
        else:
            use_data = data.copy()

        for trans_step, para in reversed(list(zip(self.step,self.para))):
            if i is not None and i in para.keys():
                trans_step.set_para(**para[i])
            use_data = trans_step.inverse_transform(use_data)

        if isinstance(data, pd.DataFrame):
            rtn.iloc[:,:] = use_data
        else:
            rtn = use_data
        return rtn

    def inv_overall(self, data:np.array):
        data = deepcopy(data)
        return self._inv_apply(data)

    def inv_individual(self, data:Iterable, i=None):
        data = deepcopy(data)
        if i is None:
            samples = []
            for i, sample in enumerate(data):
                new_sample = self._inv_apply(sample,i)
                samples.append(new_sample)

            data = Data(data)
            data.extend(samples)
            data.transform_step.dft_data = data
            return data
        else:
            return self._inv_apply(data, i)

    def inv_by_id(self, data, i):
        data = deepcopy(data)
        return self._inv_apply(data, i)

    def to_dict(self):
        return {i: step.to_dict() for i, step in enumerate(self.step)}

    def __repr__(self):
        rtn = self.to_dict()
        return rtn.__repr__()

if __name__ == '__main__':
    aa = pd.DataFrame([[1,2,3],[4,5,6]])
    bb = pd.DataFrame([[11,22,33],[44,55,66]])
    data_obj = Data(True)
