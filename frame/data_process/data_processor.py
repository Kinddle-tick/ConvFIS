import warnings

import datasets
import pandas as pd
from data_source.Interface.DatabaseHandle import FlightPathDataHandle, FlightPropertyDataHandle, DatabaseHandle
from tqdm.auto import tqdm
import os
import torch

from ..basic_protocal import Data
# from ..interface import IData
"""
用于对数据进行各种预处理
Preprocess data
"""

class Processor(object):
    name = "data_processor_default"
    def __init__(self,**kwargs):
        self.detail = {**kwargs}
        for k,v in self.detail.items():
            self.__setattr__(k,v)

    def process(self, data, *args,**kwargs):
        data = data.copy()
        new_samples = []
        with tqdm(total=len(data), desc=f"processing track in {self.name}") as pbar:
            success_count=0
            fail_count=0
            for i, sample in enumerate(data):
                new_sample = self._process_by_sample(sample, i)
                if new_sample is not None:
                    new_samples.append(new_sample)
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "pass": fail_count})
        new_data = Data(data)
        new_data.extend(new_samples)
        # new_data.append_process_step({self.name: self.detail})
        return new_data

    def _process_by_sample(self, sample, i):
        raise NotImplementedError("_process function in BaseClass Processor should not be called")

    def __repr__(self):
        return f"{self.name}: {self.detail}"

    def to_dict(self):
        rtn = {self.name: self.detail}
        return rtn

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
    
    # def __getattr__(self, item):
    #     if hasattr(self,"detail") and item in self.detail:
    #         return self.detail[item]
    #     else:
    #         raise AttributeError(f"Attribute {item} not found")

class DeduplicateProcessor(Processor):
    """
    去重的方法
    Deduplicate
    """
    name = "Deduplicate"
    def __init__(self, rule = "first", columns = ("time",), **kwargs):
        """
        :param keep: the rule used in drop_duplicates
        :param deduplicate_columns: para subset of drop_duplicates
        :param kwargs:
        """
        # self.keep = keep
        # self.detail["rule"] = self.keep
        # self.detail["columns"] = list(deduplicate_columns)
        super().__init__(rule=rule, columns=list(columns))
    # def process(self, data:Data, *args):
    #     new_samples = []
    #     for sample in data:
    #         new_sample = sample.drop_duplicates(subset=self.detail["columns"], keep=self.detail["rule"], inplace=False)
    #         new_samples.append(new_sample)
    #     new_data = Data(data)
    #     new_data.extend(new_samples)
    #     new_data.append_process_step({self.name: self.detail})
    #     return data

    def _process_by_sample(self, sample, i):
        rtn = sample.drop_duplicates(subset=self.detail["columns"], keep=self.detail["rule"], inplace=False)
        # rtn = sample.drop_duplicates(subset=self.detail["columns"], keep=self.detail["rule"], inplace=False)
        return rtn

class MaskProcessor(Processor):
    """
    根据条件筛选数据，不满足条件的数据会被调节为NaN
    """
    name="Mask"

    def __init__(self, lim_dict:dict=None, **kwargs):
        """
        :param lim_dict: usage -- {"column_name":(min,max),"column_name":(min,max),"column_name":(min,max),... }
        """
        self.lim_dict = {} if lim_dict is None else lim_dict
        # self.detail["lim_dict"] = self.lim_dict
        super().__init__(lim_dict = self.lim_dict)

    def set_lim(self, key, min_=None, max_=None):
        self.lim_dict[key] = (min_, max_)
        self.detail["lim_dict"] = self.lim_dict
        return self

    def process(self,data:Data, *args):
        data = data.copy()
        raw_data = data.get_total_data()
        lim_dict = self.detail["lim_dict"]
        for k, (min_, max_) in lim_dict.items():
            if k in raw_data.columns:
                raw_data[k] = raw_data[k].mask(raw_data[k] > max_)
                raw_data[k] = raw_data[k].mask(raw_data[k] < min_)
        data.replace_total_data(raw_data)
        # data.append_process_step({self.name: self.detail})
        return data

class ClipProcessor(Processor):
    """
    根据条件限定数据上下界，不满足条件的数据会被调节为最高值
    """
    name = "Clip"

    def __init__(self, lim_dict:dict=None, **kwargs):
        """
        :param lim_dict: usage -- {"column_name":(min,max),"column_name":(min,max),"column_name":(min,max),... }
        """
        self.lim_dict = {} if lim_dict is None else lim_dict
        # self.detail["lim_dict"] = self.lim_dict
        super().__init__(lim_dict = self.lim_dict)

    def set_lim(self, key, min_=None, max_=None):
        self.lim_dict[key] = (min_, max_)
        self.detail["lim_dict"] = self.lim_dict
        return self

    def process(self,data:Data, *args):
        data = data.copy()
        raw_data = data.get_total_data()
        lim_dict = self.detail["lim_dict"]
        for k, (min_, max_) in lim_dict.items():
            if k in raw_data.columns:
                raw_data[k] = raw_data[k].clip(min_, max_)
        data.replace_total_data(raw_data)
        # data.append_process_step({self.name: self.detail})
        return data

class ResampleProcessor(Processor):
    name= "TrackResample"
    def __init__(self, time_seq_final = "4s",time_seq_unit = "1s",
                 interpolate_kwargs = None,
                 min_raw_data_len = 10,min_final_data_len = 15,
                 time_key="time"):
        super().__init__()
        self.time_key = time_key
        self.min_raw_data_len = min_raw_data_len
        self.min_final_data_len = min_final_data_len
        self.time_seq_unit = time_seq_unit
        self.time_seq_final = time_seq_final
        if interpolate_kwargs is None:
            self.interpolate_kwargs = {"method": "polynomial", "order": 2}
        else:
            self.interpolate_kwargs = interpolate_kwargs
        super().__init__(time_seq_unit=time_seq_unit, time_seq_final=time_seq_final,
                         min_raw_data_len=min_raw_data_len, min_final_data_len=min_final_data_len,
                         interpolate_kwargs=self.interpolate_kwargs,
                         time_key=time_key)
        # self.detail["time_seq_unit"] = self.time_seq_unit
        # self.detail["time_seq_final"] = self.time_seq_final
        # self.detail["raw_data_min_len"] = self.min_raw_data_len
        # self.detail["final_data_min_len"] = self.min_final_data_len
        # self.detail["interpolate_kwargs"] = self.interpolate_kwargs

    def _process_by_sample(self, track, track_id):
        time_key = self.detail["time_key"]
        min_raw_data_len = self.detail["min_raw_data_len"]
        min_final_data_len = self.detail["min_final_data_len"]
        interpolate_kwargs = self.detail["interpolate_kwargs"]
        time_seq_unit = self.detail["time_seq_unit"]
        time_seq_final = self.detail["time_seq_final"]

        track.set_index(time_key, inplace=True)
        if track.index.isna().all():
            # 如果出现时间全为空的情况
            print(f"ignore track id {track_id}: not have enough time data")
            return None
        if len(track.index) < min_raw_data_len:
            print(f"ignore track id {track_id}: too few time data")
            return None

        # resample
        unit_df = track.resample(time_seq_unit).asfreq().interpolate(**interpolate_kwargs)
        new_df = unit_df.resample(time_seq_final).asfreq().dropna()  # 然后按照time_seq_final采样，降低到符合要求的采样时间

        if len(new_df) < min_final_data_len:
            print(f"ignore track id {track_id}: too short processed data")
            return None
        return new_df.reset_index().copy()

class LengthFilterProcessor(Processor):
    name = "LengthFilter"
    def __init__(self, length, pass_when_longer=True):
        super().__init__()
        self.length = length
        self.pass_when_longer = pass_when_longer
        # if flag_less_than:
        #     self.filter = lambda obj: len(obj) < self.length
        # else:
        #     self.filter = lambda obj: len(obj) > self.length
        # self.detail["filter"] = "shorter" if flag_less_than else "longer"
        super().__init__(length=length, pass_when_longer=pass_when_longer)
        # self.detail["length"] = self.length

    def _process_by_sample(self, track, track_id):
        # if self.flag_less_than:
        #     if len(track) <self.length:
        #         return track
        if self.pass_when_longer and len(track) > self.length:
            return track
        elif not self.pass_when_longer and len(track) < self.length:
            return track
        else:
            return None
        # if self.filter(track):
        #     return track
        # else:
        #     # data.data_num-=1
        #     return None

class FilterProcessor(Processor):
    name = "Filter"
    def __init__(self):
        super().__init__()



class ProcessorFactory:
    def __init__(self):
        self.member_dict = {DeduplicateProcessor.name: DeduplicateProcessor,
                            MaskProcessor.name: MaskProcessor,
                            ClipProcessor.name: ClipProcessor,
                            ResampleProcessor.name: ResampleProcessor,
                            LengthFilterProcessor.name: LengthFilterProcessor,}

    def info(self):
        return self.member_dict.keys()

    def create(self, name, **kwargs):
        if name in self.member_dict:
            return self.member_dict[name](**kwargs)
        else:
            return None































