#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/16 18:10
# @Author  : oliver
# @File    : data_reader.py
# @Software: PyCharm
import json

import pandas as pd
from data_source.Interface.DatabaseHandle import FlightPathDataHandle, FlightPropertyDataHandle, DatabaseHandle
from tqdm.auto import tqdm
import os
import torch
# from ..basic_proto import *
from ..basic_protocal import *

class DataReader:
    def __init__(self, key_map=None, time_key_name="time"):
        self.key_map = key_map  # 用于指定提取那些列以及这些列之后的命名
        self.time_key_name = time_key_name

        self.data = Data()
        self._info = {}
        # self.data = Data(show_input_sort_id=True)

        # self.raw_data: pd.DataFrame
        # self.data_buffer_len = 0
        # self.info_dict = {}
        
        
    def buffer_add_track(self, track: pd.DataFrame):
        if self.key_map is not None:
            track = track.loc[:, self.key_map.keys()].rename(columns=self.key_map)
        if track.isna().all().any():
            # 若提取的表中有列全部都是空值（很宽松的设定了）, 那就不做
            return 0

        # track["track_id"] = self.data_buffer_len
        track[self.time_key_name] = pd.to_datetime(track[self.time_key_name])
        self.data.append(track)

        return len(track)

    def read(self, *args, **kwargs):
        return self.data

    @property
    def info(self):
        return {**self._info, **self.data.gene_info_dict()}
    
    
class DataReaderDatabase(DataReader):
    # 暂时的

    def read(self, database_path, chosen_track_ids:list[int], *args, **kwargs):
        database_path = database_path if database_path is not None else self.from_path
        # self.info_json_dict["db_chosen_track_ids"].append(chosen_track_ids)

        all_path_data = FlightPathDataHandle(database_path)  # 航路信息
        property_data = FlightPropertyDataHandle(database_path)  # 航线信息，按照出发地和目的地分离排列
        self._info.update({"chosen_track_ids": chosen_track_ids})
        tmp_property = []
        tmp_path_data = []
        tmp_info = {}
        for i in chosen_track_ids:
            property_now = property_data[i]
            tracks_hbid = {}
            cfd, ddd = property_now.loc[i, ["CFD", "DDD"]]
            tmp_property.append(property_now)
            with tqdm(total=len(property_now), desc=f"{cfd} -> {ddd}") as pbar:
                for hbid in (property_now.loc[:, "HBID"]):

                    path_data = all_path_data[hbid]
                    if path_data is None:
                        pbar.set_postfix({"hbid": hbid, "state":"Failed"})
                        pbar.update(1)
                        continue
                    if self.buffer_add_track(path_data):
                        tmp_path_data.append(path_data)
                        tracks_hbid.update({len(tmp_path_data) : hbid})

                    pbar.set_postfix({"hbid": hbid, "state":"Success"})
                    pbar.update(1)
            tmp_info.update({f"{cfd} -> {ddd}": tracks_hbid})
            pbar.set_postfix_str(f"finish processing of id {i}")
        # self.info_json_dict.update(tmp_info)
        # self.track_detail_dict.update(tmp_info)
        if "track_detail_dict" in self._info:
            self._info["track_detail_dict"].append(tmp_info)
        else:
            self._info.update({"track_detail_dict": tmp_info})

        return self.data

class DataReaderCSV(DataReader):

    def read(self, csv_path, track_id_key=None):
        """
        :param csv_path:  要读取的csv的路径
        # :param path_key_map: 读取文件本身的列名与读取保存后的列名的映射。不写则默认全部保存为文件本身的列名
        :param track_id_key: 标志csv文件中不同轨迹id的列 None则认为全体为一个轨迹
        :return:
        """
        df = pd.read_csv(csv_path, index_col=None, header=0)
        if track_id_key is None:
            self.buffer_add_track(df)
        else:
            for k, track in df.groupby(track_id_key):
                self.buffer_add_track(track)
            # for track_id in df[track_id_key].unique():
            #     self.buffer_add_track(df[df[track_id_key] == track_id], path_key_map)
        return self.data

    def read_dir(self, csv_dir, track_id_key=None):
        csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if
                     os.path.isfile(os.path.join(csv_dir, f)) and f.endswith('.csv')]
        for csv_file in csv_files:
            self.read(csv_file, track_id_key)

class DataReaderPickle(DataReader):
    def read(self, pt_path):
        tracks = torch.load(pt_path, map_location=torch.device('cpu'), weights_only=True)
        for track in tracks:
            self.buffer_add_track(track)
        return self.data

class DataReaderJSON(DataReader):
    pass

class DataReaderDataGroup(DataReaderPickle):
    def read(self, dg_path:str):
        track_path = os.path.join(dg_path, FILENAME_data_group_track_pt)
        info_path = os.path.join(dg_path,FILENAME_data_group_info)
        csv_path = os.path.join(dg_path,FILENAME_data_group_csv)

        tracks = pd.read_csv(csv_path, index_col=None, header=0)
        # tracks = torch.load(track_path, map_location=torch.device('cpu'), weights_only=True)

        data_columns = [i for i in tracks.columns if i != COLUMN_divide_data]
        self.data.extend([track[data_columns] for _,track in tracks.groupby(COLUMN_divide_data)])
        self.data.update_info_by_json(info_path)
        # df = pd.DataFrame(torch.cat(tracks), columns=self.data.columns)
        #
        # if self.time_key_name in df.columns:
        #     df[self.time_key_name] = pd.to_datetime(df[self.time_key_name])
        #     self.data.time_header = self.time_key_name
        # else:
        #     print("[!] Cannot find the time key in the reading original_data.")
        #
        # if self.data.sort_id_header in df.columns:
        #     df[self.data.sort_id_header] = df[self.data.sort_id_header].astype(int)
        #
        # self.data.raw_data = df

        return self.data


