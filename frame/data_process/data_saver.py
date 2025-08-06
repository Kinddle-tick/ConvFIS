#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/16 18:10
# @Author  : oliver
# @File    : data_saver.py
# @Software: PyCharm
import os
import shutil
from enum import Enum
import datasets
import pandas as pd
from data_source.Interface.DatabaseHandle import FlightPathDataHandle, FlightPropertyDataHandle, DatabaseHandle
from tqdm.auto import tqdm
import os
import torch
from ..basic_protocal import *
from matplotlib import pyplot as plt
import numpy as np
import random
import json
import csv

class NamesakeStrategy(Enum):
    COVER = "cover"
    NUMBER = "number"

class Saver(object):
    save_type = "torch"
    def __init__(self, to_path:str, flag_pt_shuffle=True,
                 # flag_plot_analyze=True,
                 flag_div_track=False,flag_strategy_namesake=NamesakeStrategy.COVER, ):
        self.to_path = to_path
        self.group_root:str = os.path.join(self.to_path)
        self.flag_strategy_namesake = flag_strategy_namesake
        self.flag_pt_shuffle = flag_pt_shuffle
        # self.flag_plot = flag_plot_analyze
        self.flag_div_track = flag_div_track

    def __dir_generate__(self, sub_group_name, *, suffix_int=0):
        # 生成相关基本目录
        if suffix_int == 0:
            tmp_path = os.path.join(self.group_root, sub_group_name)
        else:
            tmp_path = os.path.join(self.group_root, sub_group_name + "_" + str(suffix_int))

        if os.path.exists(tmp_path):
            if self.flag_strategy_namesake == NamesakeStrategy.NUMBER:
                return self.__dir_generate__(sub_group_name, suffix_int=suffix_int + 1)
            else:
                # shutil.rmtree(tmp_path)
                print(f"[!] Covered {tmp_path}")
        else:
            print(f"[!] Created {tmp_path}")

        os.makedirs(tmp_path, exist_ok=True)
        self.sub_group_root = tmp_path
        self.csv_dir = os.path.join(tmp_path, "csv")
        # self.pic_dir = os.path.join(tmp_path, "pic")
        if self.flag_div_track:
            shutil.rmtree(self.csv_dir, ignore_errors=True)
            os.makedirs(self.csv_dir)
        # if self.flag_plot:
        #     os.makedirs(self.pic_dir)
        return os.path.basename(tmp_path)
            # return sub_group_name if suffix_int == 0 else sub_group_name + "_" + str(suffix_int)

    def save_csv(self,data:Data):
        csv_path = os.path.join(self.sub_group_root, FILENAME_data_group_csv)

        csv_header = pd.DataFrame(columns=list(data.columns) + [COLUMN_divide_data])
        csv_header.to_csv(csv_path, index=False)
        tk_id = 0
        for track in data:
            track = track.copy()
            track[COLUMN_divide_data] = tk_id
            track.to_csv(csv_path,mode="a", index=False, header=False)

            tk_id+=1


        # with open(csv_path, "w", newline="") as csv_fp:
        #     header = data.columns
        #     csv_fp.write(",".join(header) + ",sample_id\n")
        #     writer = csv.writer(csv_fp)
        #     for tk_id, track in enumerate(data):
        #         writer.writerows(track)
        #
        #         if self.flag_div_track:
        #             track.to_csv(os.path.join(self.csv_dir, f"tk_{tk_id}.csv"), index=False)

        # processed_data = data.get_total_data()
        # processed_data.to_csv(os.path.join(self.sub_group_root, FILENAME_data_group_csv), index=False)
        # if self.flag_div_track:
        #     if not os.path.exists(self.csv_dir):
        #         os.makedirs(self.csv_dir)
        #     for tk_id, track in processed_data.groupby(data.sort_id_header):
        #         track.to_csv(os.path.join(self.csv_dir, f"tk_{tk_id}.csv"), index=False)
        return

    def save_pt(self,data:Data):
        # processed_data = data.raw_data
        track_list = []
        # for _, track in processed_data.groupby(data.sort_id_header):
        for track in data:
            # track.drop("track_id", axis=1, inplace=True)
            track_save = track.to_numpy(dtype=np.float64)
            if self.save_type == "torch":
                track_save = torch.from_numpy(track_save)
            track_list.append(track_save)
        if self.flag_pt_shuffle:
            random.shuffle(track_list)
            print("[!] Saved pt tracks shuffled")
        torch.save(track_list, os.path.join(self.sub_group_root, FILENAME_data_group_track_pt))

    def save_info(self,data:Data):
        # self.info_json_dict.update({"track_detail": self.track_detail_dict})
        with open(os.path.join(self.sub_group_root, FILENAME_data_group_info), "w", encoding="utf8") as F:
            json.dump(data.gene_info_dict(), F, indent=2, ensure_ascii=False)

    def save(self, data:Data):
        gp_name = self.__dir_generate__(self.to_path)  # 生成保存目录
        print("[+] Saving on {}".format(gp_name))
        # self.processed_buffer = processed_data
        self.save_csv(data)
        self.save_pt(data)
        self.save_info(data)
        print("[+] Save actions over ")

# class CsvSaver(Saver):
#     pass
#
# class PtSaver(Saver):
#     pass
#
# class InfoSaver(Saver):
#     pass


