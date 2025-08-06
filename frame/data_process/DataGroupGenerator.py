#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DataGroupGenerator.py
# @Time      :2024/4/2 16:32
# @Author    :Oliver
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data_gene_group.py
# @Time      :2024/2/27 14:36
# @Author    :Oliver
"""
从总数据库生成数据集，以及配套的sqlite csv 数据库部分图片分析等

数据组生成器，一组数据由以下几个部分构成
group   |- track.pt     # track collected by list
        |- info.json
        |- db.sqlite
        |- csv  |- xxx.csv
                |...
        |- pic  |- xxx.png
                |-...
其中有以track.pt最为主要

分为数据读取，数据预处理，数据保存三个方面 此外还有辅助用的数据分析

"""
from enum import Enum

from data_source.Interface.DatabaseHandle import FlightPathDataHandle, FlightPropertyDataHandle, DatabaseHandle
from tqdm.auto import tqdm
import json
from config import get_relative_path, get_str_time
import scipy.signal as signal
import os
import pandas as pd
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
import shutil

db_default_map = {"WZSJ": "time",
                  "JD": "x",
                  "WD": "y",
                  "GD": "h",
                  "SD": "v",
                  }
class NamesakeStrategy(Enum):
    COVER = "cover"
    NUMBER = "number"

class DataReader:
    pass

class DataSaver:
    pass

class DataAnalyze:
    pass

class GroupGenerator(object):

    def __init__(self):
        pass

    def read(self):
        pass



class GroupDatasetGenerator(object):

    # 插值相关
    time_seq_unit = "1s"
    time_seq_final = "4s"
    # interpolate_method = "polynomial"
    interpolate_kwargs = {"method": "polynomial",
                          "order": 2}
    flag_plot = True
    flag_strategy_namesake = NamesakeStrategy.COVER
    time_key_name = "time"
    min_raw_data_len = 10
    min_track_len = 10

    def __init__(self, sub_dataset_name, from_path:str, to_path:str, save_type="torch",*,
                 flag_plot_analyze=True, flag_div_track=False, flag_strategy_namesake=NamesakeStrategy.COVER,):
        """
        用于在总的数据集中提取一部分，组成子数据集。
        :param save_type: "torch" or "numpy", define the type in the pt file    *unused now
        """
        self.sub_dataset_name = sub_dataset_name
        self.from_path = from_path
        self.to_path = to_path
        self.save_type = save_type
        self.flag_plot = flag_plot_analyze
        self.flag_div_track = flag_div_track
        self.flag_strategy_namesake = flag_strategy_namesake
        self.group_root = os.path.join(self.to_path)
        self.info_json_dict = {"gene_time": get_str_time(timeDiv=":"),
                               "time_seq_final": self.time_seq_final,
                               "interpolate_method": self.interpolate_kwargs,
                               "db_chosen_track_ids": []}
        self.track_detail_dict = {}
        self.data_lim_dict = {}
        self.rawdata_buffer = None
        self.data_buffer_len = 0
        self.processed_buffer = None
        self.sub_db_detail = []
        self.sub_db_property = []

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
                shutil.rmtree(tmp_path)
                print(f"Covered {tmp_path}")
        else:
            print(f"Created {tmp_path}")

        os.makedirs(tmp_path)
        self.sub_group_root = tmp_path
        self.csv_dir = os.path.join(tmp_path, "csv")
        self.pic_dir = os.path.join(tmp_path, "pic")
        if self.flag_div_track:
            os.makedirs(self.csv_dir)
        if self.flag_plot:
            os.makedirs(self.pic_dir)
        return os.path.basename(tmp_path)
            # return sub_group_name if suffix_int == 0 else sub_group_name + "_" + str(suffix_int)

    def _save_fig(self, fig, path):
        fig.savefig(path, bbox_inches='tight', format="png")
        fig.savefig(path[:-3] + "svg", bbox_inches='tight', format="svg")
        return

    def buffer_add_track(self, track: pd.DataFrame, key_map=None):
        if key_map is not None:
            track = track.loc[:, key_map.keys()].rename(columns=key_map)
        if track.isna().all().any():
            # 若提取的表中有列全部都是空值（很宽松的设定了）
            return 0

        track["track_id"] = self.data_buffer_len
        track[self.time_key_name] = pd.to_datetime(track[self.time_key_name])
        if self.rawdata_buffer is None:
            self.rawdata_buffer = pd.DataFrame(track, columns=[*track.columns])
            self.info_json_dict.update({"data_columns_info": list(self.rawdata_buffer.columns)})
        else:
            try:
                assert (track.columns == self.rawdata_buffer.columns).all()
            except AssertionError as Z:
                print(f"轨迹的列名不统一:{Z}")
            self.rawdata_buffer = pd.concat([self.rawdata_buffer, track], ignore_index=True)

        self.data_buffer_len += 1
        return len(track)

    def read_database(self, chosen_track_ids, path_key_map: dict = None, database_path=None):
        database_path = database_path if database_path is not None else self.from_path
        self.info_json_dict["db_chosen_track_ids"].append(chosen_track_ids)

        all_path_data = FlightPathDataHandle(database_path)  # 航路信息
        property_data = FlightPropertyDataHandle(database_path)  # 航线信息，按照出发地和目的地分离排列
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
                    if self.buffer_add_track(path_data, path_key_map):
                        tmp_path_data.append(path_data)
                        tracks_hbid.update({self.data_buffer_len:hbid})

                    pbar.set_postfix({"hbid": hbid})
                    pbar.update(1)
            tmp_info.update({f"{cfd} -> {ddd}": tracks_hbid})
            pbar.set_postfix_str(f"finish processing of id {i}")
        # self.info_json_dict.update(tmp_info)
        self.track_detail_dict.update(tmp_info)
        self.sub_db_detail.extend(tmp_path_data)
        self.sub_db_property.extend(tmp_property)

    def read_csv(self, csv_path, path_key_map=None, track_id_key=None):
        """
        :param csv_path:  要读取的csv的路径
        :param path_key_map: 读取文件本身的列名与读取保存后的列名的映射。不写则默认全部保存为文件本身的列名
        :param track_id_key: 标志csv文件中不同轨迹id的列 None则认为全体为一个轨迹
        :return:
        """
        df = pd.read_csv(csv_path, index_col=None, header=0)
        if track_id_key is None:
            self.buffer_add_track(df, path_key_map)
        else:
            for k, track in df.groupby(track_id_key):
                self.buffer_add_track(track, path_key_map)
            # for track_id in df[track_id_key].unique():
            #     self.buffer_add_track(df[df[track_id_key] == track_id], path_key_map)
        return None

    def read_csv_dir(self, dir_path, path_key_map=None, track_id_key=None):
        csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                     os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]
        for csv_file in csv_files:
            self.read_csv(csv_file, path_key_map, track_id_key)

    def read_pt(self, pt, path_key_map=None, track_type="torch"):
        tracks = torch.load(pt, map_location=torch.device('cpu'), weights_only=True)
        for track in tracks:
            self.buffer_add_track(track, path_key_map)

    def _save_sub_db(self, remove_nan=False):
        """
        只有从数据库中读取数据时候才允许使用这个方法生成一个子库，其他时候这个功能意义不大
        子库的数据没有做任何预处理 只是总数据库的一个分离
        """
        if not self.sub_db_detail:
            return None
        sub_database_path = os.path.join(self.sub_group_root, 'db.sqlite')
        sub_database_hd = DatabaseHandle(sub_database_path, frozen=False)
        choose_path_property = pd.concat(self.sub_db_property)
        choose_path_detail = pd.concat(self.sub_db_detail)
        # if remove_nan:
        #     print("清除null")
        #     print(choose_path_detail.isnull().sum())
        #     choose_path_detail.dropna(axis=0, inplace=True)
        sub_database_hd.save_table(choose_path_property, "fw_flightProperty")
        sub_database_hd.save_table(choose_path_detail, "fw_flightHJ")
        print(f"new database saved in {get_relative_path(sub_database_path)}")
        print(f"dir: {get_relative_path(os.path.abspath(sub_database_path), True)}")

        if self.flag_plot:
            num = min(1000, len(self.sub_db_detail))
            fig = plt.figure(figsize=(6, 6))
            ax = plt.subplot(111)
            for i in range(num):
                raw_path = self.sub_db_detail[i]
                ax.scatter(raw_path.iloc[:, 3], raw_path.iloc[:, 4], s=20, marker="+")
            # ax.legend(["interpolated", "raw"])
            ax.set_aspect("equal")
            self._save_fig(fig, os.path.join(self.pic_dir, "raw_chosen_track.png"))

    def _save_csv(self, processed_data):
        # 记得保存时间！
        processed_data.to_csv(os.path.join(self.sub_group_root, "all_track.csv"), index=False)
        if self.flag_div_track:
            if not os.path.exists(self.csv_dir):
                os.makedirs(self.csv_dir)
            for tk_id, track in processed_data.groupby("track_id"):
                track.to_csv(os.path.join(self.csv_dir, f"tk_{tk_id}.csv"), index=False)

    def _save_pt(self, processed_data, shuffle=True):
        track_list = []
        for _, track in processed_data.groupby("track_id"):
            # track.drop("track_id", axis=1, inplace=True)
            track_save = track.to_numpy(dtype=np.float64)
            if self.save_type == "torch":
                track_save = torch.from_numpy(track_save)
            track_list.append(track_save)
        if shuffle:
            random.shuffle(track_list)
            print("\n[!] saved pt tracks shuffled")
        torch.save(track_list, os.path.join(self.sub_group_root, "tracklist.pt"))

    def _save_info(self):
        self.info_json_dict.update({"track_detail": self.track_detail_dict})
        with open(os.path.join(self.sub_group_root, "info.json"), "w", encoding="utf8") as F:
            json.dump(self.info_json_dict, F, indent=2, ensure_ascii=False)

    def set_min_track_len(self, min_track_len):
        self.min_track_len = min_track_len
        self.info_json_dict.update({"min_track_len": min_track_len})
        return self

    def set_lim(self, key, min_=None, max_=None):
        self.data_lim_dict.update({key: (min_, max_)})
        return self

    def run(self,pt_shuffle=True, analyze_fig_size=(12, 6)):
        processed_data = self.preprocess_tracks()  # 保存前处理轨迹
        gp_name = self.__dir_generate__(f"{self.sub_dataset_name}_{self.time_seq_final}")   #生成保存目录
        print("saving on {}\n".format(gp_name))
        if self.flag_plot:
            self._plot_preprocess_track(processed_data)
        # self.processed_buffer = processed_data
        self._save_csv(processed_data)
        self._save_pt(processed_data, pt_shuffle)
        self._save_sub_db()
        self._save_info()
        print("saving done, analyzing original_data...")
        self.analyze_processed_data(self.rawdata_buffer, processed_data, analyze_fig_size)

    def run_database(self, chosen_track_ids):
        self.read_database(chosen_track_ids)
        self._save_sub_db()
        self._save_info()

    def _process_track(self, old_track: pd.DataFrame, track_id, min_track_len):
        # 去除重复项的默认方法 保留第一项
        old_track.drop_duplicates(subset=[self.time_key_name], inplace=True, keep='first')
        # 设置时间为index
        # old_track[self.time_key_name] = pd.to_datetime(old_track[self.time_key_name])
        old_track.drop("track_id", axis=1, inplace=True)
        old_track.set_index(self.time_key_name, inplace=True)
        if old_track.index.isna().all():
            # 如果出现时间全为空的情况
            print(f"ignore track id {track_id}: not have enough time original_data")
            return None
        if len(old_track.index) < self.min_raw_data_len:
            print(f"ignore track id {track_id}: too few time original_data")
            return None
        # 重采样
        # 首先按照1s采样，充分利用所有数据
        for k, (min_, max_) in self.data_lim_dict.items():
            if k in old_track.columns:
                old_track[k] = old_track[k].mask(old_track[k] > max_)
                old_track[k] = old_track[k].mask(old_track[k] < min_)
        unit_df = old_track.resample(self.time_seq_unit).asfreq().interpolate(**self.interpolate_kwargs)
        new_df = unit_df.resample(self.time_seq_final).asfreq().dropna()  # 然后按照time_seq_final采样，降低到符合要求的采样时间
        if len(new_df) < min_track_len:
            print(f"ignore track id {track_id}: too short processed original_data")
            return None
        for k, (min_, max_) in self.data_lim_dict.items():
            if k in new_df.columns:
                new_df[k] = new_df[k].clip(min_, max_)
        new_df["track_id"] = track_id
        return new_df.reset_index().copy()

    # def _process_track_manual(self, old_track: pd.DataFrame, track_id, min_track_len):
    #     pass

    def preprocess_tracks(self):
        """
        处理轨迹，保存前使用
        """
        if self.rawdata_buffer is None:
            return None
        else:
            # self.processed_buffer = pd.DataFrame(columns=[*self.data_buffer.columns])
            tmp_processed_buffer = []
            groups = self.rawdata_buffer.groupby("track_id")
            with tqdm(total=len(groups), desc=f"processing track") as pbar:
                for track_id, track in groups:
                    new_track = self._process_track(track, track_id, self.min_track_len)
                    if new_track is not None:
                        tmp_processed_buffer.append(new_track)
                    pbar.update(1)
                    # count += 1
            processed_data = pd.concat(tmp_processed_buffer, ignore_index=True)

            return processed_data

    def _plot_preprocess_track(self, processed_data):
        groups = self.rawdata_buffer.groupby("track_id")
        used_track_id = [tk_id for tk_id, _ in processed_data.groupby("track_id")][:200]
        path_raw_buffer = {tk_id: track for tk_id, track in groups}
        path_new_buffer = {tk_id: track for tk_id, track in processed_data.groupby("track_id")}
        # num = min(1000, len(path_raw_buffer))
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111)
        ax.set_aspect("equal")
        for i in used_track_id:
            raw_path = path_raw_buffer[i]
            new_path = path_new_buffer[i]
            ax.plot(new_path.loc[:, "x"], new_path.loc[:, "y"])
            ax.scatter(raw_path.loc[:, "x"], raw_path.loc[:, "y"], s=20, marker="+")
        ax.legend(["interpolated", "raw"])
        self._save_fig(fig, os.path.join(self.pic_dir, "interpolated_xy.png"))
        # plt.show()

        fig = plt.figure(figsize=(6, 6))
        ax_gd = plt.subplot(121)
        ax_gd.set_title("GD Data")

        ax_sd = plt.subplot(122)
        ax_sd.set_title("SD Data")
        for i in used_track_id:
            raw_path = path_raw_buffer[i]
            new_path = path_new_buffer[i]
            new_time = new_path.loc[:, "time"]
            raw_time = raw_path.loc[:, "time"]
            new_idx = (new_time - new_time.iloc[0]).to_numpy(dtype=np.float64) / 1e9
            raw_idx = (raw_time - raw_time.iloc[0]).to_numpy(dtype=np.float64) / 1e9
            ax_gd.plot(new_idx, new_path.loc[:, "h"])
            ax_gd.scatter(raw_idx, raw_path.loc[:, "h"], s=20, marker="+")
            ax_sd.plot(new_idx, new_path.loc[:, "v"])
            ax_sd.scatter(raw_idx, raw_path.loc[:, "v"], s=20, marker="+")

        ax_gd.set_xlabel("relative time(s)")
        ax_gd.legend(["interpolated", "raw"])
        ax_sd.set_xlabel("relative time(s)")
        ax_sd.legend(["interpolated", "raw"])

        self._save_fig(fig, os.path.join(self.pic_dir, "interpolated_height_speed.png"))

    def analyze_processed_data(self, raw_data, new_data, figsize=(6, 6)):
        """
        对处理前后的数据进行分析 获得数据的统计特性并画图
        统计特性包括：
        1. 时间跨度
        2. 点的数量
        """
        # plt.figure()
        fig, (raw_ax, new_ax) = plt.subplots(1, 2, sharey=True, figsize=figsize)

        def analyze_pic_time(ax, data):
            for tk_id, track in data.groupby("track_id"):
                if pd.isna(track.loc[:, "time"].iloc[0]):
                    continue
                delta_time = (track.loc[:, "time"] - track.loc[:, "time"].iloc[0]).to_numpy(dtype=np.float64) / 1e9
                ax.scatter(delta_time, [tk_id] * len(track), s=0.1)
                ax.plot(delta_time, [tk_id] * len(track), linewidth=0.08)
            ax.set_xlabel("time(s)")

        raw_ax.set_ylabel("track id")
        analyze_pic_time(raw_ax, raw_data)
        analyze_pic_time(new_ax, new_data)
        self._save_fig(fig, os.path.join(self.pic_dir, "analyze_data_time.png"))
        plt.close(fig)

        fig, (raw_ax, new_ax) = plt.subplots(1, 2, sharey=True, figsize=figsize)

        def analyze_pic_relative(ax, data):
            for tk_id, track in data.groupby("track_id"):
                time_serial = track.loc[:, "time"]
                if pd.isna(time_serial.iloc[0]):
                    continue
                relative_time = (time_serial - time_serial.iloc[0]) / (time_serial.iloc[-1] - time_serial.iloc[0])
                delta_time = relative_time.to_numpy(dtype=np.float64)
                ax.scatter(delta_time, [tk_id] * len(track), s=0.1)
                ax.plot(delta_time, [tk_id] * len(track), linewidth=0.08)
            ax.set_xlabel("Flight completion progress")

        raw_ax.set_ylabel("track id")
        analyze_pic_relative(raw_ax, raw_data)
        analyze_pic_relative(new_ax, new_data)
        self._save_fig(fig, os.path.join(self.pic_dir, "analyze_data_percent.png"))

        plt.close(fig)

        used_raw_data = raw_data[raw_data["track_id"].isin(pd.unique(new_data["track_id"]))]
        hist_data_time = ((used_raw_data.groupby("track_id").last() - used_raw_data.groupby("track_id").first())["time"]
                          .replace(0, pd.NA).dropna().to_numpy(dtype=np.float64) / 1e9)
        hist_data_point = used_raw_data.groupby("track_id").count()["time"].replace(0, pd.NA).dropna().to_list()

        fig, (raw_ax, new_ax) = plt.subplots(1, 2, sharey=False, figsize=figsize)

        def analyze_pic_hist(ax, data):
            # hist_data = original_data.groupby("track_id").count()["time"].replace(0,pd.NA).dropna().to_list()
            ax.hist(data, bins='rice', density=False, rwidth=0.95)

        raw_ax.set_ylabel("hist")
        new_ax.set_ylabel("hist")
        raw_ax.set_xlabel("s")
        new_ax.set_xlabel("points")
        analyze_pic_hist(raw_ax, hist_data_time)
        analyze_pic_hist(new_ax, hist_data_point)
        self._save_fig(fig, os.path.join(self.pic_dir, "analyze_data_hist.png"))

        return 0


