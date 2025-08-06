#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/16 18:10
# @Author  : oliver
# @File    : data_analyze.py
# @Software: PyCharm
import datasets
import pandas as pd
import random

from data_source.Interface.DatabaseHandle import FlightPathDataHandle, FlightPropertyDataHandle, DatabaseHandle
from tqdm.auto import tqdm
import os
import torch
from ..basic_protocal import Data
from matplotlib import pyplot as plt
import numpy as np
from ..painter_format import plot_template

class DataAnalyze:
    def __init__(self, to_path):
        self.save_formats = ["svg","png"]
        self.save_dir = os.path.join(to_path, "pic")
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_fig(self,fig, filename):
        formats = self.save_formats
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
            formats = [ext]
            # if ext not in formats:
            #     formats.append(ext)
        else:
            name = filename

        path = os.path.join(self.save_dir, name)
        for fmt in formats:
            try:
                fig.savefig(path+"."+fmt, format=fmt,bbox_inches='tight')
            except FileNotFoundError as e:
                print(f"[!-] cannot find the path to save the pic, try to save original_data first. \n{e}")
        return path

    def analyze(self, *args, **kwargs):
        pass

class TrackCompareAnalyze(DataAnalyze):

    def basic_data_statistics(self,data:Data):
        # track_data = data.raw_data
        track_dict = {tk_id: track for tk_id, track in enumerate(data)}
        data_track_num = len(track_dict.keys())
        points = len(data.get_total_data())
        print("[+] using {} tracks, including {} points".format(data_track_num,points))
        pass

    def draw_2d(self,data:Data, prefix="xy",  max_show_number=200, x_col="x", y_col="y",
                grid=False, equal=False, align_x=False,x_label="longitude", y_label="latitude" ):
        used_track_id = [i for i in range(len(data))]
        random.shuffle(used_track_id)
        used_track_id = used_track_id[:max_show_number]

        used_track = {tk_id: data[tk_id] for tk_id in used_track_id}
        fig = plt.figure(**plot_template.temp_fig())
        ax = fig.add_subplot(111)

        if equal:
            ax.set_aspect("equal")

        for i, track in used_track.items():
            y = track.loc[:, y_col]
            x = track.loc[:, x_col]
            if align_x:
                x = (x - x.iloc[0]).to_numpy(dtype=np.float64) / 1e9
            ax.plot(x, y, linewidth=0.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(grid)

        self._save_fig(fig, "overview_"+prefix)
        plt.close(fig)
        return

    def draw_2d_heat(self,data:Data, prefix="xy",x_col="x", y_col="y",
                     bins=100, reverse=True,equal=True, max_clip=10, grid=False):
        total_data = data.get_total_data()
        y = total_data.loc[:,y_col]
        x = total_data.loc[:,x_col]
        H, yedges, xedges = np.histogram2d(y, x, bins=bins)
        if reverse:
            H = max_clip - np.clip(H, 0, max_clip)
        else:
            H = np.clip(H, 0, max_clip)

        fig = plt.figure(**plot_template.temp_fig())
        ax = fig.add_subplot(111)
        if equal:
            ax.set_aspect("equal")
        ax.pcolormesh(xedges, yedges, H, cmap='Greys')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel("Longitude(°)")
        ax.set_ylabel("Latitude(°)")
        # ax.set_xlabel("Longitude(°)", fontsize=plt.rcParams["legend.fontsize"])
        # ax.set_ylabel("Latitude(°)", fontsize=plt.rcParams["legend.fontsize"])
        ax.grid(grid)
        self._save_fig(fig, "heatmap_"+prefix)
        plt.close(fig)
        return

    def script_draw_data(self,data, prefix="analyze", max_show_num=256, heat_bin=256):
        self.draw_2d(data, f"{prefix}_xy", max_show_num, "x", "y", grid=True, equal=True)
        self.draw_2d(data, f"{prefix}_height", max_show_num, "time", "h", align_x=True, x_label="time", y_label="height")
        self.draw_2d(data, f"{prefix}_speed", max_show_num, "time", "v", align_x=True, x_label="time", y_label="speed")
        self.draw_2d_heat(data, f"{prefix}_xy", "x", "y", heat_bin)
        return

    def analyze(self, old_data:Data, new_data:Data, max_show_number=200):
        used_track_id = [i for i in range(len(new_data))]
        random.shuffle(used_track_id)
        used_track_id = used_track_id[:max_show_number]
        path_raw_buffer = {tk_id: old_data[tk_id] for tk_id in used_track_id}
        path_new_buffer = {tk_id: new_data[tk_id] for tk_id in used_track_id}
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        ax.set_aspect("equal")
        for i in used_track_id:
            raw_path = path_raw_buffer[i]
            new_path = path_new_buffer[i]
            ax.plot(new_path.loc[:, "x"], new_path.loc[:, "y"])
            ax.scatter(raw_path.loc[:, "x"], raw_path.loc[:, "y"], s=20, marker="+")
        ax.legend(["interpolated", "raw"])
        plt.grid(True)
        self._save_fig(fig, "interpolated_xy.png")
        # plt.show()

        fig_gd = plt.figure(figsize=(6, 6))
        ax_gd = plt.subplot(111)
        ax_gd.set_title("GD Data")

        fig_sd = plt.figure(figsize=(6, 6))
        ax_sd = plt.subplot(111)
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

        self._save_fig(fig_gd, "interpolated_height.png")
        self._save_fig(fig_sd, "interpolated_speed.png")
        return


