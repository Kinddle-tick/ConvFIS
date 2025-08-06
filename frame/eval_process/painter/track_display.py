#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :track_display.py
# @Time      :2024/1/29 14:22
# @Author    :Oliver
import os.path

import torch
import pandas as pd
from matplotlib import pyplot as plt
from ...reporter import Reporter
from ...painter_format import PlotTemplate
import numpy as np
"""
增加一个绘制轨迹编号参考的功能。
"""




def track_display(track_np, report: Reporter, plot_temp:PlotTemplate, note_text="", flag_ref=True):
    # plotter_dict = {} if plotter_dict is None else plotter_dict

    # plotter = Plotter(**plotter_dict)
    # track_np = [track.cpu().detach().numpy() for track in tracks_list]

    fig = plt.figure(**plot_temp.temp_fig())
    # ax = fig.add_subplot(211)
    ax = fig.add_subplot(111)
    # fig, ax = plotter.gene_half_figure()
    ax.set_title("Track")
    ax.set_aspect("equal")
    for track in track_np:
        ax.plot(track[..., 0], track[..., 1], label="", linewidth=0.3)
    ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
    # ax.set_xlabel("Longitude(°)", fontsize=plt.rcParams["legend.fontsize"])
    # ax.set_ylabel("Latitude(°)", fontsize=plt.rcParams["legend.fontsize"])
    pic_track_path = report.save_figure_to_file(fig, f"{note_text}_track")
    plt.close(fig)
    # plt.legend()

    # fig = plt.figure(dpi=128, figsize=(8, 8))
    # ax = fig.add_subplot(111)
    fig = plt.figure(**plot_temp.temp_fig())
    # ax = fig.add_subplot(211)
    ax = fig.add_subplot(111)
    ax.set_title("Height")
    for track in track_np:
        ax.plot(track[..., 2], label="")
    # ax.legend()
    pic_height_path = report.save_figure_to_file(fig, f"{note_text}_height")
    plt.close(fig)

    # fig = plt.figure(dpi=128, figsize=(8, 8))
    # ax = fig.add_subplot(111)
    fig = plt.figure(**plot_temp.temp_fig())
    # ax = fig.add_subplot(211)
    ax = fig.add_subplot(111)
    ax.set_title("Speed")
    for track in track_np:
        ax.plot(track[..., 3], label="")
    # ax.legend()
    pic_speed_path = report.save_figure_to_file(fig, f"{note_text}_speed")
    plt.close(fig)

    # plt.close(fig)

    if flag_ref:
        fig_ref = plt.figure(**plot_temp.temp_fig())
        ax_ref = fig_ref.add_subplot(111)
        ax_ref.set_aspect("equal")
        for track in track_np:
            ax_ref.plot(track[..., 0], track[..., 1], color="green", alpha=0.1, label="")
        for idx, track in enumerate(track_np):
            report(f"\r {idx+1} / {len(track_np)}...")
            tmp, = ax_ref.plot(track[..., 0], track[..., 1],color="red", label="")
            report.save_figure_to_file(fig_ref, os.path.join("split_track", f"{note_text}_{idx}"))
            tmp.remove()
        plt.close(fig_ref)

    # del plotter
    return pic_track_path, pic_height_path, pic_speed_path

def track_display_heatmap(track_np, report: Reporter, plot_temp:PlotTemplate, note_text="", bins=100,*, flag_binary=True,
                          x_dim=0, y_dim=1):
    track_np = np.concatenate(track_np)
    x = track_np[...,x_dim].reshape(-1)
    y = track_np[...,y_dim].reshape(-1)
    H, yedges, xedges = np.histogram2d(y, x, bins=bins)
    H = 10 - np.clip(H, 0, 10)
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)
    ax.pcolormesh(xedges, yedges, H, cmap='Greys')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.grid()
    pic_track_path = report.save_figure_to_file(fig, f"{note_text}_track_heatmap")
    plt.close(fig)
    return pic_track_path

