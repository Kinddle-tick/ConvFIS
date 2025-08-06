#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :willow_graph.py
# @Time      :2024/1/2 20:58
# @Author    :Oliver

"""
绘制整体轨迹，然后取轨迹上随机（或者一定间隔）的点，绘制从这个点开始预测的结果，最后的结果类似柳枝和柳枝上的叶子
结果中的分叉越少则效果越好
一张图中可以涉及到的轨迹可以有很多条

"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from ...reporter import Reporter
from ...painter_format import PlotTemplate


# def _generate_color_palette(n_colors):
#     # 使用numpy生成均匀分布的颜色
#     colormap = plt.get_cmap("Accent")
#     colors = colormap(np.linspace(0, 1, n_colors))
#     return colors


def _willow_branch(ax, raw_track, color="b"):
    raw_track_x = raw_track[..., 0]
    raw_track_y = raw_track[..., 1]

    return ax.plot(raw_track_x, raw_track_y, '-', color=color, linewidth=1.2, alpha=0.9,
                   label="real track", zorder=0)


def _willow_leaf(ax, willow_track, color="r", prefix=""):
    willow_track_x = willow_track[..., 0].T
    willow_track_y = willow_track[..., 1].T
    return ax.plot(willow_track_x, willow_track_y, '-', color=color, linewidth=0.5, alpha=0.6,
                   label=f"predict track {prefix}", zorder=1)


def _single_willow_graph(ax, raw_track, willow_track, color_leaf="r", color_branch="b"):
    rtn_branch = _willow_branch(ax, raw_track, color_branch)
    rtn_willow = _willow_leaf(ax, willow_track, color_leaf)
    return rtn_branch, rtn_willow


def willow_graph_self(raw_track, willow_track, sample_rule: list or slice or int,
                      used_tracks: list or int or None,
                      reporter: Reporter,plot_temp:PlotTemplate, prefix="dft"):
    # plotter_dict = {} if plotter_dict is None else plotter_dict
    # plotter = Plotter(**plotter_dict)
    if isinstance(sample_rule, int):
        sample_rule = slice(0, None, sample_rule)   #int输入是等间隔采样

    if isinstance(used_tracks, int):
        used_tracks = slice(None, used_tracks)
    elif used_tracks is None:
        used_tracks = slice(None, None) # 使用全部的估计
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)
    # fig, ax = plotter.gene_figure()
    ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_aspect("equal")
    with tqdm(total=len(np.arange(len(raw_track))[used_tracks]), leave=False) as pbar:
        for idx in pd.Series(willow_track.keys())[used_tracks]:
            if idx in willow_track.keys():
                track = raw_track[idx]
                willow = willow_track[idx][sample_rule]
                _willow_branch(ax, track)
                _willow_leaf(ax, willow)
            else:
                reporter(f"[x]draw willow id {idx} failed cause there is no data in willow_track", flag_log=True)
            pbar.update(1)
            pbar.set_postfix_str(f"drawing track id {idx}")
            # _single_willow_graph(ax, track, willow)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1], handles[0]], [labels[-1], labels[0]], fontsize= plot_temp.params["fontsize.legend"])
    # ax.legend([handles[-1], handles[0]], [labels[-1], labels[0]], fontsize= plot_temp.params["fontsize.legend"])
    rtn = reporter.save_figure_to_file(fig, f"willow_graph_{prefix}")
    # plt.show()
    plt.close(fig)
    # plotter.close(fig)
    return rtn


def willow_graph_cross(raw_track, willow_track, sample_rule: slice or int, used_tracks: list or slice,reporter:Reporter,
                       plot_temp:PlotTemplate,
                       *, x_lim=None, y_lim=None,):
    # plotter_dict = {} if plotter_dict is None else plotter_dict
    #
    # plotter = Plotter(**plotter_dict)
    if isinstance(sample_rule, int):
        sample_rule = slice(0, None, sample_rule)
    if isinstance(used_tracks, int):
        used_tracks = slice(None, used_tracks)
    elif used_tracks is None:
        used_tracks = slice(None, None) # 使用全部的估计

    if len(willow_track.keys()) <= 5:
        color_palette = ["black", "blue", "red", "green", "orange", "gray",]
        # color_palette = _generate_color_palette(len(willow_track.keys()) + 1)
    else:
        # color_palette = plot_temp.generate_color_palette(len(willow_track.keys()) + 1)
        color_palette = plot_temp.color_palette(len(willow_track.keys()) + 1)

    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)
    # fig,ax = plotter.gene_figure()
    ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
    # ax.set_xlabel("Longitude(°)", fontsize=plt.rcParams["legend.fontsize"])
    # ax.set_ylabel("Latitude(°)", fontsize=plt.rcParams["legend.fontsize"])

    ax.set_aspect("equal")
    flag_first = True
    legend_handle = []
    track_id = list(raw_track.keys())
    # for used_track_id in tqdm.trange(used_tracks) if isinstance(used_tracks, int) else used_tracks:
    with tqdm(total=len(track_id[used_tracks]), leave=False) as pbar:
        for used_track_id in track_id[used_tracks]:
            branch_axes = _willow_branch(ax, raw_track[used_track_id], color_palette[0])[0]
            if flag_first:
                legend_handle.append(branch_axes)
            for idx, (prefix, willow_data) in enumerate(willow_track.items()):
                if used_track_id in willow_data:
                    leaf_axes = _willow_leaf(ax, willow_data[used_track_id][sample_rule], color_palette[idx+1], prefix)
                    if flag_first:
                        legend_handle.append(leaf_axes[0])
                else:
                    reporter(f"draw willow id {idx} failed cause there is no data in willow_track", flag_log=True)

            pbar.update(1)
            pbar.set_postfix_str(f"drawing track id {used_track_id}")
            flag_first = False
    ax.legend(legend_handle, [handle.get_label() for handle in legend_handle], fontsize= plot_temp.params["fontsize.legend"])
    # ax.legend(legend_handle, [handle.get_label() for handle in legend_handle],
    #           fontsize=plt.rcParams["legend.fontsize"])
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    plt.close(fig)
    return fig, ax
    # rtn = reporter.save_figure_to_file(fig, f"willow_graph_cross")
    # # plt.show()
    # plt.close(fig)
    # return rtn
