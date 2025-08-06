#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/12 12:39
# @Author  : Oliver
# @File    : comparisons_func.py
# @Software: PyCharm

from enum import Enum
from typing import Dict

import torch
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from ...painter_format import PlotTemplate
from ...reporter import Reporter
"""
对于输入（行为模型，列为指标）
"""

def calculate_offset(sizes, ax, fig):
    # 计算半径（点单位）
    radii = np.sqrt(sizes) / 2  # 近似为圆形半径

    # 获取坐标轴转换参数
    # fig = ax.figure
    ymin, ymax = ax.get_ylim()
    ax_height_data = ymax - ymin
    ax_bbox_pixels = ax.get_window_extent().height
    pixels_per_data = ax_bbox_pixels / ax_height_data

    # 将半径从点转换为数据单位
    dpi = fig.dpi
    radius_data = radii * (1 / 72 * dpi) / pixels_per_data
    return radius_data

def compare_bar(charactor_series:pd.Series, palette,reporter:Reporter, plot_temp:PlotTemplate, main_model_name,
                y_alias=None, y_log=False, x_label=None, xtick_rotation=45,
                draw_main_horizon_line=True,
                with_text=True,  with_grid=False, text_round=1, ):

    # 创建图形和坐标轴
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)

    # 设置坐标轴对数刻度
    if y_log:
        ax.set_yscale('log')

    column = charactor_series.name
    fig_save_name = f"compare_bar_{column}"
    x = np.arange(len(charactor_series))
    bars = ax.bar(x, charactor_series, color=palette)

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(charactor_series.index, rotation=xtick_rotation, ha='right')
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel(y_alias if y_alias is not None else column, fontsize=plot_temp.params["fontsize.label"])

    # Add horizontal line for main model
    if draw_main_horizon_line and main_model_name in charactor_series.index:
        ax.axhline(charactor_series[main_model_name], color='black', linestyle='--', linewidth=0.5)

    # Add value labels
    if with_text:
        for bar in bars:
            height = bar.get_height()
            if text_round <= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(round(height,text_round))}',
                        ha='center', va='bottom', fontsize=plot_temp.params["fontsize.bar_top"])
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{round(height,text_round)}',
                        ha='center', va='bottom', fontsize=plot_temp.params["fontsize.bar_top"])
            # if text_int:
            #     ax.text(bar.get_x() + bar.get_width() / 2, height,
            #             f'{round(height,0)}', ha='center', va='bottom')
            # else:
            #     ax.text(bar.get_x() + bar.get_width() / 2, height,
            #             f'{round(height,1)}', ha='center', va='bottom')
    if with_grid:
        ax.grid(
            axis='y',  # 仅显示横线
            color='gray',  # 网格颜色
            # linestyle='--',  # 虚线样式
            linewidth=0.5,  # 线宽
            alpha=0.4  # 透明度
        )

    # 抬高画幅上沿，避免text被遮挡
    if with_text:
        if y_log:
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(bottom=y_min / 10, top=y_max * 10)  # 强制扩展5%
        else:
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(top=y_min + (y_max - y_min) * 1.05)  # 强制扩展5%

    fig_save_name = fig_save_name.replace(" ", "_")
    fig_save_name = fig_save_name.replace("/", "-")
    # Save and close
    plt.tight_layout()
    path = reporter.save_figure_to_file(fig, save_name=fig_save_name)
    plt.close(fig)

    return path

def compare_scatter(charactor_df:pd.DataFrame,palette, reporter:Reporter, plot_temp:PlotTemplate, main_model_name,
                    x_alias=None,y_alias=None, x_log=False, y_log=False, z_log=False,
                    draw_main_model_horizon_line = True, draw_main_model_vertical_line=True,
                    with_text=True, with_legend=True, with_grid=False, text_round=1,
                    size_map=None):
    model_names = list(charactor_df.index)
    columns_name = list(charactor_df.columns)

    # 创建图形和坐标轴
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)

    # 设置坐标轴对数刻度
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    # 绘制2d的图片 默认情况下，前两个维度分别对应x轴和y轴，第三个维度是圆点大小的输入
    if size_map is None:
        size_map = lambda arr: plot_temp.adopt_scatter_size(arr, log=z_log)

    if len(columns_name) == 2:
        sizes = pd.Series([plot_temp["scatter.size.default"]] * len(model_names), index=charactor_df.index, dtype="float")
    else:
        sizes = pd.Series(list(size_map(charactor_df.iloc[:, 2])), index=charactor_df.index, dtype="float")

    xs = charactor_df.iloc[:, 0]
    ys = charactor_df.iloc[:, 1]
    # sizes = pd.DataFrame(sizes, index=charactor_df.index, dtype="float")
    tmp_name = "-".join(columns_name)
    fig_save_name = f"compare_2d_[{tmp_name}]"
    for i, model in enumerate(model_names):
        ax.scatter(xs[model], ys[model], s=sizes[model], color=palette[model], label=model, alpha=0.8)

    # 绘制参考线
    if main_model_name in model_names:
        if draw_main_model_horizon_line:
            ax.axhline(ys[main_model_name], color='gray', linestyle='--', linewidth=0.5)
        if draw_main_model_vertical_line:
            ax.axvline(xs[main_model_name], color='gray', linestyle='--', linewidth=0.5)

    # 标记Obj名称
    if with_text:
        for i, model in enumerate(model_names):
            color = palette[model]
            x, y = xs[model], ys[model]
            offset = calculate_offset(sizes[model], ax, fig)
            ax.text(x, y + offset, model, color=color, ha='center', va='bottom',fontsize=plot_temp.params["scatter.text.fontsize"])
    # 添加图例
    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), fontsize= plot_temp.params["fontsize.legend"])
    # 添加网格
    if with_grid:
        ax.grid(
            color='gray',  # 网格颜色
            # linestyle='--',  # 虚线样式
            linewidth=0.5,  # 线宽
            alpha=0.4  # 透明度
        )
    # 设置轴标签和标题
    ax.set_xlabel(columns_name[0] if x_alias is None else x_alias, fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel(columns_name[1] if y_alias is None else y_alias, fontsize=plot_temp.params["fontsize.label"])

    # 抬高画幅上沿，避免text被遮挡
    if with_text:
        if y_log:
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(bottom=y_min / 10, top=y_max * 10)  # 强制扩展5%
        else:
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(top=y_min + (y_max - y_min) * 1.05)  # 强制扩展5%

    fig_save_name = fig_save_name.replace(" ", "_")
    fig_save_name = fig_save_name.replace("/", "-")
    # Save and close
    plt.tight_layout()
    path = reporter.save_figure_to_file(fig, save_name=fig_save_name)
    plt.close(fig)
    return path

def compare_df(charactor_df:pd.DataFrame, reporter:Reporter, plot_temp:PlotTemplate, main_model_name, bar_mode=False,
               x_alias=None, y_alias=None, z_alias=None, x_log=False, y_log=False, z_log=False, xtick_rotation=45,
               main_hline = True, main_vline=True, with_text=True, with_legend=True, with_grid=True,
               text_round=1, size_map=None):
    rtn_fig_paths = []
    model_names = list(charactor_df.index)
    if isinstance(charactor_df, pd.Series):
        columns_name = [charactor_df.name]
        charactor_df = pd.DataFrame(charactor_df)
    else:
        columns_name = list(charactor_df.columns)


    # 调色盘 以及将主模组提取出来 (主模组放在最后一个位置，调色盘的第一个颜色是特殊颜色，放在对应）
    palette = plot_temp.color_palette(len(model_names), None if main_model_name not in model_names else model_names.index(main_model_name))
    palette = pd.Series(palette, index=charactor_df.index)


    # 如果只有一列，那只能画一维直方图
    if len(columns_name)==1:
        bar_mode = True

    # 直方图对比时，x轴不会取对数 也不会有垂直标记
    if bar_mode:
        x_log = False
        main_vline = False

    # main_plot:
    if bar_mode:
        if len(columns_name)>1:
            for column in columns_name:
                path = compare_df(charactor_df[column], reporter, plot_temp, main_model_name, bar_mode=True, x_log=False, y_log=y_log,
                                  main_hline=main_hline, main_vline=False,
                                  with_text=with_text, with_legend=with_legend, with_grid=with_grid, text_round=text_round, )
                rtn_fig_paths.extend(path)
        else:
            path = compare_bar(charactor_df[columns_name[0]], palette, reporter, plot_temp, main_model_name,
                               y_alias=None, y_log=y_log, x_label=x_alias, xtick_rotation=xtick_rotation,
                               draw_main_horizon_line=main_hline,
                               with_text=with_text, with_grid=with_grid, text_round=text_round)
            rtn_fig_paths.append(path)

    else:
        path = compare_scatter(charactor_df, palette, reporter, plot_temp, main_model_name,
                               x_alias, y_alias, x_log, y_log, z_log,
                               main_hline, main_vline, with_text, with_legend, with_grid, text_round, size_map)
        rtn_fig_paths.append(path)
        if len(columns_name) >= 3:
            reporter(f"[+] add explain of size {columns_name[2]}")
            path = compare_bar(charactor_df.iloc[:, 2], palette, reporter, plot_temp, main_model_name,
                               y_alias=z_alias, y_log=z_log, x_label=x_alias, xtick_rotation=xtick_rotation,
                               draw_main_horizon_line=main_hline,
                               with_text=with_text, with_grid=with_grid, text_round=text_round)
            rtn_fig_paths.append(path)

    return rtn_fig_paths