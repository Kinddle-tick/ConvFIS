#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 21:49
# @Author  : Oliver
# @File    : interpretable.py
# @Software: PyCharm
from collections import OrderedDict
from sys import stdout

import pandas as pd
from jupyter_server.serverapp import flags
from matplotlib.collections import LineCollection

from ...painter_format import PlotTemplate, plot_template
import numpy as np
import os

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from ...reporter import Reporter

def rule_frequency(categorical_credentials, reporter:Reporter, plot_temp:PlotTemplate,
                      info:str, rule_mode, xtick_show_id_interval=1, sort=True, log_y=False):

    bar_data = pd.Series(np.mean(categorical_credentials, axis=0))
    palette = pd.Series(plot_temp.color_palette(len(bar_data) + 20)[20:])
    x = np.arange(len(bar_data))
    info = info + "_" + rule_mode

    if sort:
        draw_bar_data = bar_data.sort_values(ascending=False)
        x_ticks = [idx+1 for idx in bar_data.index]
        color = palette.reindex(draw_bar_data.index)
    else:
        draw_bar_data = bar_data
        x_ticks = [idx+1 for idx in bar_data.index]
        color = palette

    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)
    if log_y:
        plt.yscale('log')
    ax.bar(x, draw_bar_data, color=color)
    if xtick_show_id_interval != 0: #避免标签过于密集
        ax.set_xticks(range(0, len(bar_data), xtick_show_id_interval))
        ax.set_xticklabels(x_ticks[::xtick_show_id_interval])
    else:
        ax.set_xticklabels(x_ticks)

    if sort:
        ax.set_xticks([])
        # ax.set_xlabel("Sorted rule", fontsize=plot_temp.params["fontsize.label"])
    else:
        ax.set_xlabel("Rule id", fontsize=plot_temp.params["fontsize.label"])
    # ax.set_xlabel("rule id", fontsize=plot_temp.params["fontsize.label"])
    if rule_mode == "rw":
        ax.set_ylabel("Average rule weight", fontsize=plot_temp.params["fontsize.label"])
    elif rule_mode == "fl":
        ax.set_ylabel("Average firing level", fontsize=plot_temp.params["fontsize.label"])
    else:
        ax.set_ylabel("Average firing level", fontsize=plot_temp.params["fontsize.label"])

    name_add =  ("sort" if sort else "") +("_" if sort and log_y else "")+ ("logy" if log_y else "")
    rtn = reporter.add_figure_by_data(fig, f"interpretable_frequency_{info}_{name_add}",
                                title=f"frequency of rules in {info}")
    plt.close(fig)
    return rtn

def show_defuzzifier_2d(data, level, reporter:Reporter, plot_temp:PlotTemplate, file_info,
                        x_dim=0, y_dim=1, x_label="x", y_label="y", xy_limit=None, frame=None,
                        level_limit=None):
    """
    class_limit: 给出列表时，只会选择指定的几个level绘制对应轨迹。给出数字时，选择level最大的若干个绘制轨迹
    """
    palette = pd.Series(plot_temp.color_palette(len(level)))
    # num_level = len(level)
    if isinstance(level_limit, list):
        level_filter = level_limit
    else:
        level_filter = np.argsort(-level)[:level_limit]



    xy = data[level_filter][..., [x_dim, y_dim]].copy()
    color = palette[level_filter]

    # Setup main figure
    fig, ax = plot_temp.get_fig()
    ax.set_aspect("equal")
    ax.set_xlabel(x_label, fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel(y_label, fontsize=plot_temp.params["fontsize.label"])
    line_collection = LineCollection(xy[::-1], linewidths=0.6, colors=color[::-1])    #::-1使得第一条轨迹显示在最上层
    ax.add_collection(line_collection)

    if xy_limit is None:
        ax.margins(0.025)
    else:
        if isinstance(xy_limit, int):
            x_max = y_max = xy_limit
            x_min = y_min = -xy_limit
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        elif isinstance(xy_limit, tuple or list):
            x_min, y_min, x_max, y_max = xy_limit
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.margins(0.025)
    if frame is not None:
        plot_temp.add_frame_counter(fig,text=f"step={frame}")
    reporter.add_figure_by_data(fig, f"explain_2d_defuzzifier_len({len(level_filter)})_{file_info}", f"Defuzzifier of {file_info}")
    plt.close(fig)

def show_samples_2d(data, level, reporter:Reporter, plot_temp:PlotTemplate, file_info,
                    x_dim=0, y_dim=1, x_label="x", y_label="y", frame=None,
                    level_limit=None):
    palette = pd.Series(plot_temp.color_palette(len(level)))
    level_sort = np.argsort(-level.mean(axis=0)).tolist() # 大的在前面
    sample_category = np.argmax(level, axis=1)
    # num_level = len(level)
    if isinstance(level_limit, list):
        used_level = sorted(level_limit, key=lambda x:level_sort.index(x))
    else:
        used_level = level_sort[:level_limit]
        # level_filter = np.argsort(-level)[:level_limit]


    xy = data[..., [x_dim, y_dim]]
    # color = palette[level_filter]

    # Setup main figure
    fig, ax = plot_temp.get_fig()
    ax.set_aspect("equal")
    ax.set_xlabel(x_label, fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel(y_label, fontsize=plot_temp.params["fontsize.label"])
    used_sample = {}
    legend_handles = []  # 存储图例句柄
    # for level_id in reversed(used_level):
    for level_id in used_level:
        samples = xy[sample_category == level_id]
        if len(samples) == 0:
            continue
        else:
            used_sample[level_id] = len(samples)
        color = palette[level_id]

        line_collection = LineCollection(samples, linewidths=0.6,alpha=0.5, colors=color, label=level_id)
        handle = plt.Line2D([], [],
                        color=color,
                        linewidth=1.2,  # 图例线条更粗
                        label=f'{level_id}',
                        alpha=1)
        legend_handles.append(handle)

        ax.add_collection(line_collection)
    ax.margins(0.025)
    ax.legend(
        handles=legend_handles,
        title="Rule id",
        title_fontsize=plot_temp.params["fontsize.legend.title"],
        fontsize=plot_temp.params["fontsize.legend"],
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    plot_temp.add_frame_counter(fig,text=f"step={frame}")
    reporter.add_figure_by_data(fig, f"explain_2d_samples_{file_info}", f"Samples of {file_info}")
    plt.close(fig)
    return used_sample


def generic_interpretable_plot(
        data_to_show,
        categorical_credentials,
        reporter: Reporter,
        plot_temp: PlotTemplate,
        info: str,
        mode: str = 'alpha',  # 'alpha', 'max', 'divided'
        show_category: int = None,
        show_data_limit: int = 50,
        draw_sub_fig: bool = True,
        show_legend: bool = None,
        ranking_legend: bool = True,
        x_dim: int = 0,
        y_dim: int = 1,
        flag_same_limit: bool = True,
        **kwargs
):
    # Determine categorical count based on mode
    if mode == 'divided':
        categorical_count = len(data_to_show)
        # print(f"test:{categorical_credentials.shape[-2]} == {categorical_count}")
    else:
        categorical_count = categorical_credentials.shape[1]

    # Set default values
    show_category = categorical_count if show_category is None else min(show_category, categorical_count)
    show_legend = show_category <= 16 if show_legend is None else show_legend

    # Prepare data coordinates
    datas_x = data_to_show[..., x_dim]
    datas_y = data_to_show[..., y_dim]

    # Calculate axis limits
    x_lim_max, x_lim_min = datas_x.max(), datas_x.min()
    y_lim_max, y_lim_min = datas_y.max(), datas_y.min()

    # Create color palette
    palette = pd.Series(plot_temp.color_palette(categorical_count))
    # palette = pd.Series(plot_temp.color_palette(categorical_count + 20)[20:])

    # Setup main figure
    main_fig = plt.figure(**plot_temp.temp_fig(width_scale=1))
    main_ax = main_fig.add_subplot(111)
    main_ax.set_aspect("equal")
    main_ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    main_ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])

    # Set axis limits based on mode
    if mode != 'divided':
        main_ax.set_xlim(x_lim_min, x_lim_max)
        main_ax.set_ylim(y_lim_min, y_lim_max)

    main_ax_legend_axes = []

    # Prepare data based on mode
    if mode == 'max':
        favorite_rule_dict = {i : np.where(categorical_credentials.argmax(axis=1) == i)[0] for i in range(categorical_count)}
        total_dict = {i:ids[np.argsort(categorical_credentials[ids,i])[::-1]] for i, ids in favorite_rule_dict.items() if len(ids)>0}
        ordered_category_id = np.argsort(np.sum(categorical_credentials, axis=0))[::-1]
        total_iterations = min(show_category, len(ordered_category_id))
    else:
        total_dict = {i:np.argsort(categorical_credentials[...,i])[::1] for i in range(categorical_count)}
        ordered_category_id = np.argsort(np.sum(categorical_credentials, axis=0))[::-1]
        total_iterations = show_category

    # Processing loop
    with tqdm(total=total_iterations, desc=f"Interpretable {mode} [{info}]", file=stdout, position=0) as pbar:
        for ranking_id in range(total_iterations):
            if mode == 'max':
                internal_id = int(ordered_category_id[ranking_id])
            else:
                internal_id = int(ordered_category_id[ranking_id])

            # Get data based on mode
            if mode == 'alpha':
                # draw_id = np.argsort(categorical_credentials[:, internal_id])[::-1][:show_data_limit]
                draw_id = total_dict[internal_id][:show_data_limit]
                plot_x = datas_x[draw_id]
                plot_y = datas_y[draw_id]
                alpha = np.clip(categorical_credentials[draw_id, internal_id] * 4, 0.4, 0.9)
            elif mode == 'max':
                if internal_id in total_dict:
                    draw_id = total_dict[internal_id][:show_data_limit]
                    plot_x = datas_x[draw_id]
                    plot_y = datas_y[draw_id]
                    alpha = 0.8
                else:
                    pbar.update(1)
                    continue
            elif mode == 'divided':
                plot_x = datas_x[internal_id]
                plot_y = datas_y[internal_id]
                alpha = 1

            label = f"rank_{ranking_id+1}" if ranking_legend else f"{internal_id}"

            # Plot main figure
            if mode == 'divided':
                main_ax.plot(plot_x.T, plot_y.T, color=palette[internal_id], linewidth=1, alpha=alpha,
                             zorder=-ranking_id if ranking_legend else -internal_id)
            else:
                segments = [np.column_stack([x, y]) for x, y in zip(plot_x, plot_y)]
                lc = LineCollection(segments, colors=[palette[internal_id]] * len(segments), alpha=alpha,
                                    linewidth=0.6, zorder=-ranking_id if ranking_legend else -internal_id)
                main_ax.add_collection(lc)

            proxy_line = plt.Line2D([], [], color=palette[internal_id], linewidth=1, label=label)
            main_ax_legend_axes.append((proxy_line, ranking_id if ranking_legend else internal_id))

            # Create sub-figures
            if draw_sub_fig:
                fig = plt.figure(**plot_temp.temp_fig())
                ax = fig.add_subplot(111)
                ax.set_aspect("equal")
                if mode == 'divided':
                    ax.plot(plot_x.T, plot_y.T, color=palette[internal_id], linewidth=1, alpha=alpha)
                else:
                    segments = [np.column_stack([x, y]) for x, y in zip(plot_x, plot_y)]
                    lc = LineCollection(segments, colors=[palette[internal_id]] * len(segments), alpha=alpha,
                                        linewidth=0.6)
                    ax.add_collection(lc)

                ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
                ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
                if flag_same_limit:
                    ax.set_xlim(x_lim_min, x_lim_max)
                    ax.set_ylim(y_lim_min, y_lim_max)

                # Save sub-figure
                sub_dir = "explain_{}_detail".format("category_" + mode if mode != 'divided' else info)
                fname = "explain_{}_{}_cls{}".format(
                    "category_" + mode if mode != 'divided' else info,
                    info,
                    internal_id
                )
                # reporter.save_figure_to_file(fig, os.path.join(sub_dir, fname))
                reporter.save_figure_to_file(fig, os.path.join(sub_dir, fname + ".png"))
                reporter.save_figure_to_file(fig, os.path.join(sub_dir, fname + ".svg"))
                plt.close(fig)

            pbar.update(1)

    # Legend handling
    if not ranking_legend:
        main_ax_legend_axes = sorted(main_ax_legend_axes, key=lambda t: t[1])
    if show_legend:
        main_ax.legend(
            *zip(*main_ax_legend_axes),
            title="Rule rank" if ranking_legend else "Interval id",
            title_fontsize=plot_temp.params["fontsize.legend.title"],
            fontsize=plot_temp.params["fontsize.legend"],
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )

    # Save main figure
    fig_name = "explain_main_{}_{}".format(
        "category_" + mode if mode != 'divided' else info,
        info
    )
    reporter.add_figure_by_data(main_fig, fig_name, title=f"explain_{mode}_{info}")
    plt.close(main_fig)

    return show_category if mode != 'divided' else None


# 原始函数作为包装器
def data_in_alpha(*args, **kwargs):
    return generic_interpretable_plot(*args, mode='alpha', **kwargs)


def data_in_max(*args, **kwargs):
    return generic_interpretable_plot(*args, mode='max', **kwargs)


def divided_data_show(*args, **kwargs):
    return generic_interpretable_plot(*args, mode='divided', **kwargs)