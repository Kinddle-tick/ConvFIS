#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :regression_eval.py
# @Time      :2024/1/2 20:57
# @Author    :Oliver
from enum import Enum
from typing import Dict

import torch
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from ..error_fuction import _ErrorCalculator
# from ...Tracker.Reporter import Report
from ...painter_format import PlotTemplate
from ...reporter import Reporter
from ...util import EvalPrediction


# def _path_dir(root_dir, type="mse"):
#     track_dir = os.path.join(root_dir, "track_analyze", type)
#     if not os.path.exists(track_dir):
#         os.makedirs(track_dir)
#     return track_dir
#
#
# def analyze_save(_raw_data, root_dir, prefix):
#     track_dir = _path_dir(root_dir)
#     for idx, m in enumerate(_raw_data):
#         pd.DataFrame(m).to_csv(os.path.join(track_dir, f"{prefix}_mse_data_{idx}.csv"))
#     torch.save(_raw_data, os.path.join(track_dir, f"{prefix}_mse_data_total.pt"))
#     return track_dir
#
#
# def analyze_load(root_dir,file_name="mse_data_total.pt"):
#     track_dir = _path_dir(root_dir)
#     return torch.load(os.path.join(track_dir, file_name))
#
#
# def mse_data(x, y, batch_axis=0):
#     error = (x - y) ** 2
#     mean = np.mean(error, axis=batch_axis)
#     std = np.std(error, axis=batch_axis, ddof=1)
#     down_quan = np.quantile(error,0.025, method="lower", axis=batch_axis)
#     up_quan = np.quantile(error,0.975, method="higher", axis=batch_axis)
#     return mean, up_quan, down_quan
#
#
# def mae_data(x, y, batch_axis=0):
#     error = np.abs(x - y)
#     mean = np.mean(error, axis=batch_axis)
#     std = np.std(error, axis=batch_axis, ddof=1)
#     down_quan = np.quantile(error,0.025, method="lower", axis=batch_axis)
#     up_quan = np.quantile(error,0.975, method="higher", axis=batch_axis)
#     return mean, up_quan, down_quan
#
#
# def rmse_data(x, y, batch_axis=0):
#     """只返回0维数据"""
#     error = (x - y) ** 2
#     mean = np.sqrt(np.mean(error, axis=batch_axis))
#     std = 0
#     down_quan = mean
#     up_quan = mean
#     return mean, up_quan, down_quan
#
#
# def euclid_data(x, y, batch_axis=0, euclid_axis=-1):
#     error = np.sqrt(np.sum(((x - y) ** 2), axis=euclid_axis,keepdims=True))
#     mean = np.mean(error, axis=batch_axis)
#     std = np.std(error, axis=batch_axis, ddof=1)
#     down_quan = np.quantile(error,0.025, method="lower", axis=batch_axis)
#     up_quan = np.quantile(error,0.975, method="higher", axis=batch_axis)
#     return mean, up_quan, down_quan
#
# _metrics_strategies = {
#     "mse": mse_data,
#     "rmse": rmse_data,
#     "mae": mae_data,
#     "euclid": euclid_data
# }

def metric_compare_by_time(error_dict:dict[str, _ErrorCalculator], main_model_name:str, reporter:Reporter,plot_temp:PlotTemplate,
                           x_unit_s=1, show_dims:list=(0,), columns_name=None, show_dims_method="mean",
                           draw_confidence_interval=False,
                           ):
    if show_dims_method not in ["mean", "individual"]:
        reporter(f"[-] invalid show_dims_method: {show_dims_method}")
        return

    rtn={}
    error_name=""
    file_name = ""

    fig = plt.figure(**plot_temp.temp_fig())
    palette = plot_temp.color_palette(len(error_dict),
                                      list(error_dict.keys()).index(main_model_name) if main_model_name in error_dict else None)
    count=0
    for data_name, error in error_dict.items():
        # error_mean = error.mean
        if error_name == "":
            error_name=error.formal_name
            file_name = error.nick_name

        x = (np.arange(0, error.mean.shape[-2]) + 1) * x_unit_s

        if show_dims_method == "mean":
            if draw_confidence_interval:
                plt.fill_between(x, np.mean(error.quan_up[...,show_dims]), np.mean(error.quan_down[...,show_dims], axis=-1),
                                 alpha=0.2, color=palette[count], zorder=-count)
            plt.plot(x, np.mean(error.mean[...,show_dims], axis=-1), label=f"{data_name}" ,
                     color=palette[count], zorder=-count if data_name != main_model_name else 100)

        elif show_dims_method == "individual":
            label = [columns_name[i] if i<len(columns_name)else f"dim{i}" for i,dim in enumerate(show_dims)]
            for i in show_dims:
                if draw_confidence_interval:
                    plt.fill_between(x, error.quan_up[..., i], error.quan_down[..., i],
                                     alpha=0.2, color=palette[count], zorder=-count)
                plt.plot(x, error.mean[..., i], label=f"{data_name}_{label[i]}", color=palette[count], zorder=-count if data_name != main_model_name else 100)

        plt.legend()
        # plt.legend(fontsize=plt.rcParams["legend.fontsize"])
        plt.xlabel("predict_time (s)", fontsize=plot_temp.params["fontsize.label"])
        plt.ylabel(error_name, fontsize=plot_temp.params["fontsize.label"])
        rtn[data_name] = np.mean(error.mean[...,show_dims])
        count+=1

    reporter.add_figure_by_data(fig, save_name=f"{file_name}_compare_by_time_pic",
                                title=f"# {error_name} of tracks")
    plt.close(fig)
    # plotter.close(fig)
    return rtn

def metric_bar_compare(error_dict:dict[str, _ErrorCalculator], reporter:Reporter, plot_temp:PlotTemplate,
                        main_model_name,show_dims:list=(0,), draw_horizon_line=True, log_y=False, top_number=True, text_round=4, box_like=True):
    """
    :param error_dict:  error的字典，一般是model名:和error的结果
    :param prefix:      保存图片的前缀， 一般是计算error的名字 只用于文件名
    :param reporter:    reporter
    :param plot_temp:   plot_temp
    :param main_model_name:     选定一个主要模型，将其在最后一个展示
    :param show_dims:   选择展示哪些维度（同一种方法的不同维度会直接取平均）
    :param draw_horizon_line:   是否根据主要模型绘制横线
    :param log_y:       是否将y轴log化
    :param top_number:  是否在顶端标数字
    :param box_like:    是否绘制箱型图类似的触须
    :return:
    """
    # rtn={}
    # for model_name, error in error_dict.items():
    #     error_mean = np.mean(error.mean[...,show_dims])
    rtn = {}
    # Prepare model order with main model last
    model_names = list(error_dict.keys())
    palette = plot_temp.color_palette(len(model_names), None if main_model_name not in model_names else model_names.index(main_model_name))
    palette = pd.Series(palette, index=model_names)


    means, stds = [], []
    error_name = ""
    file_name = ""
    for model in model_names:
        error = error_dict[model]
        # Calculate mean across selected dimensions
        error_mean = np.mean(error.mean[..., show_dims])
        means.append(error_mean)
        rtn[model] = error_mean  # Store in results
        if error_name == "":
            error_name = error.formal_name
            file_name = error.nick_name
        # Calculate std if needed
        if box_like:
            error_std = np.mean(error.std[..., show_dims])
            error_se = error_std /np.sqrt(error.n)
            stds.append(error_se)
        else:
            stds.append(0)

    # Create plot
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)

    # Plot bars with error bars
    x = np.arange(len(model_names))
    if box_like and any(stds):
        bars = ax.bar(x, means, yerr=stds, capsize=5, error_kw={'elinewidth': 1, 'capsize': 3}, color = palette)
    else:
        bars = ax.bar(x, means, color = palette)

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(error_name, fontsize=plot_temp.params["fontsize.label"])
    ymin, ymax = ax.get_ylim()

    # # 动态计算扩展比例
    # max_bar_height = max([bar.get_height() for bar in bars])
    # expansion_factor = 1.05 if max_bar_height > 0 else 0.95
    # new_ymax = max(ymax, max_bar_height * expansion_factor)

    # 设置扩展后的y轴范围
    # ax.set_ylim(top=new_ymax * 1.05)  # 强制扩展5%
    ax.set_ylim(top=ymin + (ymax-ymin) * 1.05)  # 强制扩展5%

    # Add horizontal line for main model
    if draw_horizon_line and main_model_name in model_names:
        ax.axhline(means[-1], color='gray', linestyle='--', linewidth=1)

    # Add value labels
    if top_number:
        for bar in bars:
            height = bar.get_height()
            if text_round <= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(round(height,text_round))}',
                        ha='center', va='bottom', fontsize=plot_temp.params["fontsize.bar_top"])
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{round(height,text_round)}',
                        ha='center', va='bottom', fontsize=plot_temp.params["fontsize.bar_top"])

            # ax.text(bar.get_x() + bar.get_width() / 2, height,
            #         f'{height:.3f}', ha='center', va='bottom', fontsize=plot_temp.params["fontsize.bar_top"])

    # Set log scale
    if log_y:
        ax.set_yscale('log')

    # Save and close
    plt.tight_layout()
    # reporter.save_fig(fig, f"{prefix}_bar_compare")
    reporter.add_figure_by_data(fig, save_name=f"{file_name}_bar_compare",
                                title=f"# Compare of {error_name} in tracks")
    plt.close(fig)

    return rtn

    pass

# def metric_compare_by_time_old(eval_predict_dict:dict[str,EvalPrediction],reporter:Reporter,plot_temp:PlotTemplate,
#                            dims:list, columns_name=None,
#                            method="mse", x_unit_s=1, draw_confidence_interval=True):
#     # plotter_dict = {} if plotter_dict is None else plotter_dict
#
#     # plotter = Plotter(**plotter_dict)
#     if method not in _metrics_strategies:
#         reporter("there is no metric method named {}, pass this step".format(method))
#         return None
#     else:
#         method_func = _metrics_strategies[method]
#     columns_name = dims if columns_name is None else columns_name
#     rtn={}
#     # fig = plt.figure(dpi=128, figsize=(10, 6))
#     fig = plt.figure(**plot_temp.temp_fig())
#     # fig,_ = plotter.gene_half_figure()
#     for data_name, data in eval_predict_dict.items():
#         prediction, label = data.output[...,dims], data.target[...,dims]
#         x = (np.arange(0, prediction.shape[-2]) + 1) * x_unit_s
#         mean, up, down = method_func(prediction, label)
#         for i in range(mean.shape[-1]):
#             plt.plot(x, mean[..., i], label=f"{data_name}_{columns_name[i]}")
#             if draw_confidence_interval:
#                 plt.fill_between(x, up[..., i], down[..., i], alpha=0.2)
#         plt.legend(fontsize=plot_temp.plot_para_dict["legend.fontsize"])
#         rtn[data_name] = np.mean(mean)
#     reporter.add_figure_by_data(fig, save_name=f"{method}_compare_pic",
#                                 title=f"# {method} of tracks")
#     plt.close(fig)
#     # plotter.close(fig)
#     return rtn


# def metric_analyze(prediction, label, reporter:Reporter,plot_temp:PlotTemplate, note_text, columns=None,
#                    method="mse", x_unit_s=1, draw_confidence_interval=True, ):
#     # plotter_dict = {} if plotter_dict is None else plotter_dict
#
#     # plotter = Plotter(**plotter_dict)
#     if method not in _metrics_strategies:
#         reporter("there is no metric method named {}, pass this step".format(method))
#         return None
#     else:
#         method_func = _metrics_strategies[method]
#     x = (np.arange(0, prediction.shape[-2]) + 1) * x_unit_s
#
#     mean, up, down = method_func(prediction,label)
#     # fig = plt.figure(dpi=128, figsize=(10, 6))
#     fig = plt.figure(**plot_temp.temp_fig())
#     # fig, _ = plotter.gene_half_figure()
#     plt.plot(x, mean)
#     if draw_confidence_interval:
#         for i in range(mean.shape[-1]):
#             plt.fill_between(x, up[...,i], down[...,i], alpha=0.2)  # 填充色块
#     plt.legend(columns, fontsize=plot_temp.plot_para_dict["legend.fontsize"])
#     reporter.add_figure_by_data(fig, save_name=f"{method}_Avg_{note_text}_pic",
#                                 title=f"# {note_text}: {method} of tracks in dataset {note_text}")
#     plt.close(fig)
#     # plotter.close(fig)
#     return np.mean(mean)

# def pic_single_para(raw_data, reporter: Reporter, note_text,columns=None,
#                     x_unit_s=1, key_word="MSE", focus_average=True,
#                     start="df", end="df"):
#     # 分参数分析
#     # xs = []
#     x_tmp= None
#     for idx in range(raw_data.shape[-1]):
#         fig = plt.figure(dpi=128, figsize=(10, 6))
#
#         ax = plt.subplot(111)
#         ax.set_xlabel("Predict Time(s)", fontsize=20)
#         ax.set_ylabel(key_word, fontsize=20)
#         ax.xaxis.set_tick_params(labelsize=16)
#         ax.yaxis.set_tick_params(labelsize=16)
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
#         x = (np.arange(0, raw_data.shape[-2]) + 1) * x_unit_s
#         # xs.append(max(x))
#         if x_tmp is None:
#             x_tmp = x
#         if focus_average:
#             ax.plot(x, raw_data[..., idx].mean(axis=0), label='Avg', alpha=1)
#             ax.plot(x, raw_data[..., idx].T, linewidth=0.5, alpha=0.4)
#         else:
#             ax.plot(x, raw_data[..., idx].mean(axis=0), label='Avg')
#             ax.plot(x, raw_data[..., idx].T)
#         ax.legend(['Avg', *range(raw_data.shape[-2])],
#                   loc="lower center", bbox_to_anchor=(0, 1.05, 1, 0.25), ncol=9, fontsize=plot_temp.plot_para_dict["legend.fontsize"])
#         # ax.set_xlim(0, max(x))
#         ax.set_xlim(0)
#         # plt.plot(mse_raw_data[:, :, idx].T)
#         reporter.add_figure_by_data(fig, save_name=f"{key_word}_({start}_{end})_para_{idx}_{note_text}_pic",
#                                     title=f"# {note_text} {key_word} from {start} to {end}:预测的第{idx}个参数轨迹之间的表现")
#         reporter(f"第{idx}个参数所有步长的mse均值：{raw_data[..., idx].mean()}", flag_md=True, flag_log=True)
#         plt.close(fig)
#
#     # 参数间分析（仅看均值
#
#     fig = plt.figure(dpi=128, figsize=(10, 6))
#     plt.plot(x_tmp, np.mean(raw_data, axis=0), label=key_word)
#     # plt.xlim(0, max(xs))
#     plt.xlim(0)
#     plt.legend(["x", "y", "h", "v"], fontsize=plot_temp.plot_para_dict["legend.fontsize"])
#     reporter.add_figure_by_data(fig, save_name=f"{key_word}_Avg_{note_text}_pic",
#                                 title=f"# {note_text}: mean of all tracks")
#     plt.close(fig)
#
# def pic_compare(raw_dict:dict, reporter: Reporter, step_slice, dims, x_unit_s=1,key_word="MSE",weight=1,
#                 start="df", end="df"):
#     avg_dict={}
#     for prefix, raw_data in raw_dict.items():
#         avg = raw_data.mean(axis=0)
#         x = (np.arange(0, raw_data.shape[-2]) + 1) * x_unit_s
#         avg_dict.update({prefix: (x, avg)})
#
#     for dim in dims:
#         fig = plt.figure(dpi=128, figsize=(10, 6))
#         ax = plt.axes()
#         ax.set_ylabel(key_word,fontsize=20)
#         ax.set_xlabel("Predict Time(s)", fontsize=20)
#         ax.xaxis.set_tick_params(labelsize=16)
#         ax.yaxis.set_tick_params(labelsize=16)
#         xs = []
#         for prefix, (x, avg_data) in avg_dict.items():
#             ax.plot(x, avg_data[..., dim], label=f"{prefix}")
#             # ax.plot(x[step_slice], avg_data[..., step_slice, dim], label=f"{prefix}")
#             xs.append(max(x))
#         ax.legend(fontsize=plot_temp.plot_para_dict["legend.fontsize"])
#         # ax.set_xlim(0, max(xs))
#         ax.set_xlim(0)
#         reporter.add_figure_by_data(fig, save_name=f"compare_{key_word}_({start}_{end})_para_{dim}_pic",
#                                     title=f"# from {start} to {end}, {key_word}  compare of dim {dim}")
#         plt.close(fig)
#
#     return avg_dict

