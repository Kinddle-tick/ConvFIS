#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/9 16:16
# @Author  : oliver
# @File    : predict_show.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# from ._painter_config import *
from ...painter_format import PlotTemplate
from matplotlib import pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width
from tqdm.auto import tqdm

from ...reporter import Reporter

_input_color = "green"
_target_color = "orange"
_predict_color = "red"



def predict_show(input_data, target_data, predict_data, reporter:Reporter,plot_temp:PlotTemplate, prefix,ids=None,
                 column=3,raw=3,save_dir_name="predict_frame", flag_show_map_line=True):
    # plotter = Plotter(**plotter_dict)
    # plotter_dict = {} if plotter_dict is None else plotter_dict
    # plotter = Plotter(**plotter_dict)
    data_len = len(input_data)
    ids = range(data_len) if ids is None else ids
    pic_id = 0
    pic_paths = []
    with tqdm(total=data_len, desc=f"pred frames of {prefix}") as pbar:
        desc_list = []
        for i in range(data_len):
            if i%(column*raw) == 0:
                fig = plt.figure(**plot_temp.temp_fig(width_scale=0.5*column, height_scale=0.5*raw))
                # fig = plt.figure(dpi=360, figsize=(4*column, 4*raw))
                desc_list.clear()
            ax = plt.subplot(raw, column, i% (column * raw) +1)
            # ax.set_title(f"{prefix}: {ids[i]}")
            desc_list.append(f"{ids[i]}")
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
            ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
            ax.plot(input_data[i,...,0],input_data[i,...,1],"-", color=_input_color, label="input",alpha=0.6)
            ax.plot(target_data[i,...,0], target_data[i,...,1],".-", color=_target_color, label="target",alpha=0.6)
            ax.plot(predict_data[i,...,0], predict_data[i,...,1],".-", color=_predict_color, label="predict",alpha=0.6)
            if flag_show_map_line:
                ax.plot([target_data[i,...,0],predict_data[i, ..., 0]],
                        [target_data[i,...,1], predict_data[i, ..., 1]], "-",lw=0.2,alpha=0.9, color="black")

            ax.legend(fontsize= plot_temp.params["fontsize.legend"])

            if (i+1)%(column*raw) == 0 or i == data_len-1:
                plt.tight_layout()
                plt.legend()
                if pic_id == 0:
                    title_str = f"Predict Frame of {prefix}"
                else:
                    title_str = ""
                pic_path = reporter.add_figure_by_data(fig,os.path.join(save_dir_name,f"predict_frame_{prefix}_{pic_id}.png"),
                                             title_str, f"{prefix}:"+", ".join(desc_list), f"predict_frame_{pic_id}",)
                pic_id += 1
                pic_paths.append(pic_path)
                plt.close(fig)
            pbar.update(1)
    return pic_paths


def compare_prediction(input_data,target_data, predict_datas:dict, main_model_name, reporter:Reporter, plot_temp:PlotTemplate,
                       main_alpha=0.8, alpha_list = 0.6,
                       save_dir_name="predict_compare"):
    """
    对于相同的输入和真值，将多个预测放在同一张图里展示
    :param input_data: 公用输入, ndarray
    :param target_data:  公用输出, ndarray
    :param main_model_name:  主要模型名称，会涉及到颜色相关
    :param predict_datas:  不同模型的预测效果{model_name: prediction_data(ndarray)}
    :param reporter: 用于保存图片的方法 usage: path = reporter.save_figure_to_file(fig, save_name)
    :param plot_temp: 管理格式的方法 usage: fig = plt.figure(**plot_temp.temp_fig())
    :param save_dir_name: 可能涉及较多图片，这里管理收集这些图片的目录名
    :return: fig的path
    """
    model_names = list(predict_datas.keys())
    # 调色盘 以及将主模组提取出来 (主模组放在最后一个位置，调色盘的第一个颜色是特殊颜色，放在对应）
    palette = plot_temp.color_palette(len(model_names), None if main_model_name not in model_names else model_names.index(main_model_name))
    palette_add = plot_temp.color_palette(len(model_names) + 2)[-2:]
    palette = pd.Series(palette, index=model_names)
    # if main_model_name in model_names:
    #     old_order =[main_model_name] + [i for i in model_names if i != main_model_name]
    #     palette = pd.Series(palette, index=old_order)
    #     palette = palette.reindex(model_names)
    # else:
    #     palette = pd.Series(palette, index=model_names)

    if isinstance(alpha_list, float):
        alpha_list = [alpha_list] * len(model_names)
    if len(alpha_list) < len(model_names):
        alpha_list += [0.6] * (len(model_names) - len(alpha_list))
    alphas = pd.Series(alpha_list, index=model_names)
    alphas[main_model_name] = main_alpha

    # 创建Figure和Axes
    fig = plt.figure(**plot_temp.temp_fig())
    ax = fig.add_subplot(111)

    # # 生成时间轴（假设input和target是连续序列）
    # input_len = input_data.shape[0]
    # target_len = target_data.shape[0]
    # time_input = np.arange(input_len)
    # time_target = np.arange(input_len, input_len + target_len)


    # 绘制各模型预测结果
    for model in model_names:
        pred_data = predict_datas[model]

        ax.plot( *pred_data,
                color=palette[model],
                linewidth=1,
                linestyle='-',
                label=model,
                alpha=alphas[model],
                zorder=101 if model == main_model_name else 10)

    # 绘制输入序列
    ax.plot(*input_data, color=palette_add[0], linewidth=1, linestyle='-', label='Input', alpha=0.8)

    # 绘制目标序列
    ax.plot(*target_data, color=palette_add[1], linewidth=1.2, label='Target',zorder=100)

    divider = Rectangle((0, 0), 1, 1, fc="w", ec="k", linestyle="-", linewidth=1, alpha=0)
    handles, labels = plt.gca().get_legend_handles_labels()

    # 在适当位置插入分割线
    handles.insert(-2, divider)
    labels.insert(-2, '')
    # 添加图例和标签
    ax.legend(handles, labels, title="Rule number", loc='best', fontsize= plot_temp.params["fontsize.legend"])
    ax.set_xlabel("Longitude(°)", fontsize=plot_temp.params["fontsize.label"])
    ax.set_ylabel("Latitude(°)", fontsize=plot_temp.params["fontsize.label"])
    # ax.set_title("Model Prediction Comparison")

    # 保存图片
    save_path = reporter.save_figure_to_file(fig, save_dir_name)
    plt.close(fig)

    return save_path





