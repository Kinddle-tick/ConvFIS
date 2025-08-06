#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :loss.py
# @Time      :2024/3/11 22:44
# @Author    :Oliver
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from ...Tracker.Reporter import Report
from ...reporter import Reporter
from ...painter_format import PlotTemplate


def draw_loss_pic(loss_dict: dict, reporter: Reporter,plot_temp:PlotTemplate,):
    rtn_fig = {}
    # plotter_dict = {} if plotter_dict is None else plotter_dict
    # plotter = Plotter(**plotter_dict)
    for name, loss_data in loss_dict.items():
        fig_step = plt.figure(**plot_temp.temp_fig())
        fig_time = plt.figure(**plot_temp.temp_fig())
        ax_step = fig_step.add_subplot(111)
        ax_time = fig_time.add_subplot(111)
        # fig, (ax_step, ax_time) = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)
        # fig, (ax_step, ax_time) = plotter.gene_subplots(1,2)
        plt.xticks(rotation=30)
        # ax_step, ax_time = ax
        ax_time.set_xlabel("Relate Time(s)", fontsize=plot_temp.params["fontsize.label"])
        ax_step.set_xlabel("epoch", fontsize=plot_temp.params["fontsize.label"])
        tmp_time = pd.to_datetime(loss_data.iloc[:, 0])
        loss_data.iloc[:, 0] = (tmp_time - tmp_time[0])/ np.timedelta64(1, 's')
        train_loss_data = loss_data
        test_loss_data = loss_data[~pd.isna(loss_data.iloc[:, 4])]
        ax_time.plot(train_loss_data.iloc[:, 0], train_loss_data.iloc[:, 3], ".-", label=f'{name} train loss')
        ax_time.plot(test_loss_data.iloc[:, 0], test_loss_data.iloc[:, 4], ".-", label=f'{name} test loss')
        # ax_time.set_xticklabels(ax_time.get_xticklabels(), rotation=30)
        ax_time.legend()
        ax_step.plot(train_loss_data.iloc[:, 3], ".-", label=f'{name} train loss')
        ax_step.plot(test_loss_data.iloc[:, 4], ".-", label=f'{name} test loss')
        ax_step.legend()
        fig_path_step = reporter.save_figure_to_file(fig_step, f"loss_step_{name}")
        fig_path_time = reporter.save_figure_to_file(fig_time, f"loss_time_{name}")
        rtn_fig.update({f"{name}_step":fig_path_step})
        rtn_fig.update({f"{name}_time":fig_path_time})
        # plotter.close(fig)
    # del plotter
    return rtn_fig
