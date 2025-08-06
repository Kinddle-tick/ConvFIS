#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5 16:34
# @Author  : oliver
# @File    : main_eval_fis_conv_dim.py
# @Software: PyCharm
# from main_train_fis_conv_dim import *
import re
import sys
import os
import torch
from collections import OrderedDict
# from config import *
from config import load_config
# from frame.eval_process.eval_analyze import EvalConvFisHandler
# from frame.trainer import Trainer
# from frame.training_args import TrainingArguments
from frame import Trainer, EvalConvFisHandler
from support_config_parser import ConfigManager
from frame import EvalTrackHandler
"""
比较若干个不同模型的推理效果, 会缺省一些独属于ConvFIS的分析。
Comparing the effects of Inference (prediction) across a number of different models,
will miss some analyses that are unique to ConvFIS.
"""
# path = "output/runs_log/20250425_16-50-23_iTransformer_quin33_6s_Linux"
section_map = {"ConvFIS":"model_ConvFis",
               "Transformer": "model_Transformer",
               "iTransformer": "model_iTransformer",
               "LSTM": "model_LSTM",
               "GPT-2": "model_GPT2",
               "DLinear":"model_DLinear"
               }

# 6s
model_dict_6s = OrderedDict({
    "GPT-2": "output/runs_log/20250429_12-34-43_Gpt_quin33_6s_Linux_train",
    "Transformer": "output/runs_log/20250419_23-45-04_Transformer_quin33_6s_Linux_train",
    "iTransformer": "output/runs_log/20250425_16-50-23_iTransformer_quin33_6s_Linux",
    "LSTM": "output/runs_log/20250421_16-20-28_LSTM_quin33_6s_Linux_train",
    "DLinear":"output/runs_log/20250426_10-53-33_DLinear_quin33_6s_Linux_train",
    "ConvFIS": "output/runs_log/20250424_20-32-48_ConvFis_quin33_6s_Linux_train",
})

# 10s
model_dict_10s = OrderedDict({
    "GPT-2": "output/runs_log/20250429_20-53-49_GPT2_quin33_10s_Linux_train",
    "Transformer": "output/runs_log/20250413_00-35-55_Transformer_quin33_10s_Linux_train",
    "iTransformer": "output/runs_log/20250414_22-29-04_iTransformer_quin33_10s_Linux_train",
    "LSTM": "output/runs_log/20250413_12-18-33_LSTM_quin33_10s_Linux_train",
    "DLinear":"output/runs_log/20250426_13-09-39_DLinear_quin33_10s_Linux_train",
    "ConvFIS": "output/runs_log/20250424_22-11-37_ConvFis_quin33_10s_Linux_train",
})

# 20s.
model_dict_20s = OrderedDict({
    "GPT-2": "output/runs_log/20250430_03-24-11_GPT2_quin33_20s_Linux_train",
    "Transformer": "output/runs_log/20250414_23-02-58_Transformer_quin33_20s_Linux_train",
    "iTransformer": "output/runs_log/20250415_21-08-10_iTransformer_quin33_20s_Linux_train",
    "LSTM": "output/runs_log/20250415_21-28-56_LSTM_quin33_20s_Linux_train",
    "DLinear":"output/runs_log/20250427_00-40-21_DLinear_quin33_20s_Linux_train",
    "ConvFIS": "output/runs_log/20250424_21-43-29_ConvFis_quin33_20s_Linux_train",
})

analyze_dict  = model_dict_20s
data_sec_name = "dataset_quin33_20s"
# analyze_dict  = model_dict_10s
# data_sec_name = "dataset_quin33_10s"
# analyze_dict  = model_dict_6s
# data_sec_name = "dataset_quin33_6s"

prefix = "CompareAll_" + data_sec_name

ignore_list = ["Transformer", "GPT-2"]
# ignore_list = []
if __name__ == '__main__':
    load_config()
    cfg = ConfigManager("default.ini", data_sec_name, save_dir_prefix = prefix,)

    columns = cfg.data_processed.columns
    raw_tracks = [track[columns].to_numpy() for track in cfg.data_original]
    reporter = cfg.root_reporter
    # eval_hd = EvalConvFisHandler(reporter,"ConvFIS", plot_mode="academic_mode")

    with EvalConvFisHandler(reporter,"ConvFIS", plot_mode="academic_mode") as eval_hd:
        for name, model_save_dir in analyze_dict.items():
            if name in ignore_list:
                continue
            else:
                cfg.parser(section_map[name], data_sec_name,  mode="test")

            model_load_path = os.path.join(model_save_dir, "models", "best_valid")
            trainer = Trainer(model_load_path, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
            cfg.config_summary(trainer.reporter)

            model_output, predict_info = trainer.predict({name: cfg.valid_dataset})
            model_original_ep = {k:cfg.data_fn_inv_eval_prediction(v) for k,v in model_output.items()}

            with eval_hd.manager_raw_data() as raw_hd:
                raw_hd.update_data(model_original_ep)
                eval_hd.register_count_paras(name, trainer.model)
                eval_hd.register_error()
                eval_hd.indicator_update(name, predict_info)

                # 整体效果把握
                eval_hd.raw_eval_willow(raw_tracks, sample_rule=20, used_tracks=30)
                eval_hd.raw_eval_predict_frames(48, pic_raw=3, pic_column=4)
        # eval_hd.finish()

        # reporter("[+] prediction info of those model:").md().log()
        # reporter(str(eval_hd.indicator)).md().log()
        eval_hd.save_indicator_csv()

        # 查看误差信息
        eval_hd.summary_metric()
        eval_hd.eval_compare_metrics_bar("ConvFIS", box_like=False, top_round=5)
        eval_hd.eval_show_indicator(["mse"],main_model_name="ConvFIS",)
        eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)

        # 查看参数与速度与mse的三视图
        # 参数-速度
        # eval_hd.set_plot_params({"figure.width_scale": 0.32, "figure.height_scale": 0.32,})
        eval_hd.eval_show_indicator(["trainable_params","eval_items_per_ms"],
                                    x_log=True, y_alias="Eval speed (it/ms)")
        # eval_hd.eval_show_indicator(["trainable_params","eval_time_per_item_ms"],
        #                             x_log=True, y_alias="Eval time (ms/it)")
        # # 速度-mse
        # eval_hd.eval_show_indicator(["eval_items_per_ms","mse"],
        #                             x_log=False, x_alias="Eval speed (it/ms)")
        # # eval_hd.eval_show_indicator(["eval_time_per_item_ms","mse"],
        # #                             x_log=False, x_alias="Eval time (ms/it)")
        # 参数-mse
        eval_hd.eval_show_indicator(["trainable_params","mse"],
                                    x_log=True,)
        # # 用参数作为散点图大小的 - 速度-mse
        # eval_hd.eval_show_indicator(["eval_items_per_ms","mse","trainable_params"],
        #                             y_log=False, z_log=True, x_alias="Eval speed (it/ms)")
        # # eval_hd.eval_show_indicator(["eval_items_per_ms","mse","trainable_params"],
        # #                             y_log=False, z_log=True, x_alias="Eval speed (it/ms)")
        # # eval_hd.eval_show_indicator(["eval_items_per_ms","mse","trainable_params"],
        # #                             y_log=False, z_log=False, x_alias="Eval speed (it/ms)")

        eval_hd.eval_show_indicator(["trainable_params"],
                                     y_log=True,)
        # eval_hd.set_plot_default()
        # 查看使用轨迹
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True, heat=True, heat_bins=200)




















