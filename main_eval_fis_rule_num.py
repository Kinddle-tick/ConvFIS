#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/9 12:31
# @Author  : Oliver
# @File    : main_eval_fis_rule_num.py
# @Software: PyCharm
import re
import sys

import numpy as np
import pandas as pd
# from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import model
from collections import OrderedDict
from config import *
from frame.eval_process.eval_analyze import EvalConvFisHandler
from frame.trainer import Trainer
from frame.training_args import TrainingArguments
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

from frame.util import EvalPrediction
# from main_eval_fis_conv_dim import model_save_dir
from model.fuzzy_inference import Layers
# from frame import AutoStandardScaler
from frame.data_process.data_reader import DataReaderDataGroup
# from datasets import load_dataset, Dataset
from frame.data_process.data_dataset import SlideWindowDataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from frame.data_process.data_transform import *
from frame.eval_process import EvalTrackHandler
from frame.util.generic import ModelOutput
from support_config_parser import ConfigManager
from frame import Trainer, EvalConvFisHandler

"""
探索规则数量对性能的影响
Exploring the impact of the number of rules on performance
"""
model_dict = OrderedDict({
    "1": "output/runs_log/20250506_13-44-09_ConvFis_quin33_20s_Linux_train",
    "2": "output/runs_log/20250506_14-05-43_ConvFis_quin33_20s_Linux_train",
    "4": "output/runs_log/20250506_14-22-57_ConvFis_quin33_20s_Linux_train",
    "8": "output/runs_log/20250506_14-34-32_ConvFis_quin33_20s_Linux_train",
    "16": "output/runs_log/20250506_14-48-24_ConvFis_quin33_20s_Linux_train",
    "32": "output/runs_log/20250506_15-05-35_ConvFis_quin33_20s_Linux_train",
    "64": "output/runs_log/20250506_15-44-05_ConvFis_quin33_20s_Linux_train",
    "128": "output/runs_log/20250506_16-56-08_ConvFis_quin33_20s_Linux_train",
})

analyze_dict  = model_dict
section_name = "model_ConvFis"
data_sec_name = "dataset_quin33_20s"

prefix = "CompareFISRule" + data_sec_name

ignore_list = []
if __name__ == '__main__':
    load_config()
    cfg = ConfigManager("default.ini", data_sec_name, save_dir_prefix = prefix)

    columns = cfg.data_processed.columns
    raw_tracks = [track[columns].to_numpy() for track in cfg.data_original]
    reporter = cfg.root_reporter
    eval_hd = EvalConvFisHandler(reporter, mark_main_model="64", plot_mode="academic_mode")

    for name,model_save_dir in analyze_dict.items():
        if name in ignore_list:
            continue
        else:
            print(name)
            cfg.parser(section_name, data_sec_name, mode="test")

        model_load_path = os.path.join(model_save_dir, "models", "best_valid")
        trainer = Trainer(model_load_path, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
        # cfg.config_summary(trainer.reporter)

        model_output, predict_info = trainer.predict({name:cfg.valid_dataset})
        model_original_ep = {k:cfg.data_fn_inv_eval_prediction(v) for k,v in model_output.items()}

        with eval_hd.manager_raw_data():
            for k, v in model_original_ep.items():
                eval_hd.add_data(k, v)
            eval_hd.register_count_paras(name, trainer.model)
            eval_hd.indicator_update(name, predict_info)

    # reporter("[+] prediction info of those model:").md().log()
    # reporter(str(eval_hd.indicator)).md().log()
    eval_hd.save_indicator_csv()

    eval_hd.eval_show_indicator(["trainable_params"], x_alias="Rule Number",  y_log=True, xtick_rotation=0,
                                text_round=0)
    eval_hd.eval_show_indicator(["mse"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
    eval_hd.eval_show_indicator(["mape"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
    # eval_hd.eval_show_indicator(["mse"], x_alias="Rule Number",y_log=True, xtick_rotation=0,text_round=4)
    eval_hd.eval_show_indicator(["trainable_params", "mse"], x_log=True)

    eval_hd.eval_compare_metrics_bar( box_like=False)

    reporter("##Consequent", flag_md=True)
    eval_hd.summary_firing_level(round_n=5)
    dfz_data = eval_hd.get_dfz_data()
    eval_hd.eval_diffusion_analyze(dfz_data, False)
    eval_hd.eval_defuzzifier(None)

    eval_hd.eval_bar_rule_frequency()
    eval_hd.eval_pearson_correlation_heatmap()
    eval_hd.eval_pearson2force_directed(node_legend=True)

    # eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)

    # eval_hd.eval_compare_predict_sample(10, save_to_md=True, seed = 20250515)
    # eval_hd.eval_predict_frames(48, pic_raw=3, pic_column=4)
    # eval_hd.eval_predict_frames(4, pic_raw=1, pic_column=1)
