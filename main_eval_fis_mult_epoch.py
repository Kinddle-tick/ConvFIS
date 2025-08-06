#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/9 12:31
# @Author  : Oliver
# @File    : main_eval_fis_mult_epoch.py
# @Software: PyCharm
# import re
# import sys
# import objgraph
# import numpy as np
# import pandas as pd
# from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import model
import os
import torch
import json
from collections import OrderedDict
from config import load_config
# from frame.eval_process.eval_analyze import EvalConvFisHandler
# from frame.trainer import Trainer
# from frame.training_args import TrainingArguments
# from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
#
# from frame.util import EvalPrediction
# # from main_eval_fis_conv_dim import model_save_dir
# from model.fuzzy_inference import Layers
# from frame import AutoStandardScaler
# from frame.data_process.data_reader import DataReaderDataGroup
# # from datasets import load_dataset, Dataset
# from frame.data_process.data_dataset import SlideWindowDataset
# from torch.utils.data import DataLoader, Dataset, ConcatDataset
# from frame.data_process.data_transform import *
# from frame.eval_process import EvalTrackHandler
# from frame.util.generic import ModelOutput
from support_config_parser import ConfigManager
from frame import Trainer, EvalConvFisHandler

"""
研究模型训练整个流程的参数变化，需要模型训练时保存足够多数据进行配合
为了生成动态图，需要结合使用script_mk_gif.py
Studying parameter variations throughout the model training process
requires that enough data be saved for model training to work with.
In order to generate dynamic images, it is necessary to use script_mk_gif.py in conjunction with the
"""

model_root = "output/runs_log/20250718_18-30-17_ConvFis_quin33_20s_trained_Linux/models"
# model_root = "output/runs_log/20250724_16-08-27_ConvFis_quin33_20s_trained_Linux/models"
# model_root = "/mnt/d/PycharmProjects/my_platform_for_deeplearning_old/output/runs_log/20250724_13-51-38_ConvFis_quin33_20s_Linux/models"
model_dict = dict()
for _dir in os.listdir(model_root):
    if "cp" in _dir:
        with open(os.path.join(model_root, _dir, "config.json"), "r") as f:
            js = json.load(f)
            epoch = js["last_epoch"]+1
            if epoch not in model_dict:
                model_dict[epoch] = os.path.join(model_root, _dir)
    elif "Ep" in _dir:
        with open(os.path.join(model_root, _dir, "config.json"), "r") as f:
            js = json.load(f)
            epoch = js["last_epoch"]+1
            if epoch not in model_dict:
                model_dict[epoch] = os.path.join(model_root, _dir)
    elif "init_train" == _dir:
        model_dict[0] = os.path.join(model_root, _dir)
    elif "best_valid" == _dir:
        with open(os.path.join(model_root, _dir, "config.json"), "r") as f:
            js = json.load(f)
            epoch = js["last_epoch"] + 1
            model_dict[epoch] = os.path.join(model_root, _dir)

# middle_epoch = list(range(1, 100)) + list(range(100, 500, 2))

model_dict = OrderedDict({str(k):v for k,v in sorted(model_dict.items()) if k%16==0 or (k<16 and k%4==0) or True})

analyze_dict  = model_dict
section_name = "model_ConvFis"
data_sec_name = "dataset_quin33_20s"

prefix = "CompareAnte_" + data_sec_name

ignore_list = []
debug_first = True

if __name__ == '__main__':
    load_config()
    cfg = ConfigManager("default.ini", data_sec_name, save_dir_prefix = prefix)

    columns = cfg.data_processed.columns
    raw_tracks = [track[columns].to_numpy() for track in cfg.data_original]
    reporter = cfg.root_reporter
    # eval_hd = EvalConvFisHandler(reporter, mark_main_model="492", plot_mode="academic_mode")
    with EvalConvFisHandler(reporter, mark_main_model="492", plot_mode="academic_mode") as eval_hd:
        for name,model_save_dir in analyze_dict.items():
            if name in ignore_list:
                continue
            else:
                cfg.parser(section_name, data_sec_name, mode="test")

            model_load_path = model_save_dir
            trainer = Trainer(model_load_path, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
            # cfg.config_summary(trainer.reporter)

            model_output, predict_info = trainer.predict({name:cfg.valid_dataset})
            model_original_ep = {k:cfg.data_fn_inv_eval_prediction(v) for k,v in model_output.items()}

            with eval_hd.manager_raw_data() as raw_hd:
                raw_hd.update_data(model_original_ep)
                eval_hd.register_count_paras(name, trainer.model)
                eval_hd.register_defuzzifier_data(name,trainer.model.defuzzifier)
                eval_hd.indicator_update(name, predict_info)
                eval_hd.register_error()
                # fix_id = eval_hd.tool_random_id_sample(show_data_limit, eval_hd.get_data_len(), 42, True, False)
                # stride = 1
                fix_id = None
                stride = 10
                eval_hd.raw_eval_antecedent_category_of_track(show_category=16, stride=stride, fix_id=fix_id, frame=True)
                eval_hd.raw_eval_antecedent_category_of_track(show_category=64, stride=stride, fix_id=fix_id, frame=True)
                # eval_hd.eval_antecedent_category_of_track_alpha(show_category=16, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)
                # eval_hd.eval_antecedent_category_of_track_alpha(show_category=64, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)

        # reporter(str(eval_hd.indicator)).md().log()

        eval_hd.save_indicator_csv()


        #consequent
        reporter("##Consequent", flag_md=True)
        eval_hd.summary_firing_level(round_n=5)
        dfz_data = eval_hd.get_dfz_data()
        eval_hd.eval_diffusion_analyze(dfz_data, False)
        eval_hd.eval_combine_diffusion_analyze(dfz_data, True)

        eval_hd.eval_show_indicator(["mse"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
        eval_hd.eval_show_indicator(["mape"], x_alias="Rule Number", xtick_rotation=0,text_round=4)


        eval_hd.eval_bar_rule_frequency(mode="fl")
        eval_hd.eval_defuzzifier(16,4,True)
        eval_hd.eval_defuzzifier(None,4,True)
        eval_hd.eval_pearson_correlation_heatmap()
        # eval_hd.eval_pearson2force_directed(node_legend=True)
        eval_hd.eval_combine_pearson_force_directed(node_legend=True,frame=True)
        # eval_hd.eval_combine_pearson_force_directed(node_legend=True,frame=True)
        # eval_hd.eval_bar_antecedent_rule_frequency(sort=False)

        # fix_id = eval_hd.tool_random_id_sample(show_data_limit, eval_hd.get_data_len(), 42, True, False)
        # stride = 1

        # show_data_limit = 1000
        # fix_id = None
        # stride = 10
        # eval_hd.raw_eval_antecedent_category_of_track(show_category=16, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)
        # eval_hd.raw_eval_antecedent_category_of_track(show_category=64, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)
        # eval_hd.eval_antecedent_category_of_track_alpha(show_category=16, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)
        # eval_hd.eval_antecedent_category_of_track_alpha(show_category=64, show_data_limit=show_data_limit, stride=stride, fix_id=fix_id, rank_legend=False)

        # eval_hd.eval_predict_frames(48, pic_raw=3, pic_column=4)
        # eval_hd.eval_predict_frames(4, pic_raw=1, pic_column=1)
