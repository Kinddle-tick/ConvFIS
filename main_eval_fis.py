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
from frame import Trainer, EvalConvFisHandler

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
研究任意一个或多个FIS模型，并进行多种分析
Study any one or more FIS models and perform multiple analyses
"""
model_dict = OrderedDict({
    # "2r": "output/runs_log/20250506_14-05-43_ConvFis_quin33_20s_Linux_train",
    # "4r": "output/runs_log/20250506_14-22-57_ConvFis_quin33_20s_Linux_train",
    # "8r": "output/runs_log/20250506_14-34-32_ConvFis_quin33_20s_Linux_train",
    # "16r": "output/runs_log/20250506_14-48-24_ConvFis_quin33_20s_Linux_train",
    # "32r": "output/runs_log/20250506_15-05-35_ConvFis_quin33_20s_Linux_train",
    # "64r": "output/runs_log/20250506_15-44-05_ConvFis_quin33_20s_Linux_train",
    # "128r": "output/runs_log/20250506_16-56-08_ConvFis_quin33_20s_Linux_train",
    # "37": "output/runs_log/20250612_10-03-38_ConvFis_quin33_20s_Linux_train",
    # "23": "output/runs_log/20250612_10-55-23_ConvFis_quin33_20s_Linux_train",
    # "78": "output/runs_log/20250612_11-44-46_ConvFis_quin33_20s_Linux_train",
    # "15": "output/runs_log/20250612_12-59-48_ConvFis_quin33_20s_Linux_train",
    # "64": "output/runs_log/20250612_16-14-04_ConvFis_quin33_20s_Linux_train",
    # "norm37": "output/runs_log/20250627_17-18-03_ConvFis_quin33_20s_Linux_train",
    # "norm23": "output/runs_log/20250627_18-41-47_ConvFis_quin33_20s_Linux_train",
    # "norm78": "output/runs_log/20250627_19-24-35_ConvFis_quin33_20s_Linux_train",
    # "norm15": "output/runs_log/20250627_20-02-09_ConvFis_quin33_20s_Linux_train",
    # "norm64": "output/runs_log/20250627_20-34-58_ConvFis_quin33_20s_Linux_train",
    "norm_base": "output/runs_log/20250516_15-21-54_ConvFis_quin33_20s_Linux_train",
    # "norm_rollback_15":"/mnt/d/PycharmProjects/my_platform_for_deeplearning_old/output/runs_log/20250723_21-03-21_ConvFis_quin33_20s_Linux_train",
    # "norm_rollback_15_2":"/mnt/d/PycharmProjects/my_platform_for_deeplearning_old/output/runs_log/20250724_13-03-28_ConvFis_quin33_20s_Linux_train",
    # "norm_rollback_15_3":"/mnt/d/PycharmProjects/my_platform_for_deeplearning_old/output/runs_log/20250724_13-51-38_ConvFis_quin33_20s_Linux",
    # "norm_rollback_16":"/mnt/d/PycharmProjects/my_platform_for_deeplearning_old/output/runs_log/20250723_23-59-38_ConvFis_quin33_20s_Linux_train",
    "norm_new": "output/runs_log/20250424_21-43-29_ConvFis_quin33_20s_Linux_train"
})
path = "output/runs_log/20250516_15-21-54_ConvFis_quin33_20s_Linux_train"

analyze_dict  = model_dict
section_name = "model_ConvFis"
data_sec_name = "dataset_quin33_20s"

prefix = "Correlate" + data_sec_name
combine_model = ["37", "23", "78", "15", "64"]
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
        cfg.config_summary(trainer.reporter)

        model_output, predict_info = trainer.predict({name:cfg.valid_dataset})
        model_original_ep = {k:cfg.data_fn_inv_eval_prediction(v) for k,v in model_output.items()}
        with eval_hd.manager_raw_data() as raw_hd:
        # for k, v in model_original_ep.items():
        #     raw_hd.add_data(k, v)
            raw_hd.update_data(model_original_ep)
            eval_hd.register_count_paras(name, trainer.model)
            eval_hd.indicator_update(name, predict_info)
            eval_hd.register_defuzzifier_data(name, trainer.model.defuzzifier)
            eval_hd.register_error()

            # show_data_limit = 100
            # eval_hd.raw_eval_antecedent_category_of_track(show_category=5, stride=10)
            eval_hd.raw_eval_antecedent_category_of_track(show_category=16, stride=10)
            eval_hd.raw_eval_antecedent_category_of_track(stride=10)

    # reporter("[+] prediction info of those model:").md().log()
    # reporter(str(eval_hd.indicator)).md().log()
    eval_hd.save_indicator_csv()

    eval_hd.eval_defuzzifier(10, 4, True)
    eval_hd.eval_defuzzifier(None,4, True)

    # Rule statistical analysis
    reporter("##Rule statistical analysis", flag_md=True)
    # eval_hd.eval_bar_antecedent_rule_frequency(sort=False, mode="rw")
    # eval_hd.eval_bar_antecedent_rule_frequency(sort=True, mode="rw")
    eval_hd.eval_bar_rule_frequency(sort=False, mode="fl")
    eval_hd.eval_bar_rule_frequency(sort=True, mode="fl")

    eval_hd.eval_pearson_correlation_heatmap()
    # eval_hd.eval_pearson2force_directed(node_legend=True)
    eval_hd.eval_combine_pearson_force_directed(frame=True)
    # eval_hd.eval_pearson_correlation(node_legend=True,mode="fl")

    #consequent
    reporter("##Consequent", flag_md=True)
    eval_hd.summary_firing_level(round_n=5)
    dfz_data = eval_hd.get_dfz_data()
    # eval_hd.eval_diffusion_analyze({k:v for k,v in dfz_data.items() if k in combine_model}, True)
    eval_hd.eval_diffusion_analyze(dfz_data, False)
    eval_hd.eval_combine_diffusion_analyze(dfz_data, frame=True)

    # antecedent
    reporter("##Antecedent", flag_md=True)


    # sth else
    reporter("##Sth else", flag_md=True)
    eval_hd.eval_show_indicator(["trainable_params"], x_alias="Rule Number",  y_log=True, xtick_rotation=0,
                                text_round=0)
    # eval_hd.eval_show_indicator(["mse"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
    eval_hd.eval_show_indicator(["mape"], x_alias="Rule Number", xtick_rotation=0,text_round=4)

    # eval_hd.eval_show_indicator(["fl_avg_pearson_non_diag"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
    # eval_hd.eval_show_indicator(["mf_avg_pearson_non_diag"], x_alias="Rule Number", xtick_rotation=0,text_round=4)
    # # eval_hd.eval_show_indicator(["mse"], x_alias="Rule Number",y_log=True, xtick_rotation=0,text_round=4)
    # # eval_hd.eval_show_indicator(["trainable_params", "mse"], x_log=True)
    # eval_hd.eval_show_indicator(["fl_avg_pearson_non_diag", "mse"], x_log=True)
    # eval_hd.eval_show_indicator(["mf_avg_pearson_non_diag", "mse"], x_log=True)
    # eval_hd.eval_show_indicator(["mf_avg_pearson_non_diag", "fl_avg_pearson_non_diag"], x_log=True)
    # eval_hd.eval_show_indicator(["trainable_params", "eval_items_per_ms"], x_log=True)
