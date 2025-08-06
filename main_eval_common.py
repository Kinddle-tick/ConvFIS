#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/5 16:34
# @Author  : oliver
# @File    : main_eval_common.py
# @Software: PyCharm
# from main_train_fis_conv_dim import *
import re
import sys
import numpy as np
from frame.eval_process.eval_analyze import EvalConvFisHandler
from frame.trainer import Trainer
from frame.data_process.data_dataset import SlideWindowDataset
from support_config_parser import ConfigManager
from frame import Trainer, EvalConvFisHandler
from config import *

data_sec_name = "dataset_quin33_6s"
model_save_dir = "output/runs_log/20250425_13-50-47_Resnet18_quin33_6s_Linux_train"
model_section_name = "model_Resnet18"
# save_dir_prefix = "TrackPred_Fuzzy"



if __name__ == '__main__':
    load_config()
    cfg = ConfigManager("default.ini", data_sec_name).parser(model_section_name, mode="test")
    # eval_hd = EvalConvFisHandler(cfg.root_reporter,plot_mode="academic_mode")
    with EvalConvFisHandler(cfg.root_reporter,plot_mode="academic_mode") as eval_hd:
        args = cfg.train_args

        model_load_path = os.path.join(model_save_dir, "models", "best_valid")
        trainer = Trainer(model_load_path, args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset,
                          )
        # trainer.load_model(load_path, args.model_device)
        cfg.config_summary(trainer.reporter)

        model_output,_ = trainer.predict()

        # original_func = SlideWindowDataset.get_original_eval_prediction(cfg.data_processed)
        model_original_ep = {k:cfg.data_fn_inv_eval_prediction(v) for k,v in model_output.items()}
        columns = cfg.data_processed.columns
        raw_tracks = [track[columns].to_numpy() for track in cfg.data_original]
        # eval_hd = EvalConvFisHandler(trainer.reporter,plot_mode="academic_mode")
        for k, v in model_original_ep.items():
            eval_hd.add_data(k, v)
            eval_hd.register_count_paras(k, trainer.model)

        # other
        # eval_hd.eval_count_paras({args.save_dir_prefix: trainer.model})
        eval_hd.summary_count_paras()
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=False)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True, heat=True, heat_bins=200)
        eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)
        eval_hd.raw_eval_willow(raw_tracks, sample_rule=20, used_tracks=25)
        eval_hd.raw_eval_predict_frames(48, pic_raw=3, pic_column=4)

    # eval_hd.get_save_dir()





















