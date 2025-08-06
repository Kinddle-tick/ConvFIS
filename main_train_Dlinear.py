#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_main_train_separation_dimension_state.py

# @Time      :2024/1/8 17:51
# @Author    :Oliver
"""

[...,in_time_dim,state_dim] -> [...,out_time_dim,out_dim==state_dim]
"""
import re

import pandas as pd
import sys
import torch.optim.lr_scheduler
from torchmetrics import Accuracy, F1Score
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score


from config import *
from frame.eval_process import EvalTrackHandler
from frame.trainer import Trainer
from frame.training_args import TrainingArguments
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from collections import namedtuple
from frame import Trainer

from model import DLinear 
from model.timeseries_model.warpper import TFmodel_warped

if __name__ == '__main__':
    from frame.data_process.data_reader import DataReaderDataGroup
    # from datasets import load_dataset, Dataset
    from frame.data_process.data_dataset import SlideWindowDataset
    from torch.utils.data import DataLoader, Dataset, ConcatDataset
    from frame.data_process.data_transform import *
    # from torchvision import datasets, transforms
    from support_config_parser import ConfigManager
    # 读取配置文件
    load_config()
    cfg = ConfigManager("default.ini", ).parser("model_DLinear")

    ModelConfig = namedtuple('Config', [
        'seq_len', 'pred_len', 'label_len', 'output_attention', 'use_norm', 'd_model',  
        'embed', 'freq', 'dropout', 'class_strategy', 'factor', 'n_heads',  
        'e_layers', 'd_ff', 'activation', 'channel_independence', 'c_out', 'dec_in', 'enc_in', 'd_layers', 'distil', 'task_name', 'moving_avg'
    ])  

    # 实例化配置对象，使用命令行作为默认值。  
    configs = ModelConfig(
        seq_len=cfg.getint("seq_len"),  #输入长度
        pred_len=cfg.getint("pred_len"), #输出长度
        label_len = 40, #  Informer配置
        output_attention=False,  # 假定一个默认值，如果脚本中没有提供这个的话  
        use_norm=False,  # 假定一个默认值，如果脚本中没有提供这个的话  
        d_model=512,  
        embed='linear',  # 假定一个可能的默认值  
        freq='h',  # 假定一个可能的默认值  
        dropout=0.1,  # 假定一个可能的默认值  
        class_strategy='simple',  # 假定一个可能的默认值  
        factor=5,  # 假定一个可能的默认值  
        n_heads=8,  # 假定一个可能的默认值  
        e_layers=6,
        d_layers=6,  # Transformer配置
        d_ff=512,  
        activation='relu',  # 假定一个可能的默认值
        channel_independence = False,  # Transformer配置
        enc_in = 4, # Transformer配置
        dec_in = 4, # Transformer配置
        c_out = 4, # Transformer配置
        distil = False, #Informer配置
        task_name= "long_term_forecast",
        moving_avg = 1
    )

    model = DLinear(configs)
    model.seq_len = configs.seq_len
    model = TFmodel_warped(model)

    # Training setup  
    optimizer = model.configure_optimizers()  

    # 可选：设置学习率调度器
    batch_size = cfg.getint("batch_size")
    num_epochs = cfg.getint("epoch_limit")
    learning_rate = cfg.getfloat("init_learning_rate")
    num_training_steps = num_epochs * len(cfg.train_dataset)//batch_size
#     optimizer = torch.optim.AdamW(model.parameters(),learning_rate,weight_decay=0.1,betas=(0.9,0.95))
#     lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
#         optimizer=optimizer,
#         num_warmup_steps=60,            # 预热步数
#         num_cycles= 10,
#         num_training_steps=num_training_steps
# )

    # # 定义训练参数
    # args = (TrainingArguments(sys.argv, valid_interval=valid_interval, save_dir_prefix=save_dir_prefix).declare_regression(metric_method_func)
    #         .set_training(learning_rate=learning_rate, batch_size=batch_size, epoch_limit=num_epochs))
    # args.model_with_loss = model_with_loss

    trainer = Trainer(model, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
    cfg.config_summary(trainer.reporter, configs._asdict())
    trainer.train()

    # ********* 测试绘图相关 *****
    # 重新加载
    args = cfg.train_args
    args.set_evaluation(batch_size=batch_size//16)
    # model_load_path = os.path.join(trainer.args.model_save_dir, "best_train")
    model_load_path = os.path.join(trainer.args.model_save_dir, "best_valid")
    new_trainer = Trainer(model_load_path, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)

    # 接续训练
    # args.set_training(epoch_limit=200)
    # trainer.train()

    model_output,_ = new_trainer.predict()

    # original_func = SlideWindowDataset.get_original_eval_prediction(cfg.data_processed)
    model_original_ = {k: cfg.data_fn_inv_eval_prediction(v) for k, v in model_output.items()}
    columns = cfg.data_processed.columns
    raw_tracks = [track[columns].to_numpy() for track in cfg.data_original]

    with EvalTrackHandler(cfg.root_reporter, plot_mode="academic_mode") as eval_hd:
        with eval_hd.manager_raw_data() as raw_hd:
            raw_hd.update_data(model_original_)

            eval_hd.raw_eval_willow(raw_tracks, sample_rule=20, used_tracks=25)
            eval_hd.raw_eval_predict_frames(48, pic_raw=3, pic_column=4)
            for k, v in model_original_.items():
                #     eval_hd.add_data(k, v)
                eval_hd.register_count_paras(k, trainer.model)
            eval_hd.register_error()


        # other
        # eval_hd.eval_count_paras({args.save_dir_prefix:trainer.model})
        eval_hd.summary_count_paras()

        eval_hd.summary_metric()
        eval_hd.eval_compare_metrics_bar("eval", box_like=False)
        eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)

        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=False)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True, heat=True, heat_bins=200)


