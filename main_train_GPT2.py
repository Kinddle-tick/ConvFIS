#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_main_train_separation_dimension_state.py

# @Time      :2024/1/8 17:51
# @Author    :Oliver
"""

[...,in_time_dim,state_dim] -> [...,out_time_dim,out_dim==state_dim]
"""
import re
from collections import namedtuple

import pandas as pd
import sys
import torch.optim.lr_scheduler
from torchmetrics import Accuracy, F1Score
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score


from config import *
from frame.eval_process import EvalTrackHandler
from frame.trainer import Trainer
from frame.training_args import TrainingArguments
# from frame import AutoStandardScaler

from model.gpt_model import GPTTrack
from transformers import get_scheduler
from frame import Trainer, EvalConvFisHandler

ModelConfig = namedtuple('Config', [
    'state_dim_in', 'state_dim_out', 'in_time_dim', 'out_time_dim', 'num_layers',
    'n_head', 'n_embd', 'dropout', "device"
])

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
    cfg = ConfigManager("default.ini", ).parser("model_GPT2")

    model_config = ModelConfig(
        state_dim_in = cfg.getint("state_dim"),
        state_dim_out = cfg.getint("state_dim_out"),
        in_time_dim = cfg.getint("seq_len"),
        out_time_dim = cfg.getint("pred_len"),
        num_layers = 6,
        n_head = 8,
        n_embd = 128,
        dropout = 0.1,
        device = cfg.train_args.model_device,
    )

    model = GPTTrack(**model_config._asdict())
    
    # 可选：设置学习率调度器
    batch_size = cfg.getint("batch_size")
    num_epochs = cfg.getint("epoch_limit")
    learning_rate = cfg.getfloat("init_learning_rate")
    num_training_steps = num_epochs * len(cfg.train_dataset)//batch_size
    optimizer = model.configure_optimizers(lr=learning_rate, weight_decay=0.1,betas=(0.9,0.95)) 
    lr_scheduler = get_scheduler(  
        'cosine_with_restarts',                      # 可选 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'  
        optimizer=optimizer,  
        num_warmup_steps=60,            # 预热步数  
        num_training_steps=num_training_steps  
)

    #
    # # 定义训练参数
    # args = (TrainingArguments(sys.argv, valid_interval=valid_interval, save_dir_prefix=save_dir_prefix)
    #         .declare_regression(metric_method_func)
    #         .set_training(learning_rate=learning_rate, batch_size=batch_size, epoch_limit=num_epochs))
    # args.model_with_loss = model_with_loss

    
    
    # trainer = Trainer(model,
    #                   args, train_dataset=train_set, eval_dataset=valid_set, data_collator=collate_fn,optimizers=(optimizer,lr_scheduler))
    # trainer.reporter("[+]general_section of used config:").md().log()
    # for k,v in general_section.items():
    #     trainer.reporter(f"\t{k}:{v}").md().log()
    # trainer.reporter("[+]model_section of used config:").md().log()
    # for k,v in model_section.items():
    #     trainer.reporter(f"\t{k}:{v}").md().log()
    # trainer.train()

    # 训练准备
    # model =ConvFis(model_config)
    trainer = Trainer(model, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
    cfg.config_summary(trainer.reporter, model_config._asdict())
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



