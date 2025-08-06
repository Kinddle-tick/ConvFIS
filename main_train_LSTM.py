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
import torch
import torch.optim.lr_scheduler
from torchmetrics import Accuracy, F1Score
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score


from config import *
from frame.eval_process import EvalTrackHandler
from frame.trainer import Trainer
from frame.training_args import TrainingArguments
from model.fuzzy_inference import Layers
# from frame import AutoStandardScaler

# from main_compare import track_path, track_slice
# from models.FuzzyInferenceSystem import Layers
# import Frox
# from Frox.DataProcessing import *
from frame import Trainer, EvalConvFisHandler

ModelConfig = namedtuple('Config', [
    'state_dim', 'in_time_dim', 'out_time_dim', 'num_layers',
    'hidden_state_dim',
])

class LSTMTrack(torch.nn.Module):
    """
    [...,in_time_dim, state_dim] -> [...,out_time_dim, state_dim]

    """
    def __init__(self,config:ModelConfig):
        super().__init__()
        # self.state_dim = state_dim
        state_dim = config.state_dim
        in_time_dim = config.in_time_dim
        out_time_dim = config.out_time_dim
        num_layers = config.num_layers
        hidden_state_dim = config.hidden_state_dim

        self.in_time_dim = in_time_dim
        self.out_time_dim = out_time_dim
        self.hidden_state_dim = hidden_state_dim
        # self.hidden_state_dim = hidden_state_dim
        self.encoder = torch.nn.LSTM(state_dim, hidden_state_dim, num_layers, batch_first=True, bidirectional=False)
        self.decoder = torch.nn.LSTM(hidden_state_dim, hidden_state_dim, num_layers, batch_first=True, bidirectional=False)
        self.projection = torch.nn.Linear(hidden_state_dim, state_dim)
        # self.shell = AutoStandardScaler(None, scale_axis=-2)
        # self.time_fuzzifier = Layers.TimeFuzzifierLayer(rule_num,state_dim_in,hidden_state_dim)
        # # self.time_inference = Layers.MultiTimeGaussianInferenceLayer(hidden_state_dim,rule_num)
        # self.time_inference = Layers.TimeGaussianInferenceLayer(hidden_state_dim,in_time_dim, pattern_num, rule_num,
        #                                                         15, 5)
        # self.defuzzifier = Layers.SingletonHeightDefuzzifierLayer(rule_num,[out_time_dim,state_dim_out], dff_tnorm_min)

        self.loss_model = torch.nn.MSELoss(reduction="none")

    def loss(self, pred, target):
        raw_loss = (torch.sum(self.loss_model(pred[..., 0:2], target[..., 0:2]), dim=-1)
                + torch.sum(self.loss_model(pred[..., 2:4], target[..., 2:4]) * 1e-5, dim=-1))
        loss = torch.sum(raw_loss, dim=-1)
        return loss

    def forward(self, samples,  labels=None, *args_, flg_train=False, flg_valid=False, **kwargs):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(samples)
        # hidden_rtn = torch.empty([samples.shape[0], self.out_time_dim, self.hidden_state_dim], device=samples.device)
        hidden_rtn = torch.zeros([samples.shape[0], self.out_time_dim, self.hidden_state_dim], device=samples.device)
        decoder_input = encoder_output[:,[-1]] # encoder_output[:,-1]== encoder_hidden[-1]
        # decoder_input = decoder_input.unsqueeze(dim=-1)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        hidden_rtn[:,0] = encoder_output[:,-1]
        # decoder_output = [decoder_input.copy()]
        # hidden_rtn[0] = decoder_input
        for i in range(1, self.out_time_dim):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden,decoder_cell))

            hidden_rtn[:, i] = decoder_output[:,-1]
            decoder_input = decoder_output

        target = self.projection(hidden_rtn)
        if labels is None:
            return target
        else:
            return target, self.loss(target,labels)



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
    cfg = ConfigManager("default.ini",).parser( "model_LSTM")

    model_config = ModelConfig(
        state_dim=cfg.getint("state_dim"),
        in_time_dim=cfg.getint("seq_len"),
        out_time_dim=cfg.getint("pred_len"),
        num_layers=1,
        hidden_state_dim=9,
    )
    model = LSTMTrack(model_config)

    trainer = Trainer(model, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
    cfg.config_summary(trainer.reporter, model_config._asdict())
    trainer.train()

    # ********* 测试绘图相关 *****
    # 重新加载
    args = cfg.train_args
    # model_load_path = os.path.join(trainer.args.model_save_dir, "best_train")
    model_load_path = os.path.join(trainer.args.model_save_dir, "best_valid")
    new_trainer = Trainer(model_load_path, cfg.train_args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
    # trainer.reporter(f"使用轨迹数据位置：{track_path}").md().log()

    # 接续训练
    # args.set_training(epoch_limit=200)
    # trainer.train()


    args.set_evaluation(batch_size=args.batch_size//16)
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


