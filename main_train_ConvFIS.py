#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_main_train_separation_dimension_state.py

# @Time      :2024/1/8 17:51
# @Author    :Oliver
"""

[...,time_dim_in,state_dim] -> [...,time_dim_out,out_dim==state_dim]
"""
import re
import os
import torch
from collections import namedtuple
import torch.optim.lr_scheduler
from config import load_config
from model.fuzzy_inference import Layers
from frame import Trainer, EvalConvFisHandler


ModelConfig = namedtuple('Config', [
    'state_dim_in', 'state_dim_out', 'time_dim_in', 'time_dim_out', 'rule_num',
    'raw_branch_out_dim', 'pattern_num', 'dff_tnorm_min',"time_horizon",
    "norm_branch_in_dim", "norm_branch_out_dim",

])

class ConvFis(torch.nn.Module):
    """
    [..., time_dim_in, state_dim] -> [..., time_dim_out, state_dim]
    The inputs and outputs are required to be standard normalized data to ensure that the random normalization rules can be adapted.
    """
    def __init__(self,config:ModelConfig):
        super().__init__()
        state_dim_in = config.state_dim_in
        state_dim_out = config.state_dim_out
        time_dim_in = config.time_dim_in
        time_dim_out = config.time_dim_out
        rule_num = config.rule_num

        norm_branch_in_dim = config.norm_branch_in_dim
        raw_branch_out_dim = config.raw_branch_out_dim
        norm_branch_out_dim = config.norm_branch_out_dim

        pattern_num = config.pattern_num
        time_horizon = config.time_horizon
        dff_tnorm_min = config.dff_tnorm_min

        # self.time_dim_in = time_dim_in
        # self.raw_branch_out_dim = raw_branch_out_dim
        self.norm_branch_in_dim = norm_branch_in_dim
        self.norm_batch = norm_branch_out_dim > 0
        self.time_fuzzifier = Layers.TimeFuzzifierLayer(rule_num,state_dim_in,raw_branch_out_dim)
        if self.norm_batch:
            self.norm_fuzzifier = Layers.TimeFuzzifierLayer(rule_num,norm_branch_in_dim,norm_branch_out_dim)
        # self.time_inference = Layers.MultiTimeGaussianInferenceLayer(raw_branch_out_dim,rule_num)
        self.time_inference = Layers.TimeGaussianInferenceLayer(raw_branch_out_dim+norm_branch_out_dim,time_dim_in,
                                                                pattern_num, rule_num,
                                                                time_horizon, 5)
        self.defuzzifier = Layers.SingletonHeightDefuzzifierLayer(rule_num,[time_dim_out,state_dim_out], dff_tnorm_min)

        self.loss_model = torch.nn.MSELoss(reduction="none")

    def loss(self, pred, target):
        raw_loss = (torch.sum(self.loss_model(pred[..., 0:2], target[..., 0:2]), dim=-1)
                + torch.sum(self.loss_model(pred[..., 2:4], target[..., 2:4]) * 1e-5, dim=-1))
        loss = torch.sum(raw_loss, dim=-1)
        return loss

    def forward(self, samples,  labels=None, *args_, **kwargs):
        last_point = samples[...,[-1],:]
        x_in_fz,_ = self.time_fuzzifier(samples)
        if self.norm_batch:
            position = samples[...,:,:self.norm_branch_in_dim]
            # position = samples[...,:,:self.norm_branch_in_dim]
            mean = position.mean(dim=-2, keepdim=True)
            std = position.std(dim=-2, keepdim=True)
            norm_position = (position - mean)/std
            norm_in_fz,_ = self.norm_fuzzifier(norm_position)
            mf = self.time_inference(torch.concatenate([x_in_fz,norm_in_fz], dim=-1))
        else:
            mf = self.time_inference(x_in_fz)

        output = self.defuzzifier(x_in_fz, mf) + last_point

        rtn = {"output":output}
        if labels is not None:
            rtn["loss"] = self.loss(output, labels)
        if "mode" in kwargs and kwargs["mode"] == "test":
            # rtn["last_point"] = last_point
            rtn["membership_degree"] = mf                       # membership_degree of all rule (before prod11)
            rtn["phi"] = self.defuzzifier.calculate_phi(mf)     # rule weight

        return rtn

if __name__ == '__main__':
    from frame.data_process.data_transform import *
    from support_config_parser import ConfigManager
    middle_epoch = None
    # middle_epoch = (1, 2, 4, 8, 16, 32, 64, 128, 256, 500)
    # middle_epoch = list(range(1, 100)) + list(range(100, 500, 2))
    # 读取配置文件
    load_config()
    cfg = ConfigManager("default.ini").parser("model_ConvFis")
    args = cfg.train_args.set_custom_save(init=True,middle=middle_epoch,end=True,best=True)
    model_config = ModelConfig(
        state_dim_in=cfg.getint("state_dim"),
        state_dim_out=cfg.getint("state_dim_out"),
        time_dim_in=cfg.getint("seq_len"),
        time_dim_out=cfg.getint("pred_len"),
        rule_num=cfg.getint("rule_num"),
        pattern_num=8,
        time_horizon=15,
        dff_tnorm_min=False,
        norm_branch_in_dim=2,
        raw_branch_out_dim=9,
        norm_branch_out_dim=4,
    )

    # cfg.train_args.seed = 37  # 37, 23, 78, 15, 64, 9
    # cfg.train_args.seed = 1  # 37, 23, 78, 15, 64, 9

    # 训练准备
    model =ConvFis(model_config)
    trainer = Trainer(model, args, train_dataset=cfg.train_dataset, eval_dataset=cfg.valid_dataset)
    cfg.config_summary(trainer.reporter, model_config._asdict())
    trainer.train()

    # ********* 测试绘图相关 *****
    # 重新加载(并非必要 但为了用上best_valid还是加载一下）
    # args = cfg.train_args
    args.set_evaluation(batch_size=256)
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
    # eval_hd = EvalConvFisHandler(cfg.root_reporter, plot_mode="academic_mode")
    with EvalConvFisHandler(cfg.root_reporter, plot_mode="academic_mode") as eval_hd:
        with eval_hd.manager_raw_data() as raw_hd:
            raw_hd.update_data(model_original_)

            eval_hd.raw_eval_willow(raw_tracks, sample_rule=20, used_tracks=25)
            eval_hd.raw_eval_predict_frames(48, pic_raw=3, pic_column=4)
            eval_hd.raw_eval_antecedent_category_of_track()
            for k, v in model_original_.items():
            #     eval_hd.add_data(k, v)
                eval_hd.register_count_paras(k, trainer.model)
            eval_hd.register_error()
        # other
        # eval_hd.eval_count_paras({args.save_dir_prefix: trainer.model})
        eval_hd.summary_count_paras()

        eval_hd.summary_metric()
        eval_hd.eval_compare_metrics_bar("eval", box_like=False)
        eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)

        # eval_hd.eval_metrics_by_time(columns=columns, unit_s=cfg.getint("x_unit_s"), draw_confidence_interval=False)

        eval_hd.eval_bar_rule_frequency()
        eval_hd.eval_bar_rule_frequency(sort=False)
        eval_hd.eval_defuzzifier()

        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=False)
        eval_hd.eval_track_display(raw_tracks, flag_split=False, mix=True, heat=True, heat_bins=200)
        # eval_hd.eval_antecedent_category_of_track_alpha(show_category=cfg.getint("rule_num"), show_data_limit=300)
        # try:
        #     eval_hd.eval_height_defuzzifier(trainer.model.defuzzifier, None)
        # except Exception as e:
        #     print("calculate defuzzifier failed.. caught exception: \n{}".format(e))

        # eval_hd.get_save_dir()



