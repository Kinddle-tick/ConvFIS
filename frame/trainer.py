#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/26 14:33
# @Author  : oliver
# @File    : trainer.py
# @Software: PyCharm
import os
import re
import time
import warnings
import pickle
from email.policy import default
from sched import scheduler

import numpy as np
from accelerate import Accelerator
from dataclasses import dataclass, field, Field
import socket
from datetime import datetime
import torch
import json
import random
import dill

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum

from huggingface_hub.hf_api import ModelInfo
from pyarrow.dataset import dataset
from torch.utils.data import Dataset, DataLoader
from torch.xpu import device
from tqdm.auto import tqdm

from .util import get_str_time, StateEnum, EvalPrediction
from .reporter import Reporter
from .training_args import TrainingArguments
from .trainer_callback import (
    has_length,
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    TrainerCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    TrainerState,
    TrainerControl,
    CustomCallback,
)
from .util.generic import ModelOutput, EpochSummary, MetricCalculate
from .util.util_obj import EpochProperty

# from .eval_process import EvalHandler

DEFAULT_CALLBACKS = [DefaultFlowCallback, CustomCallback]

# class MetricCalculate:
#     def __init__(self):
#         self.metrics = {}
#
#     def add_metric(self, name, metric):
#         if name not in self.metrics:
#             self.metrics[name] = metric
#         else:
#             warnings.warn("There is already a metric method [{}] in Class, Override.".format(name))
#             self.metrics[name] = metric
#         return self
#
#     def __call__(self,eval_predicts, *args, **kwargs)-> Dict[str, float]:
#         rtn = dict()
#         for name, metric in self.metrics.items():
#             rtn[name] = metric(*eval_predicts, *args, **kwargs)
#         return rtn
#
#     def __repr__(self):
#         rtn = "MetricCalculate:\n"
#         for name, metric in self.metrics.items():
#             rtn += f"\t{name}: {metric}\n"
#
#     @property
#     def names(self):
#         return self.metrics.keys()

class Trainer:

    def __init__(
            self,
            model: torch.nn.Module or str= None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            compute_loss_func: Optional[Callable or torch.nn.modules.loss] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[Callable]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),

    ):
        self.model = None
        self.model_info = {}
        self.loaded_scheduler_state_dict = None
        self.loaded_optimizer_state_dict = None

        self.args = args
        self.reporter= self.init_reporter()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.data_collator = data_collator if data_collator is not None else self.args.data_collator
        self.compute_loss_func = compute_loss_func if compute_loss_func is not None else self.args.train_compute_loss_func
        self.compute_metrics_func = compute_metrics if compute_metrics is not None else self.args.train_compute_metrics_func

        self.optimizer, self.lr_scheduler = optimizers

        self.accelerator = Accelerator()        # 和多显卡有关 目前未启用
        self.trainer_state = TrainerState()     # 训练器状态跟踪
        self.trainer_ctrl = TrainerControl(self.reporter)   # 训练跟踪器

        default_callbacks = DEFAULT_CALLBACKS
        self.callbacks = default_callbacks if callbacks is None else  default_callbacks + callbacks
        self.callback_handler = self.check_callback_handler()       # 回调相关处理

        self.init_model(model)
        self.best_loss = None
        self.save_state_dict = None
        self.grad_scaler:torch.amp.GradScaler = torch.amp.GradScaler(enabled=self.args.train_autocast)
        self.callback_handler.on_init_end(self.args, self.trainer_state, self.trainer_ctrl)

    def __del__(self):
        self.reporter.close()

    def init_reporter(self):
        rtn_reporter = self.args.reporter.band(self.__class__.__name__, self.args.flag_split_log)
        return rtn_reporter

    def check_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(opt_model.parameters(), lr=self.args.learning_rate)
            self.reporter(f"using default optimizer:{self.optimizer.__class__}").log().md()
        else:
            self.reporter(f"using customized optimizer:{self.optimizer.__class__}").log().md()
        if self.loaded_optimizer_state_dict is not None:
            try:
                self.optimizer.load_state_dict(self.loaded_optimizer_state_dict)
            except (RuntimeError, ValueError) as e:
                self.reporter(f"optimizer loaded state dict failed: {e}")
        return self.optimizer

    def check_lr_scheduler(self, optimizer=None):
        opt_optimizer = optimizer if optimizer is not None else self.optimizer
        if self.optimizer is None:
            self.check_optimizer()
            opt_optimizer = self.optimizer
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt_optimizer, mode="min", factor=0.5, patience=10, verbose=True, cooldown=5)
            self.reporter(f"using default lr_scheduler:{self.lr_scheduler.__class__}").log().md()
        else:
            self.reporter(f"using customized lr_scheduler:{self.lr_scheduler.__class__}").log().md()
        if self.loaded_scheduler_state_dict is not None:
            try:
                self.lr_scheduler.load_state_dict(self.loaded_scheduler_state_dict)
            except (RuntimeError, ValueError) as e:
                self.reporter(f"scheduler loaded state dict failed:{e}")
        return self.lr_scheduler

    def check_callback_handler(self):
        self.callback_handler = CallbackHandler(self.callbacks, self.model, None, self.optimizer,
                                                self.lr_scheduler)
        return self.callback_handler

    def get_loader(self, dataset_, shuffle=True):
        if dataset_ is None:
            raise ValueError("dataset is None")
        dataloader = DataLoader(dataset_, batch_size=self.args.batch_size, shuffle=shuffle,
                                num_workers=self.args.num_workers, pin_memory= self.args.pin_memory,
                                collate_fn=self.data_collator)
        return dataloader

    def train(self):
        self.trainer_state.state_train = StateEnum.INITIALIZING
        if self.args.epoch_limit is not None:
            epoch_limit = self.args.epoch_limit
        elif self.args.step_limit is not None and self.args.batch_size is not None:
            epoch_limit = self.args.step_limit // self.args.batch_size
        else:
            raise ValueError("Either epoch_limit or (step_limit & batch_size) must be specified")
        self.args.epoch_limit = epoch_limit

        self.model = self.model.to(device=self.args.model_device)
        self.save_state_dict = None
        self.trainer_state.best_loss = None
        train_dataloader = self.get_loader(self.train_dataset, shuffle=True)
        valid_dataloader = self.get_loader(self.eval_dataset, shuffle=False)
        self.check_optimizer()
        self.check_lr_scheduler()
        self.check_callback_handler()
        self.grad_scaler = torch.amp.GradScaler(enabled=self.args.train_autocast)

        # region 训练信息
        self.trainer_state.max_epochs = epoch_limit
        self.trainer_state.valid_interval = self.args.valid_interval
        self.trainer_state.batch_size = self.args.batch_size
        if has_length(train_dataloader.dataset):
            self.trainer_state.train_sample_num = len(train_dataloader.dataset)
        if has_length(valid_dataloader.dataset):
            self.trainer_state.valid_sample_num = len(valid_dataloader.dataset)
        self.trainer_state.flag_autocast = self.args.train_autocast
        self.trainer_state.train_lr = self._get_lr_now()
        if self.args.model_with_loss:
            self.trainer_state.loss_func_desc = 'Def In Model'
        else:
            if self.compute_loss_func is None:
                if self.args.do_regression:
                    self.compute_loss_func = lambda x,y : torch.nn.MSELoss(reduction='mean')(x,y)
                if self.args.do_classification:
                    self.compute_loss_func = lambda x,y: torch.nn.modules.loss.CrossEntropyLoss()(x,y.long())
            self.trainer_state.loss_func_desc = self.compute_loss_func.__class__.__name__
        # endregion

        self.callback_handler.on_train_begin(self.args, self.trainer_state, self.trainer_ctrl)
        self.try_save()
        self.trainer_state.epoch = self.args.epoch_current
        self.trainer_state.max_steps = self.args.step_limit if self.args.step_limit is not None else -1

        self.trainer_state.state_train = StateEnum.ACTIVE

        for epoch_count in range(self.trainer_state.epoch, epoch_limit):
            self.epoch(self.model, train_dataloader,{}, skip_backward=False)

            # if (self.trainer_state.epoch + 1) % self.args.valid_interval == 0:
            if self.trainer_ctrl.should_valid:
                self.trainer_state.epoch_property = EpochProperty.VALID
                self.epoch(self.model, valid_dataloader,{}, skip_backward=True)
                self.trainer_state.epoch_property = EpochProperty.TRAIN

            self.trainer_state.epoch = epoch_count + 1
            if self.trainer_ctrl.should_training_stop:
                break

        self.trainer_state.state_train = StateEnum.FINISHED
        self.callback_handler.on_train_end(self.args, self.trainer_state, self.trainer_ctrl)
        self.try_save()

    def predict(self, datasets: List or Dict = None):
        if datasets is None:
            datasets = {"train":self.train_dataset, "eval":self.eval_dataset}
        elif isinstance(datasets, List):
            datasets = {f"dataset_{i}":data for i, data in enumerate(datasets)}
        elif isinstance(datasets, Dict):
            datasets = datasets
        else:
            datasets = {"given_dataset":self.train_dataset}

        prediction_info = self.model_info.copy()

        model = self.model.to(device=self.args.model_device)
        # if self.args.save_dir_prefix is not None:
        #     self.args.rename_reporter_folder_name(self.args.gene_save_dir(self.args.save_dir_prefix))

        self.trainer_state.epoch_property = EpochProperty.TEST
        self.callback_handler.on_predict_begin(self.args, self.trainer_state, self.trainer_ctrl)
        eval_return = {}
        for name, dataset_ in datasets.items():
            self.trainer_state.eval_current_dataset = name
            loader = self.get_loader(dataset_, shuffle=False)
            eval_prediction = self.epoch(model,loader, {}, skip_backward=True, skip_loss=True)

            self.callback_handler.on_evaluate(self.args,self.trainer_state, self.trainer_ctrl, None)
            eval_return.update({name: eval_prediction})
        self.callback_handler.on_predict_end(self.args, self.trainer_state, self.trainer_ctrl)
        prediction_info.update({"eval_time_per_item_ms":self.trainer_state.eval_time_per_item_ms,
                                "eval_items_per_ms":1/self.trainer_state.eval_time_per_item_ms})
        return eval_return, prediction_info

    def epoch(self, model, data_loader, md_kwargs, *, skip_backward = False, skip_loss=False)->EpochSummary or None:
        self.trainer_state.state_epoch = StateEnum.INITIALIZING
        total_loss = 0.
        dataset_len = len(data_loader.dataset)
        data_loader_len = len(data_loader)
        self.trainer_state.epoch_avg_loss_train = self.trainer_state.epoch_avg_loss_valid = None
        md_kwargs["mode"] = self.trainer_state.epoch_property.value
        # metric_use_data_list = []
        model_output_list = []
        model_input_list = []
        nan_number = 0
        model_inference_time_ms = 0


        self.trainer_state.state_epoch = StateEnum.ACTIVE
        self.callback_handler.on_epoch_begin(self.args, self.trainer_state, self.trainer_ctrl)
        tqdm_desc = "epoch-{:<3} {:<7}".format(self.trainer_state.epoch+1,f"[{self.trainer_state.epoch_property.value}]")

        with tqdm(total=dataset_len, unit="it", desc=tqdm_desc,
                  leave=True, file=self.reporter.terminal_ctrl.tqdm_stream_) as pbar:
            self.reporter.terminal_ctrl.driving_pbar = pbar
            pbar_update_len = 0
            for i,(sample, target, *args) in enumerate(data_loader,1):
                self.trainer_state.step_count = i
                data_len = sample.size(0)
                model_output = self.step(model, sample, target, args, md_kwargs,
                                                        skip_backward=skip_backward, skip_loss=skip_loss)

                if self.trainer_ctrl.flag_step_nan:
                    nan_number += data_len
                    # self.trainer_ctrl.flag_step_nan=False
                    loss = model_output['loss']
                else:
                    loss = model_output['loss']
                    total_loss += np.sum(loss)
                    model_output_list.append(model_output)
                    model_input_list.append([sample, target, args])

                pbar_update_len += data_len
                if self.args.stdout_enable_:
                    pbar.update(pbar_update_len)
                    pbar_update_len = 0
                    pbar.set_postfix_str(f"step[{i}/{data_loader_len}]" +
                                         f", step_avg_loss:{loss.mean()}" if not skip_loss else "")

                model_inference_time_ms += self.trainer_state.step_cost_time_ms
                if self.trainer_ctrl.should_epoch_stop:
                    break
            pbar.set_postfix_str(f"step[{i}/{data_loader_len}], pure inference time per item:{model_inference_time_ms/dataset_len}ms")

        if nan_number >= dataset_len:
            # 临时的跳过逻辑 此时完全没有非nan数据，无法计算final_loss_num以及其他
            return None

        # update the state
        final_loss_num = total_loss / (dataset_len - nan_number)
        self.trainer_state.epoch_avg_loss = final_loss_num
        self.trainer_state.eval_time_per_item_ms = model_inference_time_ms / dataset_len
        self.trainer_state.state_epoch = StateEnum.FINISHED

        # update and get the learning rate
        if self.trainer_state.epoch_property == EpochProperty.TRAIN:
            self._update_optimizer(final_loss_num, self.trainer_state.epoch - 1)
            self.trainer_state.train_lr = self._get_lr_now()
        else:
            self.trainer_state.train_lr = None

        # get the EpochSummary Output
        if len(model_output_list) != 0:
            output_key = model_output_list[0].keys()
            output_value = zip(*[model_output.values() for model_output in model_output_list])
            model_all_output = ModelOutput(dict(zip(output_key, output_value)))
            rtn = EpochSummary(model_all_output, *zip(*model_input_list))
        else:
            rtn = None

        # calculate the metric (if it has to)
        if self.trainer_ctrl.should_calculate_metric:
            # self.trainer_state.epoch_metric_data = self.compute_metrics_func(rtn)
            self.trainer_state.epoch_metric_data = self.compute_metrics(rtn)
        self.callback_handler.on_epoch_end(self.args, self.trainer_state, self.trainer_ctrl)
        self.try_save()
        return rtn

    def step(self,model, sample, target, md_args, md_kwargs, *, skip_backward = False, skip_loss=False):
        """
        对一个batch的变量进行基于model的推理工作（model inference）
        :param model: 使用的模型，一般在外定义为self.model
        :param sample:输入到模型的内容
        :param target:目标输出
        :param md_args:输入到模型的“可变位置参数”，通常在数据集阶段定义
        :param md_kwargs:输入到模型的“关键词参数”，通常最小以epoch为单位改变
        :param skip_backward:是否跳过loss的反向传播过程 是需要指定的默认关键词参数
        :param skip_loss: 是否跳过计算loss的过程
        :return:返回loss和模型输出ModelOutput
        """
        self.trainer_state.state_step = StateEnum.INITIALIZING
        # self.reporter.check_input()
        self.callback_handler.on_step_begin(self.args, self.trainer_state, self.trainer_ctrl)

        sample = sample.to(dtype=next(model.parameters()).dtype).to(self.args.model_device).detach()
        target = target.to(dtype=next(model.parameters()).dtype).to(self.args.model_device).detach()
        self.trainer_state.state_step = StateEnum.ACTIVE

        model_output:ModelOutput = self.drive_model(model, sample, target, *md_args, **md_kwargs)
        detached_model_output = {k:v.detach().cpu().numpy() for k,v in model_output.items() if v is not None}

        if not self.trainer_state.skip_loss:
            outputs = model_output["output"]
            loss = self.compute_loss(model_output, target)
            detached_model_output["loss"] = loss.detach().cpu().numpy()
            if torch.isnan(loss).any():
                self._nan_process(sample, outputs, target)
            if not self.trainer_ctrl.flag_step_nan:
                loss_sum = torch.mean(loss, dim=0)
                self.trainer_state.step_avg_loss = loss_sum
                if not self.trainer_state.skip_backward:
                    self.optimizer.zero_grad()
                    self.grad_scaler.scale(loss_sum).backward()
                    self.grad_scaler.step(optimizer=self.optimizer)
                    self.grad_scaler.update()
            else:
                self.trainer_state.step_avg_loss = np.nan

        self.trainer_state.state_epoch = StateEnum.FINISHED
        self.callback_handler.on_step_end(self.args, self.trainer_state, self.trainer_ctrl)
        if self.trainer_state.epoch_property == EpochProperty.TEST:
            self.callback_handler.on_prediction_step(self.args, self.trainer_state, self.trainer_ctrl)
        return detached_model_output

    def drive_model(self, model, *md_args, **md_kwargs)->ModelOutput:
        """
        :return: ModelOutput 其第一项必定为output， 如果可能第二项一定是loss 其余项按照字母顺序排序
        """
        if self.args.train_autocast:
            with torch.autocast(device_type=self.args.model_device.type):
                start_time = time.time_ns()
                model_outputs = model(*md_args, **md_kwargs)
                end_time = time.time_ns()
        else:
            start_time = time.time_ns()
            model_outputs = model(*md_args, **md_kwargs)
            end_time = time.time_ns()
        self.trainer_state.step_cost_time_ms = (end_time - start_time) / 1e6

        if isinstance(model_outputs, ModelOutput):
            rtn_model_output = model_outputs
        elif isinstance(model_outputs, dict):
            rtn_model_output = ModelOutput(model_outputs)
        elif isinstance(model_outputs, tuple) or isinstance(model_outputs, list):
            if self.args.model_with_loss:
                rtn_model_output = ModelOutput({"output": model_outputs[0], "loss": model_outputs[1],
                                    **{f"args{i}":model_outputs[i+2] for i in range(len(model_outputs) - 2)}})
            else:
                rtn_model_output = ModelOutput({"output": model_outputs[0],
                                    **{f"args{i}":model_outputs[i+1] for i in range(len(model_outputs) - 1)}})
        else:
            rtn_model_output = ModelOutput({"output": model_outputs,})

        rtn_model_output = ModelOutput(sorted(rtn_model_output.items(), key=lambda t: t[0]))
        if "loss" in rtn_model_output:
            rtn_model_output.move_to_end("loss", last=False)
        else:
            rtn_model_output["loss"] = None
            rtn_model_output.move_to_end("loss", last=False)
        rtn_model_output.move_to_end("output", last=False)
        return rtn_model_output

    def compute_loss(self, model_output:ModelOutput, target):
        if "loss" not in model_output or model_output["loss"] is None:
            if self.args.model_with_loss:
                raise ValueError(
                    "args.model_with_loss is True, But the model did NOT return a loss from the model's outputs, only the following keys: "
                    f"{','.join(model_output.keys())}. Try to custom the models output with loss"
                    # f"For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            else:
                loss = self.compute_loss_func(model_output["output"], target)
        else:
            loss = model_output["loss"]
        return loss

    def compute_metrics(self, rtn:EpochSummary):
        return self.compute_metrics_func(rtn)
        # self.trainer_state.epoch_metric_data = {"default":"building..."}

    def try_save(self):
        if not self.trainer_ctrl.should_save:
            return
        else:
            self.trainer_ctrl.should_save = False
        self.save_model(self.args.model_save_dir, self.trainer_state.save_name,self.model.state_dict(),
                        self.optimizer, self.lr_scheduler)
        self.callback_handler.on_save(self.args, self.trainer_state, self.trainer_ctrl)

    def init_model(self, model: torch.nn.Module or str):
        if isinstance(model, str):
            try:
                self.load_model(model, self.args.model_device)
                self.reporter(f"[+] load model at {model}").log()
            except FileNotFoundError as e:
                self.reporter(f"[+] load fail: {e}").log().md()
                return None
        elif isinstance(model, torch.nn.Module):
            self.model = model
            # 计算总参数量
            total_params = sum(p.numel() for p in model.parameters())
            # 计算可训练的参数量
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.model_info = {"total_params": total_params, "trainable_params": trainable_params}
        else:
            return None


    def save_model(self, save_dir, comment="model", state_dict=None, optimizer_=None, scheduler_=None):
        model = self.model
        save_dir = os.path.join(save_dir, comment)
        json_path = os.path.join(save_dir, "config.json")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def _torch_save( obj, dir_: str, filename, pickle_module: any = None):
            path = os.path.join(dir_, filename)
            rel_path = os.path.relpath(path, save_dir)
            torch.save(obj, path, pickle_module=pickle if pickle_module is None else pickle_module)
            return rel_path

        json_dict = {"model_name": model.__class__.__name__, "last_epoch": self.trainer_state.epoch,
                     # "proj_dir": os.path.relpath(self.args.reporter_save_dir, self.args.project_root),
                     "model_save_dir": os.path.relpath(save_dir, self.args.reporter_save_dir),
                     "total_model": _torch_save(model, save_dir, "total_model.pt", dill),
                     "state_dict": _torch_save(state_dict, save_dir, "state_dict.pt", dill),

                     }
        if optimizer_ is not None:
            # json_dict["optimizer"] = _torch_save(optimizer_, save_dir, "optimizer.pt")
            json_dict["optimizer_state_dict"] = _torch_save(optimizer_.state_dict(), save_dir, "optimizer_state.pt")

        if scheduler_ is not None:
            # json_dict["scheduler"] = _torch_save(scheduler_, save_dir, "scheduler.pt")
            json_dict["scheduler_state_dict"] = _torch_save(scheduler_.state_dict(), save_dir, "scheduler_state.pt")

        with open(json_path, "w") as F:
            json.dump(json_dict, F, indent=2)

        return json_dict

    def load_model(self, json_path, model_device, replace=True):
        if os.path.isdir(json_path):
            json_path = os.path.join(json_path, "config.json")
        save_dir:str = os.path.dirname(json_path)
        with open(json_path, 'r') as F:
            data = json.load(F)

        optimizer_state_dict = {}
        scheduler_state_dict = {}
        model_name = data["model_name"]
        last_epoch = data["last_epoch"]
        # proj_dir = data["proj_dir"]
        # model_save_dir:str = data["model_save_dir"]
        def _torch_load(path, pickle_module: any = None,weights_only=True):
            abs_path = os.path.join(save_dir, path)
            rtn = torch.load(abs_path, map_location=model_device,pickle_module=pickle_module, weights_only=weights_only)
            return rtn
        state_dict = _torch_load(data['state_dict'],weights_only=True)
        # state_dict = torch.load(data['state_dict'], map_location=model_device, weights_only=True)
        if "optimizer_state_dict" in data.keys():
            optimizer_state_dict = _torch_load(data["optimizer_state_dict"], weights_only=True)
            # optimizer_state_dict = torch.load((data["optimizer_state_dict"]), map_location=model_device,weights_only=True)
        if "scheduler_state_dict" in data.keys():
            scheduler_state_dict = _torch_load(data["scheduler_state_dict"], weights_only=True)
            # scheduler_state_dict = torch.load(data["scheduler_state_dict"], map_location=model_device,weights_only=True)
        model = _torch_load(data["total_model"], pickle_module=dill, weights_only=False)
        # model = torch.load(data["total_model"], map_location=model_device, pickle_module=dill)
        model.load_state_dict(state_dict)
        if replace:
            self.trainer_state.epoch = last_epoch if last_epoch is not None else 0
            self.args.epoch_current = last_epoch if last_epoch is not None else 0
            self.init_model(model)
            
            self.loaded_optimizer_state_dict = optimizer_state_dict
            self.loaded_scheduler_state_dict = scheduler_state_dict
        else:
            return model, optimizer_state_dict, scheduler_state_dict, last_epoch

    def _nan_process(self,sample, output, target):
        data_len = len(sample)
        self.trainer_ctrl.flag_step_nan = True
        try:
            # batch_loss = []
            # for idx in range(data_len):
            #     # batch_loss.append(self.compute_loss_func(output[idx], target[idx]))
            #     batch_loss.append(self.compute_loss(output[idx], target[idx]))
            #     # div_loss.append(self.criterion(pred[i, ..., :2], target[i, ..., :2]))
            # new_loss = torch.stack(batch_loss)
            self.reporter(f"预测结果出现nan 坐标：(原始形状为{output.shape})", flag_log=True)
            nan_pos = torch.stack(torch.where(torch.isnan(output))).T
            if nan_pos.shape[0] > 0:
                self.reporter(str(nan_pos), flag_log=True)
                self.reporter(f"对应位置的输入与输出与标签：\n", flag_log=True)  # 如果输入的第一个维度不是batch 会失效
                self.reporter("sample:\n" + str(sample[nan_pos[0]]), flag_log=True)
                self.reporter("output:\n" + str(output[nan_pos[0]]), flag_log=True)
                self.reporter("target:\n" + str(target[nan_pos[0]]), flag_log=True)
            else:
                self.reporter("未找到nan的位置\n", flag_log=True)
        except Exception as e:
            self.reporter("查询nan细节发生错误\n" + str(e) + "\n", flag_log=True)
        self.callback_handler.on_nan_happen(self.args,self.trainer_state,self.trainer_ctrl)

    def _update_optimizer(self, loss, epoch=None):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(loss)
        elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.StepLR):
            self.lr_scheduler.step()
        elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.lr_scheduler.step()
        else:
            self.lr_scheduler.step()

    def _get_lr_now(self):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return self.optimizer.param_groups[0]['lr']
        else:
            return self.lr_scheduler.get_last_lr()[0]

    def _set_lr_now(self, lr):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            self.lr_scheduler.get_last_lr()[0] = lr


















