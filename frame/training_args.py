# # Copyright 2020 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
import os
from dataclasses import dataclass, field, Field
import socket
from datetime import datetime
import platform
from random import random
from typing import Callable, Optional, List, Dict, Union

import numpy as np
import torch
from .util import get_str_time, StateEnum, IntervalStrategy, SaveStrategy
from .reporter import Reporter
from enum import Enum

from .util.util_obj import MetricStrategy, EnableSituationStrategy
import inspect
import os

# def get_caller_path():
#     # 获取调用者的栈帧
#     caller_frame = inspect.stack()[1]
#     # 获取调用者的文件名
#     caller_file = caller_frame.filename
#     # 获取调用者的绝对路径
#     caller_abs_path = os.path.abspath(caller_file)
#     # 获取调用者所在的目录
#     caller_dir = os.path.dirname(caller_abs_path)
#     # print(f"当前运行程序的路径是: {caller_dir}")
#     return caller_dir
# from .trainer_callback import TrainerState


# @dataclass
# class TrainerStates:
#     """
#     记录大部分Trainer的内容，在callback函数中将会传递
#     """
#     state_train: StateEnum = field(default=StateEnum.CLOSED, init=False)
#     state_epoch: StateEnum = field(default=StateEnum.CLOSED, init=False)
#     state_step: StateEnum = field(default=StateEnum.CLOSED, init=False)
#     # flag_step_nan: bool = field(default=False, init=False)

# class ExplicitEnum(str, Enum):
#     """
#     Enum with more explicit error message for missing values.
#     """
#
#     @classmethod
#     def _missing_(cls, value):
#         raise ValueError(
#             f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
#         )
#
# class IntervalStrategy(ExplicitEnum):
#     NO = "no"
#     STEPS = "steps"
#     EPOCH = "epoch"
#
#
# class SaveStrategy(ExplicitEnum):
#     NO = "no"
#     STEPS = "steps"
#     EPOCH = "epoch"
#     BEST = "best"
#
#
# class EvaluationStrategy(ExplicitEnum):
#     NO = "no"
#     STEPS = "steps"
#     EPOCH = "epoch"

@dataclass
class TrainingArguments:
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = "cpu"
    draw_device = "cpu"
    data_dtype = torch.float32
    sys_argv:list = field(init=True)        # 就是输入参数

    # model_name:str = field(default=None, init=True,)
    save_dir_prefix:str = field(default='UnknownModel', init=True, repr=False)

    # 路径管理相关
    project_root: str = field(default = None, init=False)
    output_dir: str = field(default = None)
    log_root_dir: str = field(default = None)
    tb_root_dir: str = field(default = None)
    reporter_save_dir:str = field(default = None)
    model_save_dir:str = field(default = None)
    loss_csv_path: str = field(default = None)
    # dft_track_path: str
    # dft_track_slice: slice

    # runtime 部分关键参数
    valid_interval:int = field(default=5, init=True)
    learning_rate: float = field(default=0.01, init=False)
    stdout_enable_: bool = field(default = True)                # 考虑和tqdm一起做到callback中

    # check_point 部分关键参数
    # cp_save_optimizer:bool = field(default = True)

    # trainer_state: TrainerState = field(default=None, metadata={"help": "saving the trainer_state"})
    epoch_current:int = field(default=0, metadata={"help": "current epoch of the model"})
    step_current:int = field(default=0, metadata={"help": "current step"})
    epoch_limit: int = field(default=50, metadata={"help": "epoch upper limit"})
    step_limit: int = field(default=None, metadata={"help": "step upper limit"})

    # dataloader 部分参数
    num_workers: int = field(default=0, metadata={"help": "num workers of dataloader"})
    data_collator: callable = field(default=None, init=False, repr=False)
    batch_size: int = field(default=16, metadata={"help": "batch size"})
    seed: int = field(default=42, metadata={"help": "random seed"})
    pin_memory: bool = field(default=False, metadata={"help": "pin memory"})
    # log 相关
    reporter:Reporter = field(default=None, init=True, repr=False)
    flag_reporter_debug_mode:bool = field(default=False, init=False, repr=False)
    flag_split_log:bool = field(default=False, repr=False)
    # flag_state_trained:bool = field(default=False, repr=False)
    #
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    # do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the valid set."})
    done_train: bool = field(default=False)
    # done_eval: bool = field(default=False)
    done_predict: bool = field(default=False)
    #
    do_regression: bool = field(default=False, metadata={"help": "Whether this machine learning task is regression."})
    do_classification: bool = field(default=False, metadata={"help": "Whether this machine learning task is classification."})

    train_compute_loss_func: Callable = field(default=lambda *args:None, init=False, repr=False)
    train_compute_metrics_func: Callable = field(default=lambda *args:None, init=False, repr=False)
    #
    train_autocast: bool = field(default=False, metadata={"help": "Whether to train autocast."})
    model_with_loss: bool = field(default=False, metadata={"help": "Whether to use model with loss. If true, should return loss from model's output"})

    # strategies
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="best",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_covered:bool = field(default=True, metadata={"help": "Whether to save the model covered."})
    # below, activate only when save_strategy == "custom"
    save_init:bool = field(default=False, metadata={"help": "Whether to save the model initially."})
    save_best:bool = field(default=False, metadata={"help": "Whether to save the best model after training."})
    save_epoch_ids:list = field(default=(), init=False, repr=False)
    save_last:bool = field(default=False, metadata={"help": "Whether to save the model last."})

    metric_strategy: Union[MetricStrategy, str] = field(
        default="epoch",
        metadata={"help": "The metric calculate strategy to use."},
    )
    metric_situation:Union[EnableSituationStrategy,str] = field(
        default= EnableSituationStrategy.ALL,
        metadata={"help": "in what situation, will use metric strategy ."},
    )

    def __post_init__(self):
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)
        self.metric_strategy = MetricStrategy(self.metric_strategy)
        self.metric_situation = EnableSituationStrategy(self.metric_situation)

        self.analyze_sys_argv(self.sys_argv)
        # if self.model_name is None:
        #     self.model_name = "Unknown"

        self.build_time = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        if self.output_dir is None:
            self.output_dir = os.path.join(self.project_root, "output")
        if self.log_root_dir is None:
            self.log_root_dir = os.path.join(self.output_dir, "runs_log")
        if self.tb_root_dir is None:
            self.tb_root_dir = os.path.join(self.output_dir, "tensorboard")

        if self.reporter is None:
            """没有传入了reporter作为参数，则进行创建"""
            # self.set_reporter_save_dir(self.save_dir_prefix)
            # self.set_reporter_save_dir(os.path.join(self.log_root_dir, self.gene_save_dir(self.save_dir_prefix)))
            self.build_reporter()
        else:
            """传入了reporter作为参数，就是一种继承关系, 但是依旧以后者的self.save_dir_prefix为准"""
            # self.set_reporter_save_dir(self.reporter.report_base_name)
            # self.set_reporter_save_dir(os.path.join(self.project_root, self.reporter.root_dir))
            self.rename_reporter_folder_name(self.save_dir_prefix)

    def rename_reporter_folder_name(self, folder_name=None):
        if folder_name is None:
            folder_name = self.save_dir_prefix
        root_reporter = self.reporter
        old_path = root_reporter.report_path
        new_path = root_reporter.rename_save_dir(folder_name)
        self.set_reporter_save_dir(new_path)
        # self.args.reporter_save_dir = new_path
        if new_path == old_path:
            self.reporter(f"[.] keeping save log in {old_path}").log()
        else:
            self.reporter(f"[+] rename save log from <{old_path}>\n"
                          f"                    to <{new_path}>").log().md()

    def build_reporter(self):
        report_path = os.path.relpath(os.path.join(self.log_root_dir, self.save_dir_prefix), self.project_root)
        save_root, save_dir = os.path.split(report_path)
        reporter = Reporter(self, save_root, save_dir, flag_split_log=self.flag_split_log)
        self.set_reporter_save_dir(report_path)
        self.reporter= reporter
        self.reporter(f"built a brand new reporter in {reporter.report_path}. since there is no reporter given to args")

    def set_reporter_save_dir(self, save_dir):
        self.reporter_save_dir = save_dir
        self.model_save_dir = os.path.join(self.reporter_save_dir, "models")
        self.loss_csv_path = os.path.join(self.reporter_save_dir, "loss.csv")

    # def gene_save_dir(self, prefix):
    #     rtn = self.build_time + "_" + prefix + "_" + platform.system()
    #     if self.done_train:
    #         rtn += "_train"
    #     # if self.done_eval:
    #     #     rtn += "_eval"
    #     return rtn

    def set_all_seed(self,seed_value=None):
        if seed_value is None:
            seed_value = self.seed
        else:
            self.seed = seed_value
        # Python
        import random
        random.seed(seed_value)

        # NumPy
        import numpy as np
        np.random.seed(seed_value)

        # # TensorFlow
        # import tensorflow as tf
        # tf.random.set_seed(seed_value)

        # PyTorch
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 环境变量
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        #
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)


    def analyze_sys_argv(self,argv):
        scrips_path = argv[0]
        self.project_root = os.path.split(scrips_path)[0]


    def bind_reporter(self, reporter):
        reporter = reporter.band(self.save_dir_prefix, self.flag_split_log)
        self.reporter = reporter

    def set_dataloader(self,data_collator=None,num_workers=None,pin_memory=None):
        # dataloader 部分参数
        self.data_collator = data_collator if data_collator is not None else self.data_collator
        self.num_workers = num_workers if num_workers is not None else self.num_workers
        self.pin_memory = pin_memory if pin_memory is not None else self.pin_memory
        return self

    def set_training(
        self,
        learning_rate: float = None,
        batch_size: int = None,
        valid_interval:int = None,
        # weight_decay: float = 0,
        epoch_limit: int = None,
        step_limit: int = None,
        # gradient_accumulation_steps: int = 1,
        seed: int = None,
        # train_auto_cast: bool = False,
        # gradient_checkpointing: bool = False,
    ):
        self.do_train = True
        self.learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.valid_interval = valid_interval if valid_interval is not None else self.valid_interval
        self.epoch_limit = epoch_limit if epoch_limit is not None else self.epoch_limit

        self.step_limit = step_limit if step_limit is not None else self.step_limit
        self.set_all_seed(seed)
        # self.seed = seed if seed is not None else self.seed
        # self.set_all_seed(self.seed)
        return self

    def set_evaluation(
        self,
        batch_size: int = None,
        # save_dir_prefix: str = None,
    ):
        self.do_eval = True

        self.batch_size = batch_size if batch_size is not None else self.batch_size

        return self

    def set_custom_save(self,
                        init=False,
                        middle:list=None,
                        best=True,
                        end=False):
        self.save_strategy = SaveStrategy.CUSTOM
        # self.save_covered=False
        self.save_init = init
        self.save_last = end
        self.save_best = best
        self.save_epoch_ids = middle if middle is not None else ()
        return self

    def declare_regression(self, compute_metrics=None, compute_loss_func=None):
        self.do_regression = True
        self.do_classification = False
        if compute_loss_func is not None:
            self.train_compute_loss_func = compute_loss_func
        if compute_metrics is not None:
            self.train_compute_metrics_func = compute_metrics

        return self

    def declare_classification(self, compute_metrics=None, compute_loss_func=None):
        self.do_classification = True
        self.do_regression = False
        if compute_loss_func is not None:
            self.train_compute_loss_func = compute_loss_func
        if compute_metrics is not None:
            self.train_compute_metrics_func = compute_metrics

        return self




