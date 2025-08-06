#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 13:48
# @Author  : Oliver
# @File    : support_config_parser.py
# @Software: PyCharm
import sys
from frame.reporter import Reporter
from frame.data_process.data_reader import DataReaderDataGroup
from frame.data_process.data_dataset import SlideWindowDataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from frame.data_process.data_transform import *
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from config import *
from frame import MetricCalculate
from frame.training_args import TrainingArguments
import configparser


class ConfigManager:
    def __init__(self, config_filepath, dataset_section_name=None, save_dir_prefix=None):
        self.config_engine = config_engine = configparser.ConfigParser()
        self.config_engine.read(config_filepath)

        self.section_general =  config_engine["general_config"]
        if dataset_section_name is not None:
            self.default_dataset_section_name = dataset_section_name
        else:
            self.default_dataset_section_name = self.section_general.get("default_dataset_section")
        self.section_dataset = config_engine[self.default_dataset_section_name]
        self.section_model = None

        # self.save_dir_prefix = save_dir_prefix if save_dir_prefix is not None else "DefaultFrameModel"
        self.save_dir_prefix = save_dir_prefix
        self.args:TrainingArguments = TrainingArguments(sys.argv,
                                                        save_dir_prefix=save_dir_prefix if save_dir_prefix is not None else "DefaultFrameModel")
        self.root_reporter = self.args.reporter

        self.train_args:TrainingArguments = None
        # self.collate_fn:Callable = None
        # self.original_func = None
        # self.used_data = None
        # self.original_data = None

        # ctrl
        self.last_section_dataset_name = None
        self.last_dataset_process_type = None
        self.train_dataset:Dataset = None
        self.valid_dataset:Dataset = None
        self.data_original = None
        self.data_processed = None
        self.data_fn_inv_eval_prediction = None
        self.data_fn_collate = None
        self.data_fn_metric = None

        self.parse_dataset(None, _force=True)

    def get_config_dict(self):
        return {section: dict(self.config_engine.items(section)) for section in self.config_engine.sections()}

    def parse_dataset(self, dataset_section_name=None, process_type=SlideWindowDataset, _force=False):
        if dataset_section_name is None:
            dataset_section_name = self.default_dataset_section_name

        if (process_type == self.last_dataset_process_type
            and dataset_section_name == self.last_section_dataset_name
            and not _force):
            # self.root_reporter(f"[!] skipped dataset parse, since already performed <{str(process_type.__name__)}> in [{dataset_section_name}] last time")
            return
        else:
            self.section_dataset = self.config_engine[dataset_section_name]
        # 参数集中获取
        # dataset_section = self.section_dataset
        general_section = self.section_general
        seq_len = general_section.getint("seq_len")
        pred_len = general_section.getint("pred_len")
        state_dim = general_section.getint("state_dim")
        seed = general_section.getint("seed")
        train_percent = general_section.getfloat("train_percentage")
        valid_percent = general_section.getfloat("valid_percentage")
        test_percent = general_section.getfloat("test_percentage")
        valid_merge_test = general_section.getboolean("valid_merge_test")

        # 轨迹数据读取
        track_path = os.path.join(load_root(), "source", "data_group", self.section_dataset.get("track_path_dir"))

        original_data = DataReaderDataGroup().read(track_path)

        # 数据预处理
        used_data = original_data.transform_step.apply(StandardScaler(), ["time"],
                                                       general_section.get("preprocess_mode"))

        # 生成数据集
        if valid_merge_test:
            valid_end = train_percent + valid_percent + test_percent
        else:
            valid_end = train_percent + valid_percent
        train_set = ConcatDataset([
            process_type(track.iloc[:, :].to_numpy(), seq_len + pred_len, offset=0, with_idx=i)
            for track, i in used_data.cut_dataset(end=train_percent)
        ])
        valid_set = ConcatDataset([
            process_type(track.iloc[:, :].to_numpy(), seq_len + pred_len, offset=0, with_idx=i)
            for track, i in used_data.cut_dataset(start=train_percent, end=valid_end)
        ])
        collate_fn = process_type.get_collate_fn(seq_len)
        original_func = process_type.get_original_eval_prediction(used_data)

        # 计算误差的方法 -- 和计算loss区别开来
        metric_method = (MetricCalculate()
                         .add_metric("MSE", MeanSquaredError(num_outputs=state_dim))
                         .add_metric("MAE", MeanAbsoluteError(num_outputs=state_dim)))
        metric_method_func = process_type.get_modified_metric_method(metric_method, used_data)

        self.data_original = original_data
        self.data_processed = used_data
        self.train_dataset = train_set
        self.valid_dataset = valid_set
        self.data_fn_inv_eval_prediction = original_func
        self.data_fn_collate = collate_fn
        self.data_fn_metric = metric_method_func

        self.last_section_dataset_name = dataset_section_name
        self.last_dataset_process_type = process_type

    def parser(self, model_section_name, dataset_section_name=None,  mode="train"):
        self.section_model = self.config_engine[model_section_name]
        self.parse_dataset(dataset_section_name)
        # if self.save_dir_prefix is None:
        #     save_dir_prefix = self.section_model.get("name") + "_" + self.section_dataset.get("name")
        # save_dir_prefix = self.save_dir_prefix
        # if dataset_section_name is not None:
        #     self.section_dataset = self.config_engine[dataset_section_name]

        # 参数集中获取
        general_section = self.section_general
        model_section = self.section_model
        # dataset_section = self.section_dataset
        # seq_len = general_section.getint("seq_len")
        # pred_len = general_section.getint("pred_len")
        # state_dim = general_section.getint("state_dim")
        seed = general_section.getint("seed")
        # train_percent = general_section.getfloat("train_percentage")
        # valid_percent = general_section.getfloat("valid_percentage")
        # test_percent = general_section.getfloat("test_percentage")
        # valid_merge_test = general_section.getboolean("valid_merge_test")

        valid_interval = model_section.getint("valid_interval")
        learning_rate = model_section.getfloat("init_learning_rate")
        batch_size = model_section.getint("batch_size")
        batch_size_test = model_section.getint("batch_size_test")
        epoch_limit = model_section.getint("epoch_limit")
        model_with_loss = model_section.getboolean("model_with_loss")

        # # 轨迹数据读取
        # track_data_basename = dataset_section.get("track_path_dir")
        # track_path = os.path.join(load_root(), "source", "data_group", track_data_basename)
        #
        # original_data = DataReaderDataGroup().read(track_path)
        #
        # # 数据预处理
        # used_data = original_data.transform_step.apply(StandardScaler(), ["time"],
        #                                                general_section.get("preprocess_mode"))
        #
        # # 生成数据集
        # if valid_merge_test:
        #     valid_end = train_percent + valid_percent + test_percent
        # else:
        #     valid_end = train_percent + valid_percent
        # train_set = ConcatDataset([
        #     SlideWindowDataset(track.iloc[:, :].to_numpy(), seq_len + pred_len, offset=0, with_idx=i)
        #     for track, i in used_data.cut_dataset(end=train_percent)
        # ])
        # valid_set = ConcatDataset([
        #     SlideWindowDataset(track.iloc[:, :].to_numpy(), seq_len + pred_len, offset=0, with_idx=i)
        #     for track, i in used_data.cut_dataset(start=train_percent, end=valid_end)
        # ])
        # collate_fn = SlideWindowDataset.get_collate_fn(seq_len)
        # original_func = SlideWindowDataset.get_original_eval_prediction(used_data)
        #
        # # 计算误差的方法 -- 和计算loss区别开来
        # metric_method = (MetricCalculate()
        #                  .add_metric("MSE", MeanSquaredError(num_outputs=state_dim))
        #                  .add_metric("MAE", MeanAbsoluteError(num_outputs=state_dim)))
        # metric_method_func = SlideWindowDataset.get_modified_metric_method(metric_method, used_data)

        if mode == "train":
            # 定义训练参数
            if self.save_dir_prefix is None:
                save_dir_prefix = model_section.get("name") + "_" + self.section_dataset.get("track_path_dir")  # 将会作为保存路径的名字
                # self.root_reporter.rename_save_dir(save_dir_prefix)
            else:
                save_dir_prefix = self.save_dir_prefix
                # self.root_reporter.rename_save_dir(save_dir_prefix)
            args = (TrainingArguments(sys.argv, save_dir_prefix=save_dir_prefix,reporter=self.root_reporter)
                    .declare_regression(self.data_fn_metric)
                    .set_dataloader(data_collator=self.data_fn_collate)
                    .set_training(learning_rate=learning_rate,valid_interval=valid_interval, batch_size=batch_size,
                                  epoch_limit=epoch_limit, seed=seed))
            args.model_with_loss = model_with_loss
        else:
            if self.save_dir_prefix is None:
                save_dir_prefix = "Test_"+model_section.get("name") + "_" + self.section_dataset.get("track_path_dir")  # 将会作为保存路径的名字
                # self.root_reporter.rename_save_dir(save_dir_prefix)
            else:
                save_dir_prefix = "Test_" + self.save_dir_prefix
                # self.root_reporter.rename_save_dir(save_dir_prefix)
            args = (TrainingArguments(sys.argv, save_dir_prefix=save_dir_prefix,reporter=self.root_reporter)
                    .declare_regression(self.data_fn_metric)
                    .set_dataloader(data_collator=self.data_fn_collate)
                    .set_evaluation(batch_size=batch_size_test)
                    )
            args.model_with_loss = model_with_loss

        # self._raw_args = args

        # self.train_args = self.get_args()
        self.train_args = args
        # self.train_dataset = train_set
        # self.valid_dataset = valid_set
        # # self.collate_fn = collate_fn
        # self.original_func = original_func
        # self.original_data = original_data
        # self.used_data = used_data
        # self.save_dir_prefix = save_dir_prefix
        return self


    def config_summary(self, reporter:Reporter, model_init_config=None, stdout=False):
        # record
        # reporter("*******************************************************").md().log()
        reporter("[+]general_section of used config:",flag_stdout=stdout, flag_md=True, flag_log=True)
        for k, v in self.section_general.items():
            reporter(f"\t{k}:{v}",flag_stdout=stdout, flag_md=True, flag_log=True)
        if self.section_model is not None:
            reporter("[+]model_section of used config:",flag_stdout=stdout, flag_md=True, flag_log=True)
            for k, v in self.section_model.items():
                reporter(f"\t{k}:{v}",flag_stdout=stdout, flag_md=True, flag_log=True)
        if model_init_config is not None:
            reporter("[+]model init config:",flag_stdout=stdout, flag_md=True, flag_log=True)
            for k, v in model_init_config.items():
                reporter(f"\t{k}:{v}",flag_stdout=stdout, flag_md=True, flag_log=True)

        # 获取当前 PyTorch 的随机种子
        torch_seed = torch.random.initial_seed()
        # 获取当前 NumPy 的随机状态
        random_state = np.random.get_state()
        numpy_seed = random_state[1][0]  # 种子通常在状态数组的第二个元素中
        reporter(f"PyTorch random seed: {torch_seed}",flag_stdout=stdout, flag_md=True, flag_log=True)
        reporter(f"NumPy random seed: {numpy_seed}", flag_stdout=stdout, flag_md=True, flag_log=True)
        reporter(f"Args seed: {self.args.seed}", flag_stdout=stdout, flag_md=True, flag_log=True)
        # reporter("*******************************************************").md().log()

    def getint(self, item):
        if self.section_model is not None and item in self.section_model:
            return self.section_model.getint(item)
        elif item in self.section_dataset:
            return self.section_dataset.getint(item)
        elif item in self.section_general:
            return self.section_general.getint(item)
        else:
            raise ValueError(f"[!-]key {item} not in config file")

    def getboolean(self, item):
        if self.section_model is not None and item in self.section_model:
            return self.section_model.getboolean(item)
        elif item in self.section_dataset:
            return self.section_dataset.getboolean(item)
        elif item in self.section_general:
            return self.section_general.getboolean(item)
        else:
            raise ValueError(f"[!-]key {item} not in config file")

    def getfloat(self, item):
        if self.section_model is not None and item in self.section_model:
            return self.section_model.getfloat(item)
        elif item in self.section_dataset:
            return self.section_dataset.getfloat(item)
        elif item in self.section_general:
            return self.section_general.getfloat(item)
        else:
            raise ValueError(f"[!-]key {item} not in config file")
