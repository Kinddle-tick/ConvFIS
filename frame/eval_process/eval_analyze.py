#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/6 20:43
# @Author  : oliver
# @File    : eval_analyze.py
# @Software: PyCharm
import colorsys
import os
import time
import warnings
from copy import deepcopy
from typing import Hashable, Iterable, AnyStr, Any

from .painter.interpretable import show_defuzzifier_2d, show_samples_2d
# from .painter.predict_show import compare_prediction
# from .painter.track_display import track_display_heatmap
# from .painter.interpretable import data_in_max, rule_frequency, data_in_alpha, data_divided
from ..util import EvalPrediction
from ..training_args import TrainingArguments
from ..reporter import Reporter
from matplotlib import pyplot as plt
from .painter import *
from ..painter_format import PlotTemplate
from .error_fuction import (ErrorMse, ErrorRMse, ErrorMae, ErrorEuclidean, _ErrorCalculator,ErrorMape)
import numpy as np
import torch
import pandas as pd
from contextlib import contextmanager

class _DataHandler(object):
    def __init__(self, event_call_back=None):
        self.raw_data = {}
        self.step_handle_model_name = []
        self.metadata = {}
        self.event_call_back = event_call_back

    def update_data(self, dict_, **kwargs):
        for k, v in {**dict_, **kwargs}.items():
            self.add_data(k, v)

    def add_data(self, name, data):
        self.raw_data[name] = data
        self.step_handle_model_name.append(name)
        # self._event_cached_metadata(name)
        self.event_call_back("cached_metadata", name)

    def clear_raw_data(self, name=None):
        if name is None:
            for name in self.step_handle_model_name:
                if name in self.raw_data:
                    self.clear_raw_data(name)
        else:
            self.raw_data.pop(name)
            self.step_handle_model_name.remove(name)

    # def _event_cached_metadata(self, name):
    #     data = self.raw_data[name]
    #     buffer_dict = self.metadata[name] = {}
    #     return data, buffer_dict

class EvalHandler:
    error_classes = [ErrorMse, ErrorMape, ErrorRMse, ErrorMae, ErrorEuclidean]
    def __init__(self, reporter:Reporter, mark_main_model=None, plot_mode="academic_mode",
                 save_raw_data=False):
        # self._flag_free_mem_data = not save_raw_data
        self.df_main_model = mark_main_model
        self._indicator:dict[str,dict[str,Hashable]]= {}    # 这些参数一定是可以列为二维列表的
        self._registered_data :dict[str, dict] = {}         # 需要注册才会保存的数据，常常对应到register函数中，直接处理原始数据获得中间数据以备使用
        self.flag_step_mode = False                         # 标志当前是否是原始数据模式
        self.data_handler = _DataHandler(event_call_back=self._event_manager)
        # self.raw_data:dict[str,EvalPrediction] = self.data_handler.raw_data         # EvalPrediction 可能会占用大量内存 因此默认只会在manager_raw_data作用域内存在一段时间
        # self.step_handle_model_name = self.data_handler.step_handle_model_name                    # 缓存当前模式下处理的模型名称
        self._metadata :dict[str,dict] = self.data_handler.metadata                 # 默认会保存的元数据，会在退出原始数据模式时，删除原始数据并自动生成metadata
        # self.raw_data:dict[str,EvalPrediction] = {}         # EvalPrediction 可能会占用大量内存 因此默认只会在manager_raw_data作用域内存在一段时间
        # self.step_handle_model_name = []                    # 缓存当前模式下处理的模型名称
        # self._metadata :dict[str,dict] = {}                 # 默认会保存的元数据，会在退出原始数据模式时，删除原始数据并自动生成metadata

        self.reporter:Reporter = reporter
        self.plot_temp = PlotTemplate(mode=plot_mode)
        self.df_index_map = {"eval_time_per_item_ms": "time",
                             "eval_items_per_ms": "speed",}
        self.summary_warning_msg=set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reporter(f"[!] Report saved in: {self.get_save_dir()}")
        if self.summary_warning_msg:
            self.reporter("[!] warning summary:")
            for warning_msg in self.summary_warning_msg:
                self.reporter("\t"+warning_msg).md()

    def warning(self, warning_msg):
        self.reporter(warning_msg).log()
        self.summary_warning_msg.add(warning_msg)

    @property
    def indicator(self):
        return pd.DataFrame(self._indicator).T

    def indicator_update(self, name, update_dict:dict):
        """
        更新指标
        :param name: 实验组名称
        :param update_dict: 该实验组的一些信息
        :return:
        """
        # self.loc[name] = update_dict
        if name not in self._indicator:
            self._indicator[name] = {}
        self._indicator[name].update(update_dict)

    @contextmanager
    def manager_raw_data(self):
        # 进入原始数据模式的唯一方法
        try:
            if self.flag_step_mode:
                self.reporter("[-!] Invalid to use manager nested")
                yield None
            else:
                self.reporter("[+] Raw data mode on")
                self.flag_step_mode = True
                yield self.data_handler
        finally:
            self.flag_step_mode = False
            self.data_handler.clear_raw_data()
            # for name in self.step_handle_model_name:
            #     self.data_handler.clear_raw_data(name)
            self.reporter("[+] Raw data mode off")
            self.reporter("*******************************************************").md().log()

    def get_raw_data(self):
        return self.data_handler.raw_data

    # def update_data(self,dict_, **kwargs):
    #     for k, v in {**dict_ ,**kwargs}.items():
    #         self.add_data(k,v)
    #
    # def add_data(self, name, data:EvalPrediction):
    #     """
    #     添加 EvalPrediction 格式的数据
    #     :param name: 用于区分不同实验的名称
    #     :param data: EvalPrediction 数据
    #     """
    #
    #     self.raw_data[name] = data
    #     self.step_handle_model_name.append(name)
    #     self._event_cached_metadata(name)
    #     # self._metadata[name] = {}
    #     # self.time_data_change = time.time()
    #
    # def _clear_raw_data(self, name):
    #     self.step_handle_model_name.remove(name)
    #     self.raw_data.pop(name)

    def get_cached_metadata(self, func_name):
        return {k:v[func_name] for k,v in self._metadata.items() if func_name in v and v[func_name] is not None}

    def get_save_dir(self):
        return self.reporter.report_path

    def get_register_data(self, key, name=None):
        if name is None:
            return {name:v[key] for name,v in self._registered_data.items() if key in v}
        else:
            if key in self._registered_data[name]:
                return self._registered_data[name][key]
            else:
                return None

    def set_register_data(self, name, key, data):
        if name in self._registered_data:
            self._registered_data[name].update({key:data})
        else:
            self._registered_data.update({name:{key:data}})
        return data
    # def new_error_set(self, used_dim):
    #     if used_dim not in self._error_dict_set:
    #         # self._error_dict_set[used_dim] =  {error_name.nick_name:{} for error_name in self.error_classes}
    #         self._error_dict_set[used_dim] =  {error_name.nick_name:{} for error_name in self.error_classes}
    #     return self._error_dict_set[used_dim]

    # def get_error_set(self, used_dim, err_first=False):
    #     if used_dim not in self._error_dict_set:
    #         self._error_dict_set[used_dim] = {error_name.nick_name: {} for error_name in self.error_classes}
    #     rtn = self._error_dict_set[used_dim]
    #
    #     if err_first:
    #         err_keys = set(sum([list(i.keys()) for i in rtn.values()],start=[]))
    #         return {m: {k: rtn[k][m] for k in rtn if m in rtn[k]} for m in err_keys}
    #     else:
    #         return rtn

    # def reset_error_set(self, used_dim=None):
    #     if used_dim is None:
    #         self._error_dict_set.clear()
    #     elif used_dim in self._error_dict_set.keys():
    #         self._error_dict_set[used_dim].clear()
    #     else:
    #         return

    # def get_data_len(self):
    #     counts = [len(v) for k, v in self._data.items()]
    #     if len(set(counts)) == 1:
    #         return counts[0]
    #     else:
    #         return None




    @staticmethod
    def tool_reverse_double_layer_dict(data_dict:dict[str, dict[str, Any]]):
        keys = {k for d in data_dict.values() for k in d.keys()}
        return {m: {k: data_dict[k][m] for k in data_dict if m in data_dict[k]} for m in keys}

    @staticmethod
    def tool_random_id_sample(n, total_num, seed, sort=False, replace=False):
        """
        随机取样的id
        :param n: 取样个数
        :param total_num: 总数 n in [0,total_num-1]
        :param seed: 种子
        :param sort: 排序
        :param replace: 是否换位（？
        :return:
        """
        if total_num < n:
            replace = True
            warnings.warn(
                "Cannot take a larger sample than population when 'replace=False'. "
                "Forcing replace=True to allow sampling.",
                category=RuntimeWarning
            )
        rng = np.random.default_rng(seed=seed)
        samples = np.arange(total_num)
        choice = rng.choice(samples, n, replace=replace)
        return np.sort(choice) if sort else choice

    def register_count_paras(self, name, model:torch.nn.Module):
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        # 计算可训练的参数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.indicator_update(name, {"trainable_params": trainable_params, "total_params": total_params})

    def register_error(self, *args, **kwargs):
        pass

    def summary_count_paras(self):
        return self.indicator.loc[["trainable_params", "total_params"]]


    def eval_show_indicator(self, columns, main_model_name=None, index_map=None, text_round=5,
                             **kwargs):
        """
        展示数量关系的函数
        :param columns: 要展示的列 当长度为1,展示柱状图；当长度为2，展示2D图
        :param main_model_name: 作为主体模型的列
        :param index_map: 可以提供一个映射，使得名称在绘图时候发生改变{a:b}意味着indicator中名为a的列会在绘图时变成b
        :return:
        """
        main_model_name = self.df_main_model if main_model_name is None else main_model_name
        index_map = self.df_index_map if index_map is None else index_map
        if not all(col in self.indicator.index for col in columns):
            self.reporter(f"[-!] {columns} not always in self.indicator.index")
            return
        charactor = self.indicator.T[columns].copy()
        charactor.columns = [index_map[column] if column in index_map else column for column in charactor.columns]

        if len(columns)<2:
            self.reporter(f"[+] Showing indicator {columns} of models in bar graph.").md().log()
        else:
            self.reporter(f"[+] Compare indicators {columns} of models in scatter graph.").md().log()

        fig_paths = compare_df(charactor, self.reporter, self.plot_temp, main_model_name,text_round=text_round, **kwargs)
        for fig_path in fig_paths:
            self.reporter.add_figure_by_path(fig_path)

    def save_indicator_csv(self):
        save_path = os.path.join(self.reporter.report_path, "indicator.csv")
        self.indicator.to_csv(save_path)
        self.reporter(f"[+] Saved indicator in {save_path}").md()

    # def summary_eval_end(self, indicator=True, **kwargs):
    #     self.reporter("[+] All eval have finished. Now summary：")
    #     if indicator:
    #         self.save_indicator_csv()


    def set_plot_temp(self, plot_temp:PlotTemplate):
        self.plot_temp = plot_temp

    def set_plot_params(self,params):
        self.plot_temp.set_params(params)

    def reset_plot_temp(self):
        self.plot_temp.set_params_default()


    def _event_manager(self, event_name, *args,**kwargs):
        return getattr(self, "_event_"+event_name)(*args,**kwargs)

    def _event_cached_metadata(self, name):
        data = self.get_raw_data()[name]
        # data = self.raw_data[name]
        buffer_dict = self._metadata[name] = {}
        return data, buffer_dict

    @staticmethod
    def use_raw_data(func):
        def wrapper(self:EvalHandler, *args, **kwargs):
            if self.flag_step_mode:
                return func(self, *args, **kwargs)
            else:
                self.warning(f"[Warning!] This function <{func.__name__}> can be used only if in raw_data mode.\n")
                print(f"Usage: with eval_hd as eval_step:\n"
                      f"        step.{func.__name__}(...)\n")
                return None
        return wrapper

class EvalTrackHandler(EvalHandler):
    def __init__(self, reporter:Reporter, mark_main_model=None, plot_mode="academic_mode",
                 default_used_dim=(0, 1), track_id_key="arg_0",
                 save_raw_data=False):
        super().__init__(reporter,mark_main_model, plot_mode=plot_mode, save_raw_data=save_raw_data)
        self.default_used_dim = default_used_dim
        self.track_id_key = track_id_key

    @property
    def indicator(self):
        metric_df = self.summary_metric(_silence=True)
        indicator_df = pd.DataFrame(self._indicator)
        return pd.concat([indicator_df,metric_df])

    def _track_id(self, data:EvalPrediction)->pd.Series:
        return pd.Series(data[self.track_id_key])

    @EvalHandler.use_raw_data
    def _data_divided_by_idx(self, data:EvalPrediction):
        tracks = {}
        idx = self._track_id(data)
        for track_id in idx.unique():
            tmp_dict = {"sample":data.sample[idx[idx == track_id].index],
                        "output":data.output[idx[idx == track_id].index],
                        "target":data.target[idx[idx == track_id].index], }
            tracks[track_id] = EvalPrediction(**tmp_dict, concatenate=False)
        return tracks

    def _calculate_error(self, data, used_dim, *args, **kwargs):
        rtn = {}
        for error_type in self.error_classes:
            error_name = error_type.nick_name
            tmp = error_type()
            tmp(data.target[..., used_dim], data.output[..., used_dim])
            rtn[error_name] = tmp
        return rtn

    # def _build_metric_dict(self, used_dim=(0,1)):
    #     process_error_dict =self.get_error_set(used_dim)
    #     data = self.get_raw_data()
    #     for model_name, pred_data in data.items():
    #         if model_name not in process_error_dict:
    #             self.reporter(f"[+] Processing metric for {model_name}.").md().log()
    #             process_error_dict[model_name] = self._calculate_error(pred_data, used_dim)
    #             # for error_type in self.error_classes:
    #             #     error_name = error_type.nick_name
    #             #     tmp = error_type()
    #             #     tmp(pred_data.target[...,used_dim], pred_data.output[...,used_dim])
    #             #     process_error_dict[model_name][error_name] = tmp
    #     return process_error_dict
        # if self.time_error_calculate > self.time_data_change and tuple(used_dim)==self.error_last_used_dim:
        #     return self._error_dict
        # else:
        #     data = self.get_raw_data()
        #     self.reporter("[+] updated the metric dict")
        #     for name, pred_data in data.items():
        #         for error_type in self.error_classes:
        #             error_name = error_type.nick_name
        #             tmp = error_type()
        #             tmp(pred_data.target[...,used_dim], pred_data.output[...,used_dim])
        #             self._error_dict[error_name][name] = tmp
        #     self.time_error_calculate = time.time()
        #     self.error_last_used_dim = tuple(used_dim)
        #     return self._error_dict


    def register_error(self, used_dim=None, *args, **kwargs):
        used_dim = self.default_used_dim if used_dim is None else used_dim
        raw_data = self.get_raw_data()
        for name,data in raw_data.items():
            raw_err = self._calculate_error(data, used_dim)
            self.set_register_data(name, "err", raw_err)
            # self._metadata[name]["eval_metrics_by_time"] = 0

        # self._build_metric_dict(used_dim)
        # self.reporter(f"[+] registered metric for {used_dim}.").md().log()

    # def summary_metric(self, _silence=False):
    #     if not _silence:
    #         self.reporter("[+] Calculating metrics summary.")
    #     # used_dim = self.default_used_dim if used_dim is None else used_dim
    #
    #     # self._build_metric_dict(used_dim)
    #     # error_dict = self.get_error_set(used_dim)
    #     error_dict = self.get_register_data("err")
    #     if error_dict:
    #         df = pd.DataFrame()
    #         for model_name, error_dict in error_dict.items():
    #             for metric_name, errors in error_dict.items():
    #                 err = np.mean(errors.mean)
    #                 df.loc[metric_name,model_name] = err
    #     else:
    #         self.warning("[-] No metric info find, use eval_hd.register_error() when eval models")
    #         return None
    #
    #     if not _silence:
    #         self.reporter(df.T.to_string(index=True)).md().log()
    #         save_path = os.path.join(self.reporter.report_path, "metric.csv")
    #         df.to_csv(save_path)
    #         self.reporter(f"[+] Saved metrics in {save_path}").md()
    #     return df

    def summary_metric(self, _silence=False):
        if not _silence:
            self.reporter("[+] Calculating metrics summary.")

        error_dict = self.get_register_data("err")
        if not error_dict:
            self.warning("[-] No metric info found, use eval_hd.register_error() when eval models")
            return None

        # 使用列表存储所有数据，最后一次性构建DataFrame
        data = []
        for model_name, model_errors in error_dict.items():
            for metric_name, errors in model_errors.items():
                err = np.mean(errors.mean)
                data.append({
                    'metric_name': metric_name,
                    'model_name': model_name,
                    'value': err
                })

        # 一次性构建DataFrame并重塑
        if data:
            df = pd.DataFrame(data)
            df = df.pivot(index='metric_name', columns='model_name', values='value')
            return df
        else:
            self.warning("[-] No valid metric data found after processing")
            return None


    def eval_metrics_by_time(self, columns=None, main_model_name:str=None, unit_s=1, show_dims_method="mean",
                             draw_confidence_interval=True):
        self.reporter("[+] Calculating metrics by time.")
        # self._build_metric_dict(self.default_used_dim)
        main_model_name = self.df_main_model if main_model_name is None else main_model_name
        # error_dict = self.get_error_set(self.default_used_dim, err_first=True)
        error_dict = self.tool_reverse_double_layer_dict(self.get_register_data("err"))
        for metric_name, error_dict in error_dict.items():
            if metric_name == "euclidean":
                metric_compare_by_time(error_dict, main_model_name, self.reporter, self.plot_temp, unit_s,
                                       [0], columns, show_dims_method, draw_confidence_interval)
            else:
                metric_compare_by_time(error_dict, main_model_name, self.reporter, self.plot_temp, unit_s,
                                       [0,1], columns, show_dims_method, draw_confidence_interval)
        return

    def eval_compare_metrics_bar(self, main_model_name:str=None, draw_horizon_line=True, log_y=False, top_number=True, box_like=True, top_round=4):
        """
        表现形式上和 eval_show_indicator 非常接近，有空可以改一下：合并或者调用
        """
        self.reporter("[+] Calculating metrics bar.")
        # self._build_metric_dict(self.default_used_dim)
        main_model_name = self.df_main_model if main_model_name is None else main_model_name
        # error_dict = self.get_error_set(self.default_used_dim, err_first=True)
        error_dict = self.tool_reverse_double_layer_dict(self.get_register_data("err"))
        for metric_name, error_dict in error_dict.items():
            if metric_name == "euclidean":
                metric_bar_compare(error_dict, self.reporter, self.plot_temp, main_model_name,
                                   [0], draw_horizon_line, log_y, top_number,top_round, box_like)
            else:
                metric_bar_compare(error_dict, self.reporter, self.plot_temp, main_model_name,
                                   [0, 1], draw_horizon_line, log_y, top_number,top_round, box_like)
        return


    def eval_track_display(self, track_data_iter, note_text="all",
                           flag_split=False, mix=True, heat=False,
                           heat_bins=100, track_ids = None, ):
        """
        [!] 尚未完全理清楚
        :param track_ids: 指定绘制的track_id
        :param heat_bins: 热度图的参数，控制热度图分的粗细
        :param heat: 是否改为绘制热度图
        :param mix: 是否将所有数据集的轨迹混合在同一张图
        :param track_data_iter:  刻碟带的轨迹数据列表-轨迹是numpy格式
        :param note_text:   轨迹数据集的名字
        :param flag_split:    是否展开一个文件夹，以展示轨迹数据集每条轨迹id对应的位置
        :return:
        """
        self.reporter("\n[+] drawing track")
        self.reporter(f"\t split:{flag_split}\n"
                      f"\t mix all:{mix}\n"
                      f"\t heat:{heat}")
        if heat:
            self.reporter(f"\t\t heat_bins={heat_bins}")

        # 选择绘制的轨迹
        if mix:
            track_draw = {note_text: [np.array(track) for track in track_data_iter]}
        else:
            if track_ids is None:
                track_unique_ids = self.get_cached_metadata("eval_track_display")
                if not track_unique_ids:
                    track_unique_ids = {"default": list(range(len(track_data_iter)))}
                # data = self.get_raw_data()
                # if len(data):
                #     track_unique_ids = {name: self._track_id(data).unique().astype(int) for name, data in data.items()}
                # else:
                #     track_unique_ids = {"default": list(range(len(track_data_iter)))}
            else:
                track_unique_ids = track_ids
            track_draw = {name: [] for name in track_unique_ids.keys()}
            for tk_id, track in enumerate(track_data_iter):
                for name, ids in track_unique_ids.items():
                    if tk_id in ids:
                        track_draw[name].append(np.array(track))

        for name, tracks in track_draw.items():
            if heat:
                tk_pt= track_display_heatmap(tracks, report=self.reporter, plot_temp=self.plot_temp,bins=heat_bins,
                                                  note_text=name)
                self.reporter.add_figure_by_path(tk_pt, describe=f"{name} 使用轨迹如上图(heatmap)", title=f"# {name}数据集轨迹信息：")
            else:
                tk_pt, h_pt, v_pt = track_display(tracks, report=self.reporter, plot_temp=self.plot_temp,
                                                  note_text=name, flag_ref=flag_split)
                self.reporter.add_figure_by_path(tk_pt, describe=f"{name} 使用轨迹如上图", title=f"# {name}数据集轨迹信息：")
                self.reporter.add_figure_by_path(h_pt, describe=f"{name} 高度数据如上图", )
                self.reporter.add_figure_by_path(v_pt, describe=f"{name} 速度数据如上图", )

        return


    @EvalHandler.use_raw_data
    def raw_eval_compare_predict_sample(self, sample_id: int or Iterable, main_model_name:str=None, save_to_md=None, used_dim=None,
                                        seed=42, alpha=0.6):
        """
        比较某一/些样本在不同模型下的预测效果。
        使用用这个函数的前提是，所有样本的长度和映射关系是一样的
        alpha可以依次控制线条的透明度。main线条的透明度一定是0.8（后续可以在plot_temp里改）
        输入数字则默认透明度全部是该数字
        输入列表则从头到尾依次替换掉透明度设置，不够的部分会用0.6填充
        """
        data = self.get_raw_data()
        def get_data_len():
            counts = [len(v) for k, v in data.items()]
            if len(set(counts)) == 1:
                return counts[0]
            else:
                return None

        used_dim = self.default_used_dim if used_dim is None else used_dim
        main_model_name = self.df_main_model if main_model_name is None else main_model_name
        if get_data_len() is None:
            self.reporter("[+] Method 'eval_compare_predict_sample' can only used in models using save input_data.")
            return None
        else:
            self.reporter("[+] Comparing Prediction from models")
        if isinstance(sample_id, int):
            sample_id = self.tool_random_id_sample(sample_id, get_data_len(), seed=seed, sort=True)
        paths = []
        main_model_name = list(data.keys())[0] if main_model_name not in data else main_model_name

        for index in sample_id:
            input_data = data[main_model_name]["sample"][index,:,used_dim]
            target_data = data[main_model_name]["target"][index,:,used_dim]
            output_data = {name:item["output"][index,:,used_dim] for name, item in data.items()}
            save_path = compare_prediction(input_data, target_data, output_data, main_model_name,
                                           self.reporter,self.plot_temp, alpha_list=alpha)
            paths.append(save_path)
        if save_to_md is None:
            save_to_md = True if len(paths) < 5 else False
        if save_to_md:
            for i, path in enumerate(paths):
                self.reporter.add_figure_by_path(path, title=f"## Compare of sample[{sample_id[i]}]")
        return

    @EvalHandler.use_raw_data
    def raw_eval_willow(self, track_data, sample_rule=10, used_tracks=None, export_willow=False, flag_compare_willow=True):
        self.reporter("[+]drawing willow")
        data = self.get_raw_data()
        track_ids = [self._track_id(data) for data in data.values()]
        raw_tracks = {i:track_data[i] for i in sorted(pd.concat(track_ids).unique())}

        willow_dict = {}
        for data_name, pred_data in data.items():
            tracks = self._data_divided_by_idx(pred_data)
            willow_dict[data_name] = {k:eval_predict.output for k, eval_predict in tracks.items()}
            fig_path = willow_graph_self(raw_tracks, willow_dict[data_name], sample_rule, used_tracks, self.reporter,
                                         self.plot_temp, data_name)
            self.reporter.add_figure_by_path(fig_path,f"willow graph of {data_name}",
                                             f"willow graph of {data_name}",f"{data_name} 的整体表现")
        if flag_compare_willow:
            fig,ax = willow_graph_cross(raw_tracks, willow_dict, sample_rule, used_tracks, self.reporter,self.plot_temp)
            self.reporter("\r[-]willow graph finish\n")
            if export_willow:
                return fig, ax
            else:
                fig_path = self.reporter.save_figure_to_file(fig, "willow_graph_cross_({})".format("_".join(willow_dict.keys())))
                plt.close(fig)
                self.reporter.add_figure_by_path(fig_path,
                                               title="# Cross willow graph of {}".format(
                                                   ",".join(list(willow_dict.keys()))))

    @EvalHandler.use_raw_data
    def raw_eval_predict_frames(self, sample_num, sample_rule="random", seed=42,
                                dir_name="predict_frame", pic_raw=3, pic_column=3, flag_show_mapping_line=True):
        """
        选择若干个采样点，展示输入-输出-预测关系 是一个帮助整体把握预测效果的的函数
        :param flag_show_mapping_line: 是否在标签和预测之间划线
        :param seed: random 采样时，种子的大小
        :param sample_num:  决定了采样的数量，如果比样本数量要多，样本会全部被使用
        :param sample_rule: "sort" or "random"(default) 决定了采样的方法
        :param dir_name:    predict样本图片保存量较大，
        :param pic_column:  生成图片有几列
        :param pic_raw:     生成图片有几行
        :return:
        """
        # self.reporter("[+] drawing predict frames")
        data = self.get_raw_data()
        # rng = np.random.default_rng(seed=seed)
        def show_id(pred_data_:EvalPrediction):
            samples = np.arange(len(pred_data_))
            if sample_rule == "random":
                return self.tool_random_id_sample(sample_num, len(pred_data_), seed=seed, sort=True)
                # return sorted(rng.choice(samples, sample_num, replace=False))
            elif sample_rule == "sort":
                return samples[:sample_num]


        for data_name, pred_data in data.items():
            idxes = show_id(pred_data)
            track_id = self._track_id(pred_data)
            sample_pos = track_id.groupby(track_id).cumcount()
            predict_data = pred_data.output[idxes]
            target_data = pred_data.target[idxes]
            input_data = pred_data.sample[idxes]
            id_desc = [f"track{i}_no{j}" for i,j in zip(track_id[idxes], sample_pos[idxes])]
            predict_show(input_data, target_data, predict_data, self.reporter,self.plot_temp,
                         data_name,id_desc, pic_column,pic_raw, dir_name, flag_show_mapping_line)

        self.reporter("Finish predict frames draw")


    def eval_gif(self):
        pass

    def _event_cached_metadata(self, name):
        data, buffer_dict = super()._event_cached_metadata(name)
        buffer_dict["eval_track_display"] = self._track_id(data).unique().astype(int)
        return data, buffer_dict

    # def script_default_run(self,raw_track, model_dict=None, *args,**kwargs):
    #     pass


class EvalConvFisHandler(EvalTrackHandler):

    @property
    def indicator(self):
        metric_df = self.summary_metric(_silence=True)
        indicator_df = pd.DataFrame(self._indicator)
        fl_pearson_df = self.summary_pearson_correlation(mode="fl").T
        rw_pearson_df = self.summary_pearson_correlation(mode="rw").T
        return pd.concat([indicator_df, metric_df, fl_pearson_df, rw_pearson_df])



    @staticmethod
    def tool_get_rule_weight(data):
        return data["phi"][...,0] if "phi" in data else None

    @staticmethod
    def tool_get_firing_level(data):
        return np.prod(data["membership_degree"],axis=-1) if "membership_degree" in data else None


    # @property
    # def rule_weight(self)->dict:          # 归一化的firing level 又叫rule weight
    #     rtn = self.get_cached_metadata("rule_weight")
    #     if self.flag_step_mode:
    #         data = self.get_raw_data()
    #         return {**rtn, **{name: pred_data["phi"][..., 0] for name, pred_data in data.items() if "phi" in pred_data}}
    #     else:
    #         return rtn
    #     # data = self.get_raw_data()
    #     # return {name:pred_data["phi"][...,0] for name,pred_data in data.items() if "phi" in pred_data}
    #
    # @property
    # def firing_level(self) -> dict:   # 真 - firing level
    #     rtn = self.get_cached_metadata("firing_level")
    #     if self.flag_step_mode:
    #         data = self.get_raw_data()
    #         return {**rtn, **{name: np.prod(pred_data["membership_degree"], axis=-1) for name, pred_data in data.items() if
    #                  "membership_degree" in pred_data}}
    #     else:
    #         return rtn
        # return rtn

    def _get_activate_data(self, mode="fl"):
        if mode =="mf" or mode =="fl":
            rtn = self.get_cached_metadata("firing_level")
            return rtn  # 这个没有归一化
        elif mode == "rw": #rule weight
            rtn = self.get_cached_metadata("rule_weight")
            return rtn       # 这个有归一化
        else:
            self.reporter("[+] mode must be either 'mf' or 'fl'")
            return None

    def get_dfz_data(self):
        return self.get_register_data("dfz_data")

    def register_defuzzifier_data(self, name, model_defuzzifier, inv_data_source=None, device = "cpu"):
        self.reporter("[+] registering defuzzifier data.")
        data = self.get_raw_data()[name]
        rule_number = data["phi"].shape[1]

        model = model_defuzzifier.to(device=device)
        output = model(None, torch.eye(rule_number)[...,None].to(device=device)).detach().cpu().numpy()
        dfz_data = output if inv_data_source is None else inv_data_source.inv_transforms(output)
        self.set_register_data(name, "dfz_data", dfz_data)

        return dfz_data


    def summary_firing_level(self, display_top_num=5, display_last_num=2, round_n=5, display_model_num=7, mode="rw"):
        firing_levels = self._get_activate_data(mode)
        utilization = {k:pd.Series(np.mean(arr,axis=0)) for k, arr in firing_levels.items()}
        for i, (name, use_rate) in enumerate(utilization.items()):
            if i>display_model_num:
                return
            tmp = use_rate.sort_values(ascending=False)
            self.reporter(f"{name}:").md().log()
            show_index = list(tmp.iloc[:display_top_num].index) +["..."]+ list(tmp.iloc[-display_last_num:].index)
            show_value = list(tmp.iloc[:display_top_num]) +["..."]+ list(tmp.iloc[-display_last_num:])

            # 处理索引列表，生成字符串
            index_strings = [str(idx) for idx in show_index]
            value_strings = [f"{val:.{round_n}f}" if isinstance(val, float) else str(val) for val in show_value]

            # 计算每列的最大宽度
            max_width = max([max(len(idx_str), len(val_str)) for idx_str, val_str in zip(index_strings, value_strings)])
            # 拼接成行
            index_line = ' '.join([f"{s:>{max_width}}" for s in index_strings])
            value_line = ' '.join([f"{s:>{max_width}}" for s in value_strings])
            self.reporter(f"{index_line}\n{value_line}").md().log()


    def summary_pearson_correlation(self, mode="fl"):
        processed_data = self._get_activate_data(mode)
        # membership_function = self.rule_membership_function
        lst_avg_pearson = list()
        lst_avg_pearson_non_diag = list()
        for data_name, level in processed_data.items():
            # level = processed_data[data_name]
            corr_matrix, avg_pearson, avg_pearson_non_diag = pearson_calculate(level)
            lst_avg_pearson.append(avg_pearson)
            lst_avg_pearson_non_diag.append(avg_pearson_non_diag)

        df = pd.DataFrame(columns=[mode+"_avg_pearson", mode+"_avg_pearson_non_diag"])
        for i, data_name in enumerate(processed_data.keys()):
            df.loc[data_name] = [lst_avg_pearson[i], lst_avg_pearson_non_diag[i]]
        return df


    def eval_bar_rule_frequency(self, xtick_show_id_interval=5, sort=True, log_y=False, mode="fl"):
        self.reporter("[+] drawing antecedent rule frequency bar.")
        processed_data = self._get_activate_data(mode=mode)
        # firing_level = self.rule_firing_level
        for name, level in processed_data.items():
            rule_frequency(level, self.reporter,self.plot_temp, name, mode, xtick_show_id_interval,
                           sort=sort, log_y=log_y)
        return

    # def eval_antecedent_category_of_track_alpha(self, show_category=None, show_data_limit=300, flag_same_limit=True,
    #                                             draw_sub_fig=False,show_legend=None, rank_legend=True, stride=10, fix_id=None):
    #     """
    #     展示每个类别最倾向的轨迹
    #     """
    #     self.reporter("[-!] Already hidden <drawing antecedent category of track of alpha>.")
    #     return None
    #     # 不同隶属度颜色深浅不同
    #     self.reporter("[+] drawing antecedent category of track of alpha.")
    #     firing_level = self.rule_weight
    #     for data_name, pred_data in self._data.items():
    #         level = firing_level[data_name]
    #         if fix_id is not None:
    #             used_data = pred_data.sample[::stride][fix_id]
    #             used_level = level[::stride][fix_id]
    #         else:
    #             used_data = pred_data.sample[::stride]
    #             used_level = level[::stride]
    #         data_in_alpha(used_data, used_level, self.reporter, self.plot_temp, data_name,
    #                       show_category=show_category, show_data_limit=show_data_limit, draw_sub_fig=draw_sub_fig, ranking_legend=rank_legend,
    #                       flag_same_limit=flag_same_limit,show_legend=show_legend)
    #     return

    def eval_defuzzifier(self, show_category=None, set_xy_lim=None, frame=False):
        self.reporter("[+] evaluating defuzzifier.")
        firing_level = self._get_activate_data("rw")
        lst_show_data = {}
        dfz_data = self.get_register_data("dfz_data")
        for i, (data_name, show_data) in enumerate(dfz_data.items()):
            level = firing_level[data_name]
            show_defuzzifier_2d(show_data, level.mean(axis=0), self.reporter, self.plot_temp, "defuzzifier_" + data_name,
                                level_limit=show_category, xy_limit=set_xy_lim, frame=data_name if frame else None)
            lst_show_data.update({data_name:show_data})
        return lst_show_data

    def eval_diffusion_analyze(self, raw_data:dict, combine=False, addition_info=None):
        if combine:
            filename = "diffusion_combine" if addition_info is None else "diffusion_combine_" + addition_info
            self.reporter("[+] evaluating combined levy judge. ")
            data = np.concatenate(list(raw_data.values()), axis=0)
            analyze_diffusion([i for i in data], self.reporter, self.plot_temp, filename)

        else:
            self.reporter("[+] evaluating levy judge. ")
            for name, data in raw_data.items():
                analyze_diffusion([i for i in data], self.reporter, self.plot_temp, f"{name}", )

    def eval_combine_diffusion_analyze(self, raw_data:dict, frame=False):
        self.reporter("[+] combined levy judge. ")
        for f, (name, data) in enumerate(raw_data.items()):
            combine_analyze_diffusion([i for i in data], self.reporter, self.plot_temp, f"diffusion_combine_{name}",
                                      frame=name if frame else None)

    # def eval_height_defuzzifier(self, model_defuzzifier, show_category=None, data_source=None, flag_same_limit=True,
    #                             draw_sub_fig=True,show_legend=None, rank_legend=True,):
    #     """
    #     传入Data source可以将输出进行inv转化，相应的不传入就会获得原始的输入
    #     该函数将会移除，因为他只能处理当前模型的defuzzifier
    #     :return:
    #     """
    #     self.reporter("[+] showing defuzzifier analyzing. \n"
    #                   "[!-] \tthis function will be removed, use register_defuzzifier_data and eval_defuzzifier_summary instead")
    #     firing_level = self._get_activate_data("rw")
    #     rule_number = list(firing_level.values())[0].shape[1]
    #     device = "cpu"
    #     model = model_defuzzifier.to(device=device)
    #     output = model(None, torch.eye(rule_number)[...,None].to(device=device)).detach().cpu().numpy()
    #     show_data = output if data_source is None else data_source.inv_transforms(output)
    #     data = self.get_raw_data()
    #     for data_name, pred_data in data.items():
    #         level = firing_level[data_name]
    #         divided_data_show(show_data, level, self.reporter, self.plot_temp, "defuzzifier_"+data_name, show_category=show_category,
    #                           draw_sub_fig=draw_sub_fig,show_legend=show_legend, ranking_legend=rank_legend, flag_same_limit=flag_same_limit)
    #     return show_data

    def eval_pearson_correlation_heatmap(self, mode="fl"):
        processed_data = self._get_activate_data(mode)
        self.reporter("[+] drawing pearson correlation matrix.")
        for data_name, level in processed_data.items():
            # self.reporter(f"{data_name}:").md()
            rule_pearson(level, self.reporter, self.plot_temp, data_name)

    def eval_pearson2force_directed(self,node_legend=False, mode="fl"):
        processed_data = self._get_activate_data(mode)
        self.reporter("[+] drawing pearson correlation force_directed.")
        for data_name, level in processed_data.items():
            # self.reporter(f"{data_name}.").md()
            pearson2force_directed(level, self.reporter, self.plot_temp, data_name,node_legend)

    def eval_combine_pearson_force_directed(self,node_legend=False, mode="fl", frame=False):
        processed_data = self._get_activate_data(mode)
        self.reporter("[+] drawing pearson + force_directed.")
        for i, (data_name, level) in enumerate(processed_data.items()):
            # self.reporter(f"{data_name}.").md()
            combine_pearson_and_force_directed(level, self.reporter, self.plot_temp, data_name, frame=data_name if frame else None)
            # pearson2force_directed(level, self.reporter, self.plot_temp, data_name, node_legend)

    @EvalHandler.use_raw_data
    def raw_eval_antecedent_category_of_track(self, show_category=None, stride=10, fix_id=None, frame=True, *args, **kwargs):
        """
        展示每条轨迹属于的类别
        show_category 可以是数字或者列表，表示部分展示
        """
        # 不同隶属度颜色深浅不同
        self.reporter("[+] drawing antecedent category of track of max.")
        firing_level = self._get_activate_data("rw")
        data = self.get_raw_data()
        for data_name, pred_data in data.items():
            level = firing_level[data_name]
            if fix_id is not None:
                used_data = pred_data.sample[::stride][fix_id]
                used_level = level[::stride][fix_id]
            else:
                used_data = pred_data.sample[::stride]
                used_level = level[::stride]
            show_samples_2d(used_data, used_level, self.reporter, self.plot_temp, data_name,
                            x_label="Longitude(°)", y_label="Latitude(°)",frame=data_name if frame else None,
                            level_limit=show_category)
        return


    def _event_cached_metadata(self, name):
        data, buffer_dict = super()._event_cached_metadata(name)
        # buffer_dict["rule_weight"] = data["phi"][...,0] if "phi" in data else None
        # buffer_dict["firing_level"] = np.prod(data["membership_degree"],axis=-1) if "membership_degree" in data else None
        buffer_dict["rule_weight"] = self.tool_get_rule_weight(data)
        buffer_dict["firing_level"] = self.tool_get_firing_level(data)