#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/21 15:06
# @Author  : oliver
# @File    : data_dataset.py
# @Software: PyCharm
from typing import Dict, List

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from ..basic_protocal import Data
# from ..trainer import EvalPrediction
from ..util import EvalPrediction


class SlideWindowDataset(Dataset):
    def __init__(self, data_, window_size, stride=1, offset=0, with_idx=None):
        self.offset = offset
        self.window_size = window_size
        self.stride = stride
        self.with_idx = with_idx

        self.data = torch.tensor(data_)
        self.data_len = len(self.data)
        self.total_windows = (len(self.data) - self.window_size + self.stride) // self.stride

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        start_idx = (idx * self.stride + self.offset) % self.data_len
        end_idx = start_idx + self.window_size

        # 提取窗口数据
        window_data = self.data[start_idx:end_idx]

        # 如果需要返回窗口的索引信息
        if self.with_idx is not None:
            return window_data, self.with_idx
        else:
            return window_data

    @ staticmethod
    def get_collate_fn(input_len):
        """
        生成一个适配SlideWindows的collate_fn
        :param input_len: 因为SlideWindow没有划分输入轨迹和输出轨迹，此处用于切分
        :return:
        """
        def collate_fn(batch):
            track_, idx = zip(*batch)
            batch_ = torch.stack(track_)
            return batch_[:, :input_len], batch_[:, input_len:], idx
        return collate_fn

    @staticmethod
    def get_modified_metric_method(metric_method, data_source:Data, ignore_columns=("time",)):
        """
        生成一个适配SlideWindows的差错计算函数
        :param metric_method: 需要修饰的metric方法
        :param data_source: 计算metric需要将对数据集的变换转化到原始形态。
        :return:
        """
        columns = [i for i in data_source.columns if i not in ignore_columns]
        def metric_method_func(eval_predict:EvalPrediction):
            output = eval_predict.output
            label = eval_predict.target
            idx = pd.Series(eval_predict["arg_0"])

            metric_dict = {k: {} for k in metric_method.names}
            for track_id in idx.unique():
                tmp_raw_output = data_source.transform_step.inv_by_id(np.array([*output[idx[idx == track_id].index]]), i=track_id)
                tmp_raw_label = data_source.transform_step.inv_by_id(np.array([*label[idx[idx == track_id].index]]), i=track_id)

                tmp_metric = metric_method([torch.tensor(tmp_raw_output.reshape([-1,4])).contiguous(),
                                            torch.tensor(tmp_raw_label.reshape([-1,4])).contiguous()])
                for k, v in tmp_metric.items():
                    metric_dict[k][track_id] = v

            rtn_dict = {}
            info_table = pd.DataFrame(index=metric_method.names, columns=columns)
            for metric_name, track_data in metric_dict.items():
                tmp_df = pd.DataFrame(track_data, index=columns).T.sort_index()
                rtn_dict[f"detail_{metric_name}"] = tmp_df
                info_table.loc[metric_name] = tmp_df.mean()
                # rtn_dict[f"avg_{metric_name}"] = rtn_dict[f"detail_{metric_name}"].mean()
            rtn_dict["metric_table"] = info_table
            return rtn_dict
        return metric_method_func

    @staticmethod
    def get_original_eval_prediction(data_source:Data):
        def warp(eval_predict:EvalPrediction, inv_keyword = ("output", "target", "sample")):
            idx = pd.Series(eval_predict["arg_0"])
            new_eval_prediction = {k:[] for k in eval_predict.keys()}

            for track_id in idx.unique():
                index = idx[idx == track_id].index
                for k,v in eval_predict.items():
                    tmp = np.array([*v[index]])
                    if k in inv_keyword:
                        tmp = data_source.transform_step.inv_by_id(tmp, i=track_id)
                    new_eval_prediction[k].append(tmp)

            for k,v in new_eval_prediction.items():
                eval_predict.set_item(k,v)
            return eval_predict
        return warp

# class Dataset_grid(torch.utils.original_data.IterableDataset):
#     def __init__(self, seq_dict: Dict[str, List[torch.Tensor]], tokenizer, max_len, stride=2, shuffle=True, sigma=None,
#                  original_pos=None):
#         grid = tokenizer.raster_grid
#         self.center_indices = seq_dict['grid_indices']  # List[[Seq_len],]
#         self.neighbors_indices = seq_dict['neighbors_indices']  # List[[Seq_len,neighbor_num+1],] ；邻居的保存顺序按照列优先排列，包含中心点
#         self.point_position = seq_dict['point_positions']  # List[[Seq_len,2],]
#         self.max_len = max_len
#         self.stride = stride
#         self.shuffle = shuffle
#         self.pos = original_pos
#         n = grid.num_neighboring_cells_expand * 2 + 1
#         d = grid.grid_size
#         distance = self.__class__.sovle_weight(n, d, self.point_position)
#         # if sigma != None:
#         #     distance = torch.exp(-torch.square(distance)/(2*d*sigma)**2)
#         #     distance = distance/torch.exp(distance).sum(dim=-1,keepdim=True)
#         distance = F.softmax(distance)  # 行和为neighbor_num+1
#         index = 0
#         self.relative_distance = []
#         for seq in self.point_position:
#             tmp = index + len(seq)
#             self.relative_distance.append(distance[index:index + len(seq), :])
#             index = tmp
#
#         num_points = distance.shape[0]
#         self.len = (num_points - max_len) // stride
#
#         if self.pos != None:
#             self.pos = [torch.from_numpy(x.copy()) for x in self.pos]
#
#     def __len__(self):
#         return self.len
#
#     def generate_windows(self, w, s, shuffle):
#         """
#         生成滑动窗口的生成器，返回每一个序列的滑动窗口
#         :param w: 窗口大小
#         :param s: 滑动步长
#         """
#         if self.pos is None:
#             combined = list(
#                 zip(self.center_indices, self.neighbors_indices, self.point_position, self.relative_distance))
#         else:
#             combined = list(zip(self.center_indices, self.neighbors_indices, self.pos, self.relative_distance))
#         if shuffle:
#             random.shuffle(combined)
#             # # 3. Unzip back to three lists
#             # self.center_indices, self.neighbors_indices, self.point_position, self.relative = zip(*combined)
#
#         def sliding_window(original_data):
#             """
#             生成一个滑动窗口的生成器
#             :param original_data: 输入数据序列
#             :param w: 窗口大小
#             :param s: 滑动步长
#             """
#             for i in range(0, len(original_data[0]) - w + 1, s):
#                 yield [x[i:i + w] for x in original_data]
#
#         for pack in combined:
#             yield from sliding_window(pack)
#
#     def __iter__(self):
#         return self.generate_windows(self.max_len, self.stride, shuffle=self.shuffle)