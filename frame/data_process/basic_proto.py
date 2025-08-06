# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2025/2/18 14:51
# # @Author  : oliver
# # @File    : basic_proto.py
# # @Software: PyCharm
# import json
# from copy import deepcopy
#
# import numpy as np
# import torch
# import pandas as pd
# import datetime
#
# FILENAME_data_group_track_pt = "tracklist.pt"
# FILENAME_data_group_info = "info.json"
# FILENAME_data_group_csv = "all_track.csv"
# COLUMN_divide_data = "sample_id"
#
# class Data(object):
#     time_dtype="<M8[ns]"
#     def __init__(self, show_input_sort_id=False, sort_id_header="_inner_id", time_header = "time"):
#         self._raw_data: pd.DataFrame = pd.DataFrame()
#         self.buffer = {}
#         self.data_num= 0
#         self.process_step = {}
#         self.transform_step = {}
#         self.columns = None
#         # self.data_columns =None
#         self._info_dict={"gene_time":datetime.datetime.now().strftime(f"%Y-%m-%d_%H:%M:%S"),
#                         "data_num":0,
#                         "process_step":self.process_step, }
#         self.sort_id_header = sort_id_header
#         self.show_input_sort_id = show_input_sort_id
#         self.time_header = time_header
#
#     def append(self, data: pd.DataFrame):
#         if self.sort_id_header is not None:
#             data[self.sort_id_header] = self.data_num
#         self.buffer.update({self.data_num:data})
#         self.data_num+=1
#         self._build()
#
#     def extend(self, data_list: list[pd.DataFrame]):
#         for data in data_list:
#             if self.sort_id_header is not None:
#                 data[self.sort_id_header] = self.data_num
#             self.buffer.update({self.data_num: data})
#             self.data_num += 1
#         self._build()
#
#     def _build(self):
#         self._raw_data = pd.concat({-1:self._raw_data, **self.buffer}, ignore_index=True)
#         if self.sort_id_header is not None:
#             self._raw_data[self.sort_id_header] = self._raw_data[self.sort_id_header].astype(int)
#         self.buffer.clear()
#         self._info_dict["data_num"]=self.data_num
#         self.columns = self._info_dict["data_columns_info"]=list(self._raw_data.columns)
#         # self.data_columns = list(set(self.columns) - {self.time_header, self.sort_id_header})
#
#     def __add__(self, other):
#         assert isinstance(other, Data)
#         self._raw_data = pd.concat([self._raw_data, other.raw_data], ignore_index=True)
#         if self.sort_id_header is not None:
#             other.raw_data[self.sort_id_header] += self.data_num
#         self.data_num += other.data_num
#         self._build()
#         return self
#
#     def group_by_id(self, start=0., end=1.):
#         if start < end <= 1 or (isinstance(start, float) and isinstance(end, float)):
#             id_start = self.data_num*start
#             id_end = self.data_num*end
#         else:
#             id_start = start
#             id_end = end
#
#         return self._raw_data.groupby(self.sort_id_header).filter(lambda x: id_start <= x.name < id_end).groupby(self.sort_id_header)
#
#     @property
#     def raw_data(self):
#         if self.show_input_sort_id:
#             return self._raw_data
#         else:
#             return self._raw_data[self.columns]
#
#     @raw_data.setter
#     def raw_data(self, data: pd.DataFrame):
#         self._raw_data = deepcopy(data)
#
#     @property
#     def info(self):
#         self._info_dict["data_num"]=self.data_num
#         return self._info_dict
#
#     @property
#     def data_columns(self):
#         rtn = list(self.columns)
#         if self.sort_id_header in rtn:
#             rtn.remove(self.sort_id_header)
#         if self.time_header in rtn:
#             rtn.remove(self.time_header)
#         return rtn
#
#     def __repr__(self):
#         return self.info.__repr__() +f"\n data shape: {self._raw_data.shape}" + "\n"+self._raw_data.__repr__()
#
#     def copy(self):
#         return deepcopy(self)
#
#     def update_info(self, info_dict):
#         self._info_dict.update(info_dict.copy())
#
#     def update_info_by_json(self, json_path):
#         with open(json_path, 'r') as f:
#             info = json.load(f)
#         self.update_info(info)
#         self.data_num = info["data_num"]
#         self.process_step = info["process_step"]
#         self.columns = info["data_columns_info"]
#
#     def append_process_step(self, process_step):
#         self.process_step.update({str(len(self.process_step)):process_step})
#         return self
#
#     def append_transform_step(self, transform, para_dict=None):
#         self.transform_step.update({str(len(self.transform_step)): {"transform":transform, "para_dict":para_dict}})
#         return self
#
#     # def apply_process(self, processors):
#     #     data = self.copy()
#     #     for processor in processors:
#     #         data = processor.process(data)
#     #     return self
#
#     def apply_transform(self, transform):
#         data = self.copy()
#         para, data.raw_data[self.data_columns] = transform.fit_trans(data.raw_data[self.data_columns])
#         data.append_transform_step(transform,None)
#         return data
#
#     def apply_transform_by_id(self, transform):
#         data = self.copy()
#         # tracks = []
#         para_dict = {}
#         for i, track in self.group_by_id():
#             para, new_track = transform.fit_trans(track[self.data_columns])
#             para_dict.update({int(i): para})
#             data.raw_data.loc[data.raw_data[data.sort_id_header] == i, self.data_columns] = new_track
#             # tracks.append(new_track)
#         # original_data.raw_data[self.data_columns] = pd.concat(tracks, ignore_index=True)
#         data.append_transform_step(transform, para_dict)
#         return data
#
#     def inv_transforms(self, data:pd.DataFrame or torch.Tensor, depth=None, id_=None)-> pd.DataFrame or np.array:
#         if depth is None:
#             depth = len(self.transform_step)
#         # if isinstance(original_data, torch.Tensor):
#         #     df_data = np.array(original_data.cpu())
#         # elif isinstance(original_data, pd.DataFrame):
#         #     df_data = np.array(original_data.loc[:,self.data_columns])
#         # else:
#         #     df_data = original_data.copy()
#         # for k, v in reversed(self.transform_step.items()):
#         #     trans_model = v["transform"]
#         #     para = v["para_dict"]
#         #     if para is not None:
#         #         trans_model.set_para(**para[id_])
#         #     df_data = trans_model.inverse_transform(df_data)
#         #     depth -=1
#         #     if depth == 0:
#         #         break
#         if isinstance(data, torch.Tensor):
#             df_data = pd.DataFrame(data.cpu(), columns=self.data_columns)
#
#         elif isinstance(data, np.ndarray):
#             assert len(self.data_columns) == data.shape[-1]
#             cpy_data = data.copy()
#             data_view = cpy_data.reshape([-1,len(self.data_columns)])
#             df_data = pd.DataFrame(data_view, columns=self.data_columns)
#         else:
#             df_data = data.copy()
#
#         for k, v in reversed(self.transform_step.items()):
#             trans_model = v["transform"]
#             para = v["para_dict"]
#             if para is not None:
#                 trans_model.set_para(**para[id_])
#             df_data[self.data_columns] = trans_model.inverse_transform(df_data[self.data_columns])
#             depth -=1
#             if depth == 0:
#                 break
#
#         if isinstance(data, np.ndarray):
#             data_view[:] = df_data[self.data_columns]
#             return cpy_data
#         return df_data
#
#     def inv_transforms_by_id(self, data:pd.DataFrame, depth=None) -> pd.DataFrame:
#         if self.sort_id_header not in data.columns:
#             return None
#         # tracks = []
#         if isinstance(data, torch.Tensor):
#             df_data = pd.DataFrame(data.cpu(), columns=self.data_columns)
#
#         elif isinstance(data, np.ndarray):
#             assert len(self.data_columns) == data.shape[-1]
#             cpy_data = data.copy()
#             data_view = cpy_data.reshape([-1, len(self.data_columns)])
#             df_data = pd.DataFrame(data_view, columns=self.data_columns)
#         else:
#             df_data = data.copy()
#
#         for i, track in df_data.groupby(self.sort_id_header):
#             new_track = self.inv_transforms(track[self.data_columns], depth, i)
#             df_data.loc[df_data[self.sort_id_header] == i, self.data_columns] = new_track
#             # tracks.append(new_track)
#         # raw_data[self.data_columns] = tmp_data[self.data_columns]
#         if isinstance(data, np.ndarray):
#             data_view[:] = df_data[self.data_columns]
#             return cpy_data
#         return df_data
#
#
#
#
#
#
# if __name__ == '__main__':
#     aa = pd.DataFrame([[1,2,3],[4,5,6]])
#     bb = pd.DataFrame([[11,22,33],[44,55,66]])
#     data_obj = Data(True)
