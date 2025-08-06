#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/2/18 15:43
# @Author  : oliver
# @File    : main_data_flight.py
# @Software: PyCharm
import os
from config import load_config, load_model_card, load_platform, load_root
from frame.data_process.data_analyze import TrackCompareAnalyze
from frame.data_process.data_processor import DeduplicateProcessor, MaskProcessor, ClipProcessor, ResampleProcessor, \
    LengthFilterProcessor
from frame.data_process.data_saver import Saver
from frame.data_process.data_reader import DataReaderDatabase
"""
This is the steps to convert data from database to excel.
The database can not be fully disclosed, please contact the author if you need it.
"""

load_config()
system_platform = load_platform()
root = load_root()
# 读取数据的映射键值对，键为数据库中的数据列名，值为保存为数据集时的数据列名
db_map = {"WZSJ": "time",
          "JD": "x",
          "WD": "y",
          "GD": "h",
          "SD": "v",
          }

# 读取数据的路径
# database_name = "FW.sqlite"
# used_db_id = [4]
# group_name = "Flight_Tracks_" + str(used_db_id[0])

database_name = "quin33.sqlite"
used_db_id = [0,1,2,3,4,5]
group_name = "quin33"

database_path = database_name
if system_platform == "Darwin":
    database_path = os.path.join(os.path.expanduser(os.path.join("~", "Data")), database_name)
elif system_platform == "Linux":
    database_path = os.path.join(os.path.expanduser(os.path.join("~", "Data")), database_name)
elif system_platform == "Windows":
    database_path = os.path.join(os.path.expanduser(os.path.join("~", "Data")), database_name)

save_dir = os.path.join(root, "source", "data_group")
time_seq = "6s"

processors = [DeduplicateProcessor(),
              MaskProcessor().set_lim("h", 0, 500).set_lim("v", 75, 600),
              ResampleProcessor(time_seq,interpolate_kwargs = {"method": "polynomial", "order": 2}),
              ClipProcessor().set_lim("h", 0, 500).set_lim("v", 75, 600),
              LengthFilterProcessor(120),]

if __name__ == '__main__':

    save_path = os.path.join(save_dir, "_".join([group_name, time_seq]))

    reader = DataReaderDatabase(db_map)
    # print(f"[+] processing data in: {database_path}")
    data = reader.read(database_path, used_db_id)
    old_data= data.copy()

    for processor in processors:
        # data = processor(data)
        data = data.process_step.apply(processor)
    data = data.shuffle(42)
    saver = Saver(save_path)
    saver.save(data)

    analyzer = TrackCompareAnalyze(save_path)
    analyzer.basic_data_statistics(data)
    analyzer.script_draw_data(data,"new",1024,256)
    analyzer.script_draw_data(old_data,"old",1024,256)

# if __name__ == '__main__':
    if True:
        # 仅做测试
        print("[+] Test the readability of saved data...")
        from frame.data_process.data_reader import DataReaderDataGroup
        from frame.data_process.data_transform import *
        # # 数据读取测试

        reader = DataReaderDataGroup()
        original_data = reader.read(save_path)

        # 数据预处理
        # 所有轨迹一起归一化的方法
        data2 = original_data.transform_step.apply_overall(StandardScaler(),["time"])
        data3 = original_data.transform_step.apply_individual(StandardScaler(),["time"])
        # 按照轨迹归一化的方法
        # data3 = original_data.apply_transform_by_id(StandardScaler())
        data2_inv = data2.transform_step.inv_overall(data2.get_total_data())
        data3_inv = data3.transform_step.inv_individual(data3)
        # used_data = data2
        used_data = data2
        print("[-] The data is readable!")
        # eval_hd = EvalConvFisHandler(None, plot_mode="academic_mode")

