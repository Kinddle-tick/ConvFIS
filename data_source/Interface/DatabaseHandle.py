#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DatabaseHandle.py
# @Time      :2024/1/3 20:48
# @Author    :Oliver
import pandas as pd
import sqlite3
import os

"fw_flightProperty"
"fw_flightHJ"
"""
用于处理应用到的*数据库*数据
"""

class DatabaseHandle:
    """
    绑定一个数据库，并提供查找功能
    """
    # default_table_name = "main"
    used_database_path = {}

    # data_buffer = {}

    def __init__(self, db_path, default_table_name="main", frozen=True):
        self.default_table_name = default_table_name
        self.db_path = db_path
        self.frozen = frozen
        if db_path in self.used_database_path.keys():
            connection = self.used_database_path[db_path]
        else:
            try:
                print(f"[+]Connecting Database at {db_path} ...")
                connection = sqlite3.connect(db_path)
                print("[+]Connected.")
            except Exception as e:
                print(f"\nError in database path {db_path}")
                raise e
            self.used_database_path.update({db_path: connection})
        self.connection = connection

    def _free_read_sql_query(self, sql):
        return pd.read_sql_query(sql, self.connection)

    @staticmethod
    def _select_preprocess(select_data):
        if isinstance(select_data, str):
            select_str = select_data
        elif isinstance(select_data, list):
            if select_data:
                select_str = ",".join(select_data)
            else:
                select_str = "*"
        else:
            print(f"[!] invalid select sentence : {type(select_data)}")
            select_str = select_data
        return " " + select_str + " "

    def select_model(self, select: str or list = "*", from_table: str = None, where_filter: dict = None,
                     group_by: str or list = None, order_by: str = None, order_desc=True,
                     is_distinct=False):
        from_table = self.default_table_name if from_table is None else from_table
        if is_distinct:
            select_sql = "select DISTINCT " + self._select_preprocess(select)
        else:
            select_sql = "select " + self._select_preprocess(select)
        from_sql = " from " + from_table
        if where_filter is None:
            where_sql = ""
        else:
            where_sql = " where " + " and ".join([x + ' == "' + y + '"' for x, y in where_filter.items()])
        if group_by is None:
            group_sql = ""
        elif isinstance(group_by, str):
            group_sql = " group by " + group_by
        else:
            group_sql = " group by " + ",".join(group_by)

        if order_by is None:
            order_sql = ""
        else:
            order_sql = " order by " + order_by + " " + "desc" if order_desc else "asc"

        sql = select_sql + from_sql + where_sql + group_sql + order_sql
        return self._free_read_sql_query(sql)

    def save_table(self, data: pd.DataFrame, table_name):
        if self.frozen:
            return 0
        else:
            return data.to_sql(table_name, self.connection, index=False, if_exists="replace")

    # def select_all(self, select="*", table=None):
    #     return self.select_model(select,from_table=table)
    #     # table = self.default_table_name if table is None else table
    #     # select = self._select_preprocess(select)
    #
    #     # rtn = self._free_read_sql_query(f"select {select} from {table};")
    #     # return rtn
    #
    # def select_unique(self, select, table=None):
    #     """
    #     get unique item in column_name
    #     """
    #     rtn = self.select_model(select,table,is_distinct=True)
    #     # table = self.default_table_name if table is None else table
    #     # select = self._select_preprocess(select)
    #
    #     # rtn = self._free_read_sql_query(f"select DISTINCT {select} from {table};")
    #     # return rtn[select]  # 转化为序列
    #     return rtn
    #
    # def select_by(self, value, key_column, select="*", table=None):
    #     # table = self.default_table_name if table is None else table
    #     # select = self._select_preprocess(select)
    #     rtn = self.select_model(select,table,{key_column:value})
    #     # rtn = self._free_read_sql_query("select {} from {} where {} == '{}';".format(select, table, key_column, value))
    #     return rtn

    def close(self):
        self.connection.close()
        self.used_database_path.pop(self.db_path)


class _UniqueDataSet:
    """
    """
    table_name: str
    main_key_column: list
    main_keys: pd.DataFrame
    main_keys_info: pd.DataFrame
    select_columns = "*"

    def __init__(self, table_name, main_key_column, db_path):
        self.db = DatabaseHandle(db_path, table_name)
        # self.table_name = table_name
        self.main_key_column = main_key_column
        self.data_buffer = {}
        self.index_map = {}
        # self.main_keys = self.db.select_unique(main_key_column, table_name)
        self.main_keys = self.db.select_model(main_key_column, table_name, group_by=main_key_column,
                                              order_by="count(*)",
                                              is_distinct=True)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if item is None:
            return None
        if isinstance(item, int):
            if item in self.data_buffer.keys():
                return self.data_buffer[item]
            where_dict = self.main_keys.iloc[item].to_dict()
            raw_data = self.db.select_model(select=self.select_columns, where_filter=where_dict)
            data = raw_data
            self.data_buffer.update({item: data})
            return data
        else:
            return self._getitem_by_value(item)

    def _getitem_by_value(self, value: str or list or tuple):
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, str):
            value = (value,)
        if value in self.index_map:
            return self[self.index_map[value]]
        else:
            query = "&".join([f'{i}=="{j}"' for i, j in zip(self.main_key_column, value)])
            search_result = self.main_keys.query(query).index
            if len(search_result) == 0:
                idx=None
                self.index_map.update({value: idx})
                return self[idx]
            else:
                idx = int(search_result[0])
                # idx = int(idx[0])
                self.index_map.update({value: idx})
                return self[idx]

    def __len__(self):
        return len(self.main_keys)


class _DataHandle(_UniqueDataSet):
    base_table_name: str = ""
    main_key_column: str or list = ""
    select_columns: str or list = "*"

    """
    数据处理类 根据提供的主键划分数据
    """

    def __init__(self, db_path):
        self.db = DatabaseHandle(db_path, default_table_name=self.base_table_name)
        self.main_key_column = [self.main_key_column] if isinstance(self.main_key_column, str) else self.main_key_column
        # self.original_data = _UniqueDataSet(self.base_table_name, self.main_key_column, db_path)
        super().__init__(self.base_table_name,self.main_key_column,db_path)

        # self.original_data.select_columns = self.select_columns

    def get_data(self):
        return self

    def get_key(self):
        return self.main_keys

    # def select_db(self, select: str | list = "*", from_table: str = None, where_filter: dict = None,
    #                  is_distinct=False):
    #     return self.db.select_model(select,from_table,where_filter,is_distinct)

    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self[i]
    #
    # def __getitem__(self, item):
    #     if item in self.data_buffer.keys():
    #         return self.data_buffer[item]
    #     else:
    #         raw_data = self.db.select_by(self.main_keys[item], self.main_key_column)
    #         original_data = FlightPathDataFrame(raw_data)
    #         self.data_buffer.update({item: original_data})
    #         return original_data
    #
    # def __len__(self):
    #     return len(self.main_keys)


class FlightPathDataHandle(_DataHandle):
    base_table_name = "fw_flightHJ"
    main_key_column = "HBID"
    # select_columns = ["HBID", "WZSJ","RKSJ", "JD", "WD", "GD", "SD"]
    select_columns = ["HBID", "WZSJ", "JD", "WD", "GD", "SD"]
    # 航迹数据处理类 每个数据代表一条轨迹


class FlightPropertyDataHandle(_DataHandle):
    base_table_name = "fw_flightProperty"
    # main_key_column = ["CFDIATA", "DDDIATA"]
    main_key_column = ["CFDICAO", "DDDICAO"]
    select_columns = ["HBID", "HBH",
                      "CFD", "CFDIATA", "CFDICAO", "CFDJD", "CFDWD",
                      "DDD", "DDDIATA", "DDDICAO", "DDDJD", "DDDWD", ]
    # 航迹特性分类  每个数据对应了一对出发地和目的地的所有航迹，利用这些航迹的HBID可以提取到这些航迹本身

