# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# # @FileName  :_painter_config.py
# # @Time      :2024/1/3 17:17
# # @Author    :Oliver
# from matplotlib import pyplot as plt
# import numpy as np
#
# plot_para_dict = {"figure.width":7.0866,
#                   "legend.fontsize": 7}
#
# def draft_mode():
#     plt.rcdefaults()
#     plt.rcParams['svg.fonttype'] = 'none'
#     plt.rcParams['axes.unicode_minus'] = False
#
#
# def academic_mode():
#     plt.rcdefaults()
#     plt.rcParams['svg.fonttype'] = 'none'
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['figure.figsize'] = [7.0866, 5]
#     # 设置标准文本标注字体为 5 - 7 磅的无衬线字体
#     # 这里设置字体大小为 6pt，字体族为无衬线字体
#     plt.rcParams['font.size'] = 6
#     plt.rcParams['font.family'] = 'sans-serif'
#
#     # 设置希腊字符字体为 Symbol 字体
#     plt.rcParams['mathtext.fontset'] = 'custom'
#     plt.rcParams['mathtext.default'] = 'it'
#     # 修改为系统支持的无衬线字体名称
#     plt.rcParams['mathtext.rm'] = 'DejaVu Sans'
#     plt.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
#     plt.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'
#     plot_para_dict.update({"figure.width":7.0866})
#     return
#
#
# def generate_color_palette(n_colors):
#     # 使用numpy生成均匀分布的颜色
#     colormap = plt.get_cmap("Accent")
#     colors = colormap(np.linspace(0, 1, n_colors))
#     return colors
#
#
# academic_mode()
# # draft_mode()
#
#
# class PlotTemplate:
#     def __init__(self, dpi=360, figsize=(7.0866,4.3795188), mode="academic_mode"):
#         self.dpi = dpi
#         self.figsize = figsize
#         self.paras = {"dpi":dpi,
#                       "figsize":figsize}
#         self.tmp_para = {}
#         if mode == "academic_mode":
#             self.mode = academic_mode
#         elif mode == "draft_mode":
#             self.mode = draft_mode
#
#     def temp_fig(self,size_scale=(0.5,0.5), dpi=None, figsize=None, **kwargs):
#         kwargs = self.ex_tmp(kwargs)
#         dpi = self._pick_last(self.dpi, kwargs["dpi"], dpi)
#         raw_figsize = self._pick_last(self.figsize, kwargs["figsize"], figsize)
#         figsize = (raw_figsize[0]*size_scale[0], raw_figsize[1]*size_scale[1])
#         return {"dpi":dpi,
#                 "figsize":figsize,
#                 }
#
#     def __call__(self, *args, **kwargs):
#         self.tmp_para = kwargs
#         return self
#
#     def ex_tmp(self, args_dict):
#         self.mode()
#         # args_dict.update(self.tmp_para)
#         return {**self.paras, **args_dict, **self.tmp_para}
#
#     def set_para(self,**kwargs):
#         for k,v in kwargs.items():
#             self.__setattr__(k,v)
#
#     def summary(self):
#         rtn_str="summary:\n"
#         for k,v in self.paras.items():
#             rtn_str=rtn_str+"\t"+k+": "+str(v)+"\n"
#         return rtn_str
#
#     @staticmethod
#     def _pick_last(*args):
#         for arg in reversed(args):
#             if arg is not None:
#                 return arg
#
# plot_template = PlotTemplate()
#
