#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/19 17:11
# @Author  : Oliver
# @File    : painter_format.py
# @Software: PyCharm
"""
绘图格式管理器
"""

import colorsys
from matplotlib import pyplot as plt
import numpy as np
import random
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# plot_para_dict = {"figure.width": 7.0866,
#                   "legend.fontsize": 5}

def draft_mode():
    plt.rcdefaults()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["text.usetex"] = True


def academic_mode():
    plt.rcdefaults()
    plt.rcParams['svg.fonttype'] = 'none'  # 保留文本为文本对象，不嵌入字体信息 便于编辑
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 360
    plt.rcParams['figure.figsize'] = [7.0866, 5.67]
    # plt.rcParams['figure.figsize'] = [7.0866, 4.3795188]
    # 设置标准文本标注字体为 5 - 7 磅的无衬线字体
    # 这里设置字体大小为 6pt，字体族为无衬线字体
    plt.rcParams['font.size'] = 5
    plt.rcParams['legend.fontsize'] = 5
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    # plt.rcParams["text.usetex"] = True
    # plt.rcParams["font.cursive"] = ["DejaVu Sans", "sans-serif"]

    # 设置希腊字符字体为 Symbol 字体
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.default'] = 'it'
    # 修改为系统支持的无衬线字体名称
    plt.rcParams['mathtext.rm'] = 'DejaVu Sans'
    plt.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
    plt.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'
    plt.rcParams['figure.subplot.bottom'] = 0.20

    # plot_para_dict.update({"figure.width": 7.0866})
    return


academic_mode()


def hex_to_rgb(hex_color):
    """将HEX颜色转换为归一化的RGB三元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    """将归一化的RGB值转换为HEX颜色"""
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def adjust_color(hex_color, light_factor=1.15, saturation_factor=1.25, max_s=0.9):
    """
    智能颜色调整函数
    :param hex_color: 原始颜色
    :param light_factor: 亮度提升系数 (建议1.1-1.3)
    :param saturation_factor: 饱和度提升系数 (建议1.2-1.5)
    :param max_s: 最大饱和度限制 (保持颜色专业感)
    :return: 调整后的HEX颜色
    """
    # 转换为HSL
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # 动态调整策略
    orig_l = l
    new_l = min(0.95, l * light_factor)  # 防止过曝
    s_boost = (1 - orig_l) * 0.4  # 根据原始亮度动态计算饱和度增量
    new_s = min(max_s, s * saturation_factor + s_boost)

    # 转换回RGB
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, new_s)
    return rgb_to_hex(new_r, new_g, new_b)


#
#
# class ItemName:
#     def __init__(self, program_name:str, nick_name:str, file_name=None, written_name=None, legend_name=None, label_name=None, ):
#         self.program_name = program_name
#         self.nick_name = nick_name
#         self.file_name = file_name if file_name is not None else  program_name
#         self.written_name = written_name if written_name is not None else program_name.replace("_", " ")
#         self.legend_name = legend_name  if legend_name is not None else nick_name
#         self.label_name = label_name if label_name is not None else nick_name
#
#     def todict(self):
#         return {self.program_name:{"nick_name":self.nick_name, "file_name":self.file_name,
#                                    "written_name":self.written_name, "legend_name":self.legend_name,
#                                    "label_name":self.label_name}}
#
#     def transform(self, mode="nick"):
#         if mode == "program":
#             return self.program_name
#         if mode == "nick":
#             return self.nick_name
#         if mode == "file":
#             return self.file_name
#         if mode == "written":
#             return self.written_name
#         if mode == "legend":
#             return self.legend_name
#         if mode == "label":
#             return self.label_name
#         return self.program_name+ "_" + mode
#
# class NameTransform:
#     def __init__(self, items:list[ItemName]=None):
#         self._map = {}
#         if items is not None:
#             for item in items:
#                 self.add_item(item)
#
#     def add_item(self,item:ItemName):
#         """
#         主键是program_name, nickname如果有重复会将过去的nickname的映射也删除
#         """
#         self._map[item.program_name] = item
#         if item.nick_name not in self._map:
#             self._map[item.nick_name] = item
#         else:
#             self._map.pop(item.nick_name)
#
#     def transform(self, base_name, mode="program"):
#         if base_name in self._map:
#             return self._map[base_name].transform(mode)
#         else:
#             return None

class PlotTemplate:
    # plot_para_dict = plot_para_dict

    # generate_color_palette = generate_color_palette
    def __init__(self,mode="academic_mode", dpi=None, figsize=None, name_items=None, **kwargs):
        if mode == "academic_mode":
            self.basal_mode = academic_mode
        elif mode == "draft_mode":
            self.basal_mode = draft_mode
        else:
            self.basal_mode = academic_mode
        self.basal_mode()

        self.dpi = dpi if dpi is not None else plt.rcParams['figure.dpi']
        self.figsize = figsize if figsize is not None else plt.rcParams['figure.figsize']
        # self.color_seed = color_seed
        # random.seed(self.color_seed)
        # self.color_random_float = random.random()
        self.params = {"dpi": self.dpi,
                       "figsize": self.figsize,
                       "mode": mode,
                       "scatter.size.default": 50,
                       "scatter.size.min": 10,
                       "scatter.size.max": 160,
                       "scatter.text.fontsize": 6,
                       "figure.width_scale": 0.5,
                       "figure.height_scale": 0.5,
                       "fontsize.label": 7,
                       "fontsize.bar_top": 5,
                       "fontsize.legend":5,
                       "fontsize.legend.title":6,
                       **kwargs}

        # self.name_transform = NameTransform(name_items)
        self.default_params = self.params.copy()
        self.buffer_palette = []
        self.buffer_palette_rgb=[]
        self.style = "ggplot"
        # self.style = "petroff10"
        # self.style = "default"

    def set_params(self, kwargs):
        self.params.update(kwargs)
        for k,v in kwargs.items():
            if k in plt.rcParams:
                plt.rcParams[k] = v

    def set_params_default(self):
        self.params.clear()
        self.params.update(self.default_params)
        self.basal_mode()

    def temp_fig(self, width_scale=None, height_scale=None, dpi=None, figsize=None):
        """
        函数直接给出的参数 > self.params > params未定义的（不建议为定义
        """

        width_scale = width_scale if width_scale is not None else self.params["figure.width_scale"]
        height_scale = height_scale if height_scale is not None else self.params["figure.height_scale"]
        dpi = dpi if dpi is not None else self.params["dpi"]
        figsize = figsize if figsize is not None else self.params["figsize"]

        rtn_figsize = (figsize[0] * width_scale, figsize[1] * height_scale)

        return {"dpi": dpi,
                "figsize": rtn_figsize,
                }

    def get_fig(self, width_scale=None, height_scale=None, dpi=None, figsize=None) -> tuple[Figure, Axes]:
        """创建预配置的图表和坐标轴对象"""
        # 应用样式
        # plt.style.use(self.style)

        # width_scale = width_scale if width_scale is not None else self.params["figure.width_scale"]
        # height_scale = height_scale if height_scale is not None else self.params["figure.height_scale"]
        # dpi = dpi if dpi is not None else self.params["dpi"]
        # figsize = figsize if figsize is not None else self.params["figsize"]
        # rtn_figsize = (figsize[0] * width_scale, figsize[1] * height_scale)
        args = self.temp_fig(width_scale, height_scale, dpi, figsize)

        # 创建图表和坐标轴
        fig, ax = plt.subplots(figsize=args["figsize"], dpi=args["dpi"])
        # plt.subplots_adjust(bottom=0.20)
        # # 设置通用样式
        # ax.grid(True, linestyle='--', alpha=0.7)
        # ax.tick_params(axis='both', which='major', labelsize=10)
        # ax.spines[['top', 'right']].set_visible(False)

        return fig, ax

    def __call__(self, **kwargs):
        # self.tmp_para = kwargs
        return PlotTemplate(**{**self.params, **kwargs})

    def __getitem__(self, item):
        if item in self.params:
            return self.params[item]
        else:
            return None

    # def ex_tmp(self, args_dict):
    #     self.basal_mode()
    #     # args_dict.update(self.tmp_para)
    #     return {**self.params, **args_dict}

    # def set_para(self, **kwargs):
    #     for k, v in kwargs.items():
    #         self.__setattr__(k, v)

    def summary(self):
        rtn_str = "summary:\n"
        for k, v in self.params.items():
            rtn_str = rtn_str + "\t" + k + ": " + str(v) + "\n"
        return rtn_str

    def adopt_scatter_size(self, arr, log=False):
        if log:
            arr = np.log(arr)
        arr_max = arr.max()
        arr_min = arr.min()
        return self["scatter.size.min"] + (arr - arr_min) * (self["scatter.size.max"] - self["scatter.size.min"]) / (
                arr_max - arr_min)

    # @staticmethod
    # def _pick_last(*args):
    #     for arg in reversed(args):
    #         if arg is not None:
    #             return arg


    def color_palette(self, n=30, highlight=None,  highlight_mode='balanced', c_offset=5, offset=1, output_fmt="str"):
        """output_fmt: str or rgb 分别对应 #rrggbb 和浮点数组（group）
        """
        # manual_color = [
        #     '#C7B299', '#6A93B7', '#8CAA9D', '#D4A5A5','#4B8F8C',
        #     '#7EB2D3',  '#B3C4A8', '#9F8BB5', '#88BEDC',
        #     '#D8BFA9', '#729E7E', '#C89C8E', '#5F97A6', '#B9AEC6',
        #     '#A3C3B8', '#D6C3DC', '#8DA7C4', '#C2D6B5', '#E39A98', '#A88C7A'
        #     ]
        # def smart_highlight(hex_color, mode=highlight_mode):
        #     """
        #     预设模式调整
        #     :param mode:
        #         'balanced' - 平衡模式 (默认)
        #         'strong' - 强烈对比
        #         'soft' - 柔和变化
        #     """
        #     presets = {
        #         'balanced': (1.2, 1.3),
        #         'strong': (1.3, 1.5),
        #         'soft': (1.1, 1.2)
        #     }
        #     return adjust_color(hex_color, *presets[mode])

        def hsl_to_rgb(h, s, l):
            # 扩展色生成函数（正确HSL转RGB）
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return (r, g, b), f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
        if n >= len(self.buffer_palette):
            extend_colors = []
            extend_colors_rgb = []
            for i in range(len(self.buffer_palette), n):
                j=i+offset
                # 使用黄金分割比生成色相序列
                # hue = (0.40 + i * 0.618033988749895 * 2 ) % 1.0  # φ = (√5 + 1)/2 ≈ 0.618
                hue = (0.40 + j * 0.618033988749895 *5/7 ) % 1.0  # φ = (√5 + 1)/2 ≈ 0.618
                # 饱和度波动在25%-45%之间
                sat = 0.35 + 0.15 * np.sin(j * 0.7 / 2 + 1.57)
                # 明度保持在55%-75%之间
                lum = 0.65 + 0.10 * np.cos(j * 0.2 + 0.7853981633974483)
                color_rgb, color_str = hsl_to_rgb(hue, sat, lum)
                extend_colors.append(color_str)
                extend_colors_rgb.append(color_rgb)
            self.buffer_palette.extend(extend_colors)
            self.buffer_palette_rgb.extend(extend_colors_rgb)

        if output_fmt == "str":
            rtn = self.buffer_palette[:n]
        else:
            rtn = self.buffer_palette_rgb[:n]
        if c_offset !=0:
            offset = c_offset % n
            rtn =  rtn[-offset:] + rtn[:-offset]
        return rtn[:n]

    def get_buffer_palette(self):
        return self.buffer_palette

    # @staticmethod
    # def add_frame_counter(fig=None, ax=None, text="Frame: 0",
    #                       fontsize=5, color='black', alpha=0.7,
    #                       backgroundcolor='white', pad=5):
    #     """
    #     在图表的右上角添加一个帧数计数器或自定义文本
    #
    #     参数:
    #         fig: matplotlib.figure.Figure对象，默认为当前figure
    #         ax: matplotlib.axes.Axes对象，默认为当前axes
    #         text: 要显示的文本内容，默认为"Frame: 0"
    #         fontsize: 文本字体大小
    #         color: 文本颜色
    #         alpha: 文本和背景的透明度
    #         backgroundcolor: 文本背景颜色
    #         pad: 文本与边框的间距
    #     """
    #     # 获取当前的figure和axes
    #     if fig is None:
    #         fig = plt.gcf()
    #     if ax is None:
    #         ax = plt.gca()
    #
    #     # 使用annotate函数添加文本，设置为figure坐标系统
    #     # xy=(1, 1)表示右上角，xycoords='axes fraction'表示使用axes的相对坐标
    #     # xytext=(0, 0)和textcoords='offset points'结合pad参数设置文本偏移
    #     ax.annotate(text,
    #                 xy=(1, 1), xycoords='axes fraction',
    #                 xytext=(-pad, -pad), textcoords='offset points',
    #                 ha='right', va='top',
    #                 fontsize=fontsize,
    #                 color=color,
    #                 bbox=dict(boxstyle="round,pad=0.3",
    #                           fc=backgroundcolor,
    #                           ec='none',
    #                           alpha=alpha),
    #                 zorder=100  # 确保文本显示在最上层
    #                 )


    @staticmethod
    def add_frame_counter(fig=None, ax=None, text="Frame: 0",
                          fontsize=6, color='black', alpha=0.7,
                          backgroundcolor='white', pad=5):
        """
        在图表的右上角添加帧数计数器（优先使用ax，次优使用fig）

        参数逻辑:
            1. 如果ax有效，直接使用
            2. 如果ax为None但fig有效，创建虚拟ax
            3. 都未指定时使用plt.gcf()和plt.gca()
        """
        # 情况1：ax有效（无论fig是否None）
        if ax is not None:
            current_ax = ax
            current_fig = ax.figure
        # 情况2：ax为None但fig有效
        elif fig is not None:
            current_fig = fig
            # 创建透明虚拟ax（不干扰主图）
            current_ax = current_fig.add_axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
            current_ax.set_axis_off()
        # 情况3：都未指定
        else:
            current_fig = plt.gcf()
            current_ax = plt.gca()

        # 添加计数器文本（始终使用当前ax的坐标系）
        current_ax.annotate(
            text,
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-pad, -pad), textcoords='offset points',
            ha='right', va='top',
            fontsize=fontsize,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc=backgroundcolor,
                ec='none',
                alpha=alpha
            ),
            zorder=100  # 确保显示在最上层
        )

        # 返回当前使用的ax（便于后续操作）
        return current_ax

plot_template = PlotTemplate()
