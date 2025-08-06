#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/16 15:15
# @Author  : Oliver
# @File    : correlation.py
# @Software: PyCharm
from sys import stdout

import pandas as pd
from jupyter_server.serverapp import flags
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors  # 添加这行
from matplotlib.pyplot import title

from ...painter_format import PlotTemplate, plot_template
import numpy as np
import os

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from ...reporter import Reporter
from sklearn.manifold import MDS
import networkx as nx
import matplotlib.cm as cm
from matplotlib.lines import Line2D

def pearson_calculate(rule_firing_levels):
    n = rule_firing_levels.shape[-1]
    if n==1:
        return np.ones([1,1]), 1., 1.
    corr_matrix = np.corrcoef(rule_firing_levels, rowvar=False)
    # n = len(corr_matrix)
    avg_pearson = np.mean(np.abs(corr_matrix))
    avg_pearson_non_diag = (avg_pearson * n**2 - n) / (n**2 - n)
    return corr_matrix, avg_pearson, avg_pearson_non_diag

def rule_pearson(rule_firing_levels, reporter:Reporter, plot_temp:PlotTemplate, info:str):
    corr_matrix, avg_pearson, avg_pearson_non_diag = pearson_calculate(rule_firing_levels)
    n = len(corr_matrix)

    fig= plt.figure(**plot_temp.temp_fig(0.5,0.5))
    # 创建自定义色图
    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                           ['#FF0000', '#FFFFFF', '#00A1FF'],
                                           N=256)

    # 绘制热力图
    im = plt.imshow(corr_matrix, cmap=cmap, vmax=1, vmin=-1,extent=(0.5, n+0.5, 0.5, n+0.5), origin='lower')

    # 添加colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Pearson Correlation ', rotation=270, labelpad=10)

    # 设置标题和坐标轴
    # plt.title(f'{n}D-Pearson Correlate', fontsize=plot_temp.params["fontsize.label"])
    tick_step = max(1, n // 8)  # 显示大约8个ticks
    plt.xticks(range(1, n+1, tick_step), fontsize=plot_temp.params["fontsize.label"])
    plt.yticks(range(1, n+1, tick_step), fontsize=plot_temp.params["fontsize.label"])
    plt.xlabel('Rule id', fontsize=plot_temp.params["fontsize.label"])
    plt.ylabel('Rule id', fontsize=plot_temp.params["fontsize.label"])
    plt.tight_layout()

    # 保存图像
    path = reporter.add_figure_by_data(fig,f'correlation_heatmap_mpl_{info}',title="correlation heatmap",
                                       describe=f"avg_pearson = {avg_pearson};\n avg_pearson_non_diag = {avg_pearson_non_diag}")
    plt.close(fig)
    return path



def pearson2force_directed(rule_firing_levels, reporter:Reporter, plot_temp:PlotTemplate, info:str,
                           node_legend=False, edge_display_weight_threshold=0.2):
    corr_matrix, avg_pearson, avg_pearson_non_diag = pearson_calculate(rule_firing_levels)
    num_vars = len(corr_matrix)
    variables = [f'Rule{i + 1:02d}' for i in range(num_vars)]
    colors = plot_temp.color_palette(n=num_vars)

    # 创建图结构
    G = nx.Graph()

    # 添加节点
    firing_level = rule_firing_levels / np.sum(rule_firing_levels,axis=1, keepdims=True)
    rule_power = np.mean(firing_level,axis=0)
    # print(rule_power/ max(rule_power))
    for i, var in enumerate(variables):
        # 节点大小基于该节点被调用的程度

        node_size = 1 + 199 * rule_power[i] / max(rule_power)
        G.add_node(var, size=node_size, color=colors[i])

    # 添加边（过滤弱相关）
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            weight = corr_matrix[i, j]
            if abs(weight) > edge_display_weight_threshold:  # 只显示显著相关
                edge_width = 0.2 + 2 * abs(weight)  # 线宽反映相关强度
                edge_style = 'solid' if weight > 0 else 'dashed'
                G.add_edge(variables[i], variables[j],
                           # weight=weight * abs(weight),
                           weight=weight,
                           width=edge_width,
                           style=edge_style,
                           color='#1f77b4' if weight > 0 else '#d62728')

    # 力导向布局 - 优化参数
    pos = nx.spring_layout(G,
                           k=0.25,  # 节点间距
                           iterations=200,  # 确保收敛
                           seed=42)

    # 创建图形
    fig = plt.figure(**plot_temp.temp_fig(1,1))

    # 绘制节点
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=1,
                           alpha=0.9)

    # 绘制边
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=data['width'],
                               edge_color=data['color'],
                               alpha=0.5,
                               style=data['style'])
    # 创建图例元素
    legend_elements = []
    if node_legend and num_vars<=18:    # 点数太多时不处理
        for i, var in enumerate(variables):
            legend_elements.append(Line2D([0], [0],
                                          marker='o',
                                          color='w',
                                          markerfacecolor=colors[i],
                                          markersize=8,
                                          label=var
                                          ))

    # 添加边类型图例
    legend_elements.append(Line2D([0], [0],
                                  color='#1f77b4',
                                  lw=2,
                                  label='Positive Correlation'))
    legend_elements.append(Line2D([0], [0],
                                  color='#d62728',
                                  lw=2,
                                  linestyle='dashed',
                                  label='Negative Correlation'))

    # 创建图例 - 分多列显示
    legend = plt.legend(handles=legend_elements,
                        loc='center',
                        bbox_to_anchor=(0.5, -0.1),  # 放在底部中间
                        ncol=6,  # 6列布局
                        fontsize=7,
                        frameon=True,
                        framealpha=0.9,
                        # title="Variable Legend",
                        title_fontsize=10)

    # 添加标题
    # plt.title(f'Force-Directed Graph of {num_vars} Variables', fontsize=16, pad=20)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.1)

    # 隐藏坐标轴
    plt.axis('off')
    plt.gca().set_aspect("equal")  # 关键代码
    # 调整布局
    # plt.tight_layout(rect=(0., 0.1, 1., 0.95))
    plt.subplots_adjust(bottom=0.25)  # 为底部图例留出空间
    reporter.add_figure_by_data(fig,f"FD_{info}",title="FD Correlation",)
    plt.close(fig)

def combine_pearson_and_force_directed(rule_firing_levels, reporter:Reporter, plot_temp:PlotTemplate,info, frame=None,
                                       edge_display_weight_threshold=0.2):
    # 计算相关系数矩阵
    corr_matrix, avg_pearson, avg_pearson_non_diag = pearson_calculate(rule_firing_levels)
    n = len(corr_matrix)

    # 创建图形（宽度加倍以容纳两个子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, **plot_temp.temp_fig(1,0.5),
                                   gridspec_kw={'width_ratios': [1, 1.2]})

    # ========== 左侧：相关系数热力图 ==========
    # 创建自定义色图
    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                             ['#FF0000', '#FFFFFF', '#00A1FF'],
                                             N=256)
    # 绘制热力图
    im = ax1.imshow(corr_matrix, cmap=cmap, vmax=1, vmin=-1,
                    extent=(0.5, n + 0.5, 0.5, n + 0.5), origin='lower')

    # 添加colorbar
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Pearson Correlation', rotation=270, labelpad=10)

    # 设置标题和坐标轴
    tick_step = max(1, n // 8)
    ax1.set_xticks(range(1, n + 1, tick_step))
    ax1.set_yticks(range(1, n + 1, tick_step))
    ax1.set_xlabel('Rule id', fontsize=plot_temp.params["fontsize.label"])
    ax1.set_ylabel('Rule id', fontsize=plot_temp.params["fontsize.label"])
    # ax1.set_title('Correlation Matrix', fontsize=plot_temp.params["fontsize.label"])

    # ========== 右侧：力导向图 ==========
    variables = [f'Rule{i + 1:02d}' for i in range(n)]
    colors = plot_temp.color_palette(n=n)

    # 创建图结构
    G = nx.Graph()

    # 添加节点
    firing_level = rule_firing_levels / np.sum(rule_firing_levels, axis=1, keepdims=True)
    rule_power = np.mean(firing_level, axis=0)
    for i, var in enumerate(variables):
        node_size = 1 + 199 * rule_power[i] / max(rule_power)
        G.add_node(var, size=node_size, color=colors[i])

    # 添加边（过滤弱相关）
    for i in range(n):
        for j in range(i + 1, n):
            weight = corr_matrix[i, j]
            if abs(weight) > edge_display_weight_threshold:
                edge_width = 0.2 + 2 * abs(weight)
                edge_style = 'solid' if weight > 0 else 'dashed'
                G.add_edge(variables[i], variables[j],
                           weight=weight,
                           width=edge_width,
                           style=edge_style,
                           color='#1f77b4' if weight > 0 else '#d62728')

    # 力导向布局
    pos = nx.spring_layout(G, k=0.25, iterations=200, seed=42)

    # 绘制节点
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=1,
                           alpha=0.9)

    # 绘制边
    for u, v, data in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=[(u, v)],
                               width=data['width'],
                               edge_color=data['color'],
                               alpha=0.5,
                               style=data['style'])

    # 创建图例元素
    legend_elements = []
    if n <= 18:
        for i, var in enumerate(variables):
            legend_elements.append(Line2D([0], [0],
                                          marker='o',
                                          color='w',
                                          markerfacecolor=colors[i],
                                          markersize=8,
                                          label=var))

    # 添加边类型图例
    legend_elements.append(Line2D([0], [0],
                                  color='#1f77b4',
                                  lw=2,
                                  label='Positive Correlation'))
    legend_elements.append(Line2D([0], [0],
                                  color='#d62728',
                                  lw=2,
                                  linestyle='dashed',
                                  label='Negative Correlation'))

    # 创建图例
    ax2.legend(handles=legend_elements,
               loc='center',
               bbox_to_anchor=(0.5, -0.15),
               ncol=6 if n <= 18 else 1,
               fontsize=7,
               frameon=True,
               framealpha=0.9)

    # 设置图形属性
    ax2.grid(True, linestyle='--', alpha=0.1)
    ax2.axis('off')
    ax2.set_aspect("equal")
    # ax2.set_title('Force-Directed Network', fontsize=plot_temp.params["fontsize.label"])

    # 调整整体布局
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例留出空间

    plot_temp.add_frame_counter(fig, text=f"step {frame}")
    # 保存图像
    path = reporter.add_figure_by_data(
        fig,
        f'combined_pearson_fd_{info}',
        title="Combined Correlation Visualization",
        describe=f"avg_pearson = {avg_pearson:.4f};\navg_pearson_non_diag = {avg_pearson_non_diag:.4f}"
    )
    plt.close(fig)
    return path


def pearson2mds(rule_firing_levels, reporter:Reporter, plot_temp:PlotTemplate, info:str):
    corr_matrix = np.corrcoef(rule_firing_levels, rowvar=False)
    n = len(corr_matrix)
    distance_matrix = 1 - np.abs(corr_matrix)  # 距离转换

    # MDS 降维
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    # 可视化
    fig = plt.figure(**plot_temp.temp_fig())
    plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.7)

    # # 添加变量标签
    # variables = ['Var1', 'Var2', 'Var3', ...]  # 变量名称列表
    # for i, var in enumerate(variables):
    #     plt.annotate(var, (coords[i, 0], coords[i, 1]),
    #                  xytext=(5, 5), textcoords='offset points')

    # 添加辅助线表示正/负相关
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if corr_matrix[i, j] > 0.6:  # 强正相关
                plt.plot([coords[i,0], coords[j,0]],
                         [coords[i,1], coords[j,1]], 'g--', alpha=0.3)
            elif corr_matrix[i, j] < -0.6:  # 强负相关
                plt.plot([coords[i,0], coords[j,0]],
                         [coords[i,1], coords[j,1]], 'r:', alpha=0.3)

    plt.title('MDS Correlation Visualization')
    plt.grid(alpha=0.2)
    plt.gca().set_aspect("equal")  # 关键代码
    reporter.add_figure_by_data(fig,f"MDS_{info}",title="MDS Correlation",)
    plt.close(fig)









