#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 11:25
# @Author  : Oliver
# @File    : analyze_track_feature.py
# @Software: PyCharm

from config import *
from support_config_parser import ConfigManager
from frame.eval_process.painter.rule_levy_if import *
from scipy import stats

# 生成Lévy飞行轨迹的示例函数
import numpy as np
from scipy.stats import levy_stable

# # 创建虚拟报告器
# class ScientificReporter:
#     def __init__(self):
#         self.basic_path = os.path.join("output", "levy_track_sim")
#         class A:
#             def md(self):
#                 return self
#
#             def log(self):
#                 return self
#
#         self.a = A()
#     def __call__(self, msg):
#         print(f"[msg] {msg}")
#         return self.a
#
#     def add_figure_by_data(self, fig, name):
#         fig.savefig(os.path.join(self.basic_path, name+".png"), dpi=300)
#         print(f"\t保存图表: {name}")
#
# reporter = ScientificReporter()

def generate_levy_flight(num_trajectories=10, n_steps=1000, alpha=1.5, step_scale=1.0, seed=42):
    """
    生成物理合理的Lévy飞行轨迹（步长和方向解耦）

    参数:
        num_trajectories: 轨迹数量
        n_steps: 每轨迹步数
        alpha: Lévy稳定分布的特征指数 (1 < α < 2)
        step_scale: 步长缩放因子

    返回:
        List[np.ndarray]: 轨迹列表，每个轨迹形状为(n_steps, 2)
    """
    np.random.seed(seed)
    tracks = []
    for _ in range(num_trajectories):
        # 1. 生成Lévy步长（绝对正值）
        steps = np.abs(levy_stable.rvs(alpha, 0, size=n_steps, scale=step_scale))

        # 2. 生成随机方向（均匀分布）
        angles = np.random.uniform(0, 2 * np.pi, size=n_steps)

        # 3. 转换为位移向量
        dx = steps * np.cos(angles)
        dy = steps * np.sin(angles)
        displacements = np.column_stack([dx, dy])

        # 4. 计算累积位置（从原点开始）
        trajectory = np.cumsum(displacements, axis=0)
        tracks.append(trajectory)

    return tracks

def generate_random_flight(num_trajectories=10, n_steps=1000,mean=0.0, std_dev=1.0, seed=42):
    """
     生成全随机的点以作为飞行轨迹

     参数:
         num_trajectories: 轨迹数量
         n_steps: 每轨迹步数
         step_scale: 步长缩放因子

     返回:
         List[np.ndarray]: 轨迹列表，每个轨迹形状为(n_steps, 2)
     """
    np.random.seed(seed)
    tracks = []

    # 计算合理的坐标范围
    coord_range = 3 * std_dev * np.sqrt(n_steps)  # 覆盖99.7%的数据点

    for _ in range(num_trajectories):
        # 1. 在二维平面内生成正态分布的点
        x = np.random.normal(mean, std_dev, size=n_steps)
        y = np.random.normal(mean, std_dev, size=n_steps)

        # 2. 组合成轨迹
        trajectory = np.column_stack([x, y])
        tracks.append(trajectory)

    return tracks

# 生成10条轨迹，每条1000步，α=1.3

# def plot_trajectory(trajectory_list, reporter:Reporter, title="Lévy Flight", alpha=1.5):
#     fig = plt.figure(figsize=(4,4),dpi=720)
#     for trajectory in trajectory_list:
#         plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, linewidth=0.5)
#         # plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', label='Start')
#         # plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', label='End')
#         # plt.title(f"{title}\n(α={alpha}, Steps={len(trajectory)})")
#         plt.xlabel("X"); plt.ylabel("Y")
#         # plt.legend()
#         # plt.grid(True, alpha=0.3)
#         plt.axis('equal')
#         plt.axis('off')
#     # plt.show()
#     reporter.add_figure_by_data(fig, f"track_plot_{title}")




data_sec_name = "dataset_quin33_6s"
prefix = "TrackFeature" + data_sec_name
n_step = 40
max_lag = n_step - 1
if __name__ == '__main__':
    load_config()
    cfg = ConfigManager("default.ini", data_sec_name, prefix)
    reporter = cfg.root_reporter
    # 数据集轨迹
    flight_tracks = [track[cfg.data_processed.columns].to_numpy()[:,:2] for track in cfg.data_original]
    # # 创建模拟数据
    levy_tracks = generate_levy_flight(num_trajectories=64, n_steps=n_step,alpha=1.6)
    brownian_tracks = generate_levy_flight(num_trajectories=64, n_steps=n_step,alpha=2.0)  # α=2 是高斯分布(布朗运动)
    random_track = generate_random_flight(num_trajectories=64, n_steps=n_step, mean=0.0, std_dev=1.0)

    # reporter("plotting tracks")
    # plot_trajectory(levy_tracks,reporter, title="Levy")
    # plot_trajectory(brownian_tracks,reporter, "brownian")
    # plot_trajectory(random_track,reporter, "randn")
    # plot_trajectory(flight_tracks,reporter, "flight")

    # 分析扩散模式
    kurt_b, type_b, _ = analyze_diffusion(brownian_tracks, reporter, PlotTemplate(), "brownian",max_lag=max_lag, n_lag=200,)
    kurt_l, type_l, _ = analyze_diffusion(levy_tracks, reporter, PlotTemplate(), "levy",max_lag=max_lag, n_lag=200,)
    kurt_r, type_r, _ = analyze_diffusion(random_track, reporter, PlotTemplate(), "random",max_lag=max_lag, n_lag=200,)
    kurt_f, type_f, _ = analyze_diffusion(flight_tracks, reporter, PlotTemplate(), "flight",max_lag=max_lag, n_lag=200,)

    print(f"布朗运动: alpha={type_b}, 峰度={kurt_b:.1f}")
    print(f"莱维飞行: alpha={type_l}, 峰度={kurt_l:.1f}")
    print(f"随机分布: alpha={type_r}, 峰度={kurt_r:.1f}")
    print(f"飞机轨迹: alpha={type_f}, 峰度={kurt_f:.1f}")








