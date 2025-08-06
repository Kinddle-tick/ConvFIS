#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/16 15:15
# @Author  : Oliver
# @File    : rule_levy_if.py
# @Software: PyCharm
from collections import OrderedDict
from sys import stdout

import pandas as pd
from jupyter_server.serverapp import flags
from matplotlib.collections import LineCollection

from ...painter_format import PlotTemplate, plot_template
import numpy as np
import os

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from ...reporter import Reporter
from scipy.stats import expon, rayleigh, halfnorm
from scipy.stats import kurtosis, linregress
from scipy.stats import levy_stable

def distribute_difference_length(tracks, reporter:Reporter, plot_temp:PlotTemplate, info:str,
                                  fit_line=(), dims=(0,1), log_scale=False):
    """
    Input several trajectories, compute the distribution of step lengths between consecutive points.
    Optionally fit an exponential or Rayleigh distribution to the data.

    :param tracks:      Trajectories [N, T，state], where N is the number of trajectories,
                        state is the state vector, and T is the time dimension.
    :param reporter:    Callable reporter for outputting information.
    :param plot_temp:   Plot template for creating figures.
    :param info:        Additional info for naming the output plot.
    :param fit_line:    Type of distribution to fit ("exponential", "rayleigh", "halfnormal"). can use ("all") to try all
    :param dims:        Dimensions of the state vector to use for distance calculation.
    :param log_scale:       If True, using log y.
    :return:            None
    """
    all_step_lengths = []

    # Iterate over each trajectory
    for traj in tracks:
        # Extract the specified dimensions
        traj_subset = traj[..., dims]  # Shape: [T, len(dims)]
        # Compute differences between consecutive points
        diffs = np.diff(traj_subset, axis=0)  # Shape: [T-1, len(dims)]
        # Compute Euclidean distances (step lengths)
        distances = np.linalg.norm(diffs, axis=1)  # Shape: [T-1]
        all_step_lengths.extend(distances.tolist())

    all_step_lengths = np.array(all_step_lengths)

    # Report basic statistics
    reporter(f"\tComputed {len(all_step_lengths)} step lengths.")
    reporter(f"\tMean step length: {np.mean(all_step_lengths):.4f}")
    reporter(f"\tStandard deviation: {np.std(all_step_lengths):.4f}")

    # Create plot
    # fig, ax = plot_temp.get_fig()
    fig, ax = plt.subplots(**plot_temp.temp_fig())
    # Plot histogram (normalized to density)
    counts, bins, _ = ax.hist(all_step_lengths, bins=100, density=True, alpha=0.7, label='Step Lengths')

    # Fit specified distribution and plot PDF
    if len(fit_line):
        fit_line = [x.lower() for x in fit_line]
        x = np.linspace(bins[0], bins[-1], 1000)
        if "exponential" in fit_line or "all" in fit_line:
            # Fit exponential distribution (scale = 1/lambda)
            loc, scale = expon.fit(all_step_lengths)
            pdf = expon.pdf(x, loc, scale)

            ax.plot(x, pdf, '-', linewidth=1, label='Exponential Fit')
            reporter(f"Exponential fit: loc={loc:.4f}, scale={scale:.4f}").md().log()
        if "rayleigh" in fit_line or "all" in fit_line:
            # Fit Rayleigh distribution (scale = sigma)
            loc, scale = rayleigh.fit(all_step_lengths)
            pdf = rayleigh.pdf(x, loc, scale)
            ax.plot(x, pdf, '-', linewidth=1, label='Rayleigh Fit')
            reporter(f"Rayleigh fit: loc={loc:.4f}, scale={scale:.4f}").md().log()
        if "halfnormal" in fit_line or "all" in fit_line:
            loc, scale = halfnorm.fit(all_step_lengths)
            pdf = halfnorm.pdf(x, loc, scale)
            ax.plot(x, pdf, '-', linewidth=1, label='Half-Normal Fit')
            reporter(f"halfnormal fit: loc={loc:.4f}, scale={scale:.4f}").md().log()
        # else:
        #     reporter(f"Unsupported fit_line option: {fit_line}. Skipping fit.")

    ax.set_title(f"Distribution of Step Lengths ({info})")
    ax.set_xlabel("Step Length")
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel("Logarithmic Density")
    else:
        ax.set_ylabel("Density")
    ax.legend()

    # Save the figure
    if log_scale:
        info +="_log"
    reporter.add_figure_by_data(fig, f"step_length_distribution_{info}",f"Distribute of {info}")
    # reporter(f"Saved plot as 'step_length_distribution_{info}.png'")


def levy_msd(tracks, reporter: Reporter, plot_temp: PlotTemplate, info: str, dims=(0, 1)):
    """
    Enhanced MSD analysis with:
    1. Correct reference line alignment
    2. Explicit diffusion type classification
    3. Lévy flight detection
    """
    # Convert input to numpy arrays
    tracks = [np.asarray(track) for track in tracks]

    # 1. Calculate MSD with proper time alignment
    max_length = max(track.shape[0] for track in tracks)
    msd_by_time = np.zeros(max_length - 1)
    count_by_time = np.zeros(max_length - 1, dtype=int)

    for track in tracks:
        positions = track[..., dims]
        length = positions.shape[0]

        for delta_t in range(1, length):
            displacements = positions[delta_t:] - positions[:-delta_t]
            squared_displacements = np.sum(displacements ** 2, axis=1)

            if delta_t < max_length:
                msd_by_time[delta_t - 1] += np.mean(squared_displacements)
                count_by_time[delta_t - 1] += 1

    # Process MSD data
    valid_mask = count_by_time > 0
    time_points = np.arange(1, max_length)[valid_mask]
    avg_msd = msd_by_time[valid_mask] / count_by_time[valid_mask]

    # 2. Power law fitting and type classification
    def power_law(t, alpha, A):
        return A * (t ** alpha)

    try:
        log_t = np.log(time_points)
        log_msd = np.log(avg_msd)
        coeffs = np.polyfit(log_t, log_msd, 1)
        alpha_fit = coeffs[0]
        A_fit = np.exp(coeffs[1])
        residuals = log_msd - (alpha_fit * log_t + coeffs[1])
        r_squared = 1 - np.var(residuals) / np.var(log_msd)

        # --- Diffusion Type Classification ---
        diffusion_type = "unknown"
        if 0.9 < alpha_fit < 1.1:
            diffusion_type = "normal"
            type_comment = "Brownian motion (α ≈ 1)"
        elif alpha_fit < 0.9:
            diffusion_type = "subdiffusion"
            type_comment = f"Crowded/constrained (α = {alpha_fit:.2f} < 1)"
        elif alpha_fit > 1.1:
            if 1 < alpha_fit < 2:
                diffusion_type = "levy"
                mu_est = 3 - alpha_fit  # Lévy exponent
                type_comment = f"Lévy flight (α = {alpha_fit:.2f}, μ ≈ {mu_est:.2f})"
            elif alpha_fit >= 2:
                diffusion_type = "ballistic"
                type_comment = f"Directed motion (α ≥ 2)"
            else:
                diffusion_type = "superdiffusion"
                type_comment = f"Active transport (α = {alpha_fit:.2f} > 1)"

        reporter(f"MSD = {A_fit:.3f} × t^{alpha_fit:.3f} (R²={r_squared:.3f})")
        reporter(f"→ Classification: {type_comment}").log().md()

    except Exception as e:
        reporter(f"Fitting error: {str(e)}")
        alpha_fit, A_fit, r_squared = 1.0, 1.0, 0.0
        diffusion_type = "error"

    # 3. Visualization with aligned references
    fig, ax = plt.subplots(**plot_temp.temp_fig())

    # Plot data and fit
    ax.loglog(time_points, avg_msd, 'bo', markersize=4, label='MSD data')
    fit_curve = power_law(time_points, alpha_fit, A_fit)
    ax.loglog(time_points, fit_curve, 'r-', label=f'Fit (α={alpha_fit:.2f})')

    # Dynamic reference lines (aligned to data start)
    ref_start = time_points[0]
    ref_end = time_points[-1]
    ref_t = np.linspace(ref_start, ref_end, 50)
    ref_scale = avg_msd[0] / (ref_start ** 1.0)  # Normalize to α=1 line

    ax.loglog(ref_t, ref_scale * (ref_t ** 1.0), 'g--', label='Normal (α=1)')
    ax.loglog(ref_t, ref_scale * (ref_t ** 0.5), 'm--', label='Subdiffusion (α=0.5)')
    ax.loglog(ref_t, ref_scale * (ref_t ** 1.5), 'c--', label='Superdiffusion (α=1.5)')

    ax.set_title(f"{info}\nMSD ∝ t^{alpha_fit:.2f} ({diffusion_type})")
    ax.set_xlabel("Time lag ∆t")
    ax.set_ylabel("MSD")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    reporter.add_figure_by_data(fig, f"msd_analysis_{info}")

    return {
        'alpha': alpha_fit,
        'amplitude': A_fit,
        'r_squared': r_squared,
        'diffusion_type': diffusion_type,
        'type_comment': type_comment if 'type_comment' in locals() else None,
        'time_points': time_points,          # 时间滞后数组∆t
        'msd_values': avg_msd,              # 对应的MSD值数组
        'displacements': displacements      # 所有位移向量(N,2)
    }


def analyze_diffusion(trajectories, reporter: Reporter, plot_temp: PlotTemplate, info: str,
                      max_lag=1000, n_lag=None, bins=50, log_x=False, log_y=False):
    """
    分析扩散轨迹的扩散类型

    参数:
        trajectories: 轨迹列表，每个轨迹形状为(n_steps, 2)
        max_lag: MSD分析的最大时间延迟
        bins: 位移分布直方图的分箱数

    返回:
        fig1: 位移分布图
        fig2: MSD分析图
        kurtosis_value: 原始步长的峰度值
        diffusion_type: 扩散类型判断
    """
    if not trajectories:
        raise ValueError("轨迹列表不能为空")

    # 3. 峰度分析
    kurtosis_value, excess_kurtosis = _analyze_kurtosis(trajectories)

    path1 = _plot_trajectory(trajectories,reporter,plot_temp,info)

    # 1. 准备位移分布图
    fig1, ax1 = plt.subplots(**plot_temp.temp_fig())

    # 选择不同的时间间隔
    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lags)))

    for i, lag in enumerate(lags):
        all_displacements = []

        for traj in trajectories:
            if len(traj) > lag:
                # 计算指定时间间隔的位移
                displacements = traj[lag:] - traj[:-lag]
                radial = np.linalg.norm(displacements, axis=1)
                all_displacements.append(radial)

        if all_displacements:
            all_displacements = np.concatenate(all_displacements)

            # 计算直方图
            hist, bin_edges = np.histogram(all_displacements, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 过滤零值以在对数图上显示
            mask = hist > 0
            ax1.plot(bin_centers[mask], hist[mask], '-',
                     color=colors[i], alpha=0.7,
                     label=f'Δt={lag}')
    if log_x:
        ax1.set_xscale('log')
        ax1.set_xlabel('Displace Δx (log scale)')
    else:
        ax1.set_xlabel('Displace Δx')
    if log_y:
        ax1.set_yscale('log')
        ax1.set_ylabel('p.d.f. (log scale)')
    else:
        ax1.set_ylabel('p.d.f.')
    # ax1.set_title('Displacement distribution at different time intervals')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    fig1.suptitle(f"Displacement distribution analysis of {info} | Original step size kurtosis: {kurtosis_value:.1f}", fontsize=7)

    path2 = reporter.add_figure_by_data(fig1,f"diffusion_analyze_{info}_displace_distribution", title=f"Diffusion analyze of {info}")


    # 2. MSD分析
    fig2, ax2 = plt.subplots(1, 1, **plot_temp.temp_fig())
    fig3, ax3 = plt.subplots(1, 1, **plot_temp.temp_fig())

    # 计算MSD
    lags, msd, std = _calculate_msd(trajectories, n_lags=n_lag, max_lag=max_lag)
    # lags = np.arange(1, len(msd) + 1)

    # 线性坐标图
    # ax2.errorbar(lags, msd, yerr=std, fmt='-', alpha=0.7)
    ax2.errorbar(lags, msd, fmt='-', alpha=0.7)
    ax2.set_xlabel('Time interval (Δt)')
    ax2.set_ylabel('MSD')
    ax2.set_title('MSD-Δt')
    ax2.grid(True)

    # 对数坐标图
    mask = (msd > 0) & (lags > 0)
    log_msd = np.log(msd[mask])
    log_lags = np.log(lags[mask])

    ax3.plot(log_lags, log_msd, 'o-', alpha=0.5)
    ax3.set_xlabel('ln(Δt)')
    ax3.set_ylabel('ln(MSD)')
    ax3.set_title('ln MSD-Δt')
    ax3.grid(True)

    # 拟合幂律指数
    if len(log_lags) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_msd)
        ax3.plot(log_lags, slope * log_lags + intercept, 'r--',
                 label=f'fit <y=x$^α$>: α={slope:.2f}, R²={r_value ** 2:.2f}')
        ax3.legend()
        alpha = slope
        if r_value < 0.95 or True:
            slope, intercept, r_value, p_value, std_err = linregress(lags[mask], log_msd)
            ax3.plot(log_lags, slope * np.exp(log_lags) + intercept, '--', color='#FF7F0E',
                     label=f'fit<y=α$^x$>: α={slope:.2f}, R²={r_value ** 2:.2f}')
            ax3.legend()
    else:
        alpha = None


    # # 4. 扩散类型判断
    # if slope < 0.9:
    #     diffusion_type = "亚扩散 (subdiffusion)"
    # elif 0.9 <= slope <= 1.1:
    #     diffusion_type = "正常扩散 (Brownian motion)"
    # elif slope > 1.1:
    #     if excess_kurtosis > 5:  # 显著的重尾特征
    #         diffusion_type = "莱维飞行 (Lévy flight)"
    #     else:
    #         diffusion_type = "超扩散 (superdiffusion)"
    # else:
    #     diffusion_type = "混合扩散类型"

    # 添加分析结果到图表
    # fig2.suptitle(f"MSD analyze of {info}  | slope: {alpha:.1f}", fontsize=7)

    plt.tight_layout()
    path3 = reporter.add_figure_by_data(fig2,f"diffusion_analyze_MSD_{info}")
    path4 = reporter.add_figure_by_data(fig3,f"diffusion_analyze_MSD_ln_{info}")

    reporter(f"[+] -'{info}' MSD alpha={alpha:.1f}; kurtosis={kurtosis_value:.1f}").md().log()

    # plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return  kurtosis_value, alpha, [path1, path2, path3,path4]

# def combine_analyze_diffusion(trajectories, reporter: Reporter, plot_temp: PlotTemplate, info: str,
#                       max_lag=1000, n_lag=None, bins=50, log_x=False, log_y=False):
#     """
#     分析扩散轨迹的扩散类型
#
#     参数:
#         trajectories: 轨迹列表，每个轨迹形状为(n_steps, 2)
#         max_lag: MSD分析的最大时间延迟
#         bins: 位移分布直方图的分箱数
#
#     返回:
#         fig: 由四张子图组成
#             fig1: 位移分布图
#             fig2: MSD-t分析图
#             fig3: log(MSD)-log(t)分析图
#             fig4: log(MSD)-t分析图
#         kurtosis_value: 原始步长的峰度值
#         diffusion_type: 扩散类型判断
#     """
#     if not trajectories:
#         raise ValueError("轨迹列表不能为空")
#
#     # 3. 峰度分析
#     kurtosis_value, excess_kurtosis = _analyze_kurtosis(trajectories)
#
#     path1 = _plot_trajectory(trajectories,reporter,plot_temp,info)
#
#     # 1. 准备位移分布图
#     fig1, ax1 = plt.subplots(**plot_temp.temp_fig())
#
#     # 选择不同的时间间隔
#     lags = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     colors = plt.cm.viridis(np.linspace(0, 1, len(lags)))
#
#     for i, lag in enumerate(lags):
#         all_displacements = []
#
#         for traj in trajectories:
#             if len(traj) > lag:
#                 # 计算指定时间间隔的位移
#                 displacements = traj[lag:] - traj[:-lag]
#                 radial = np.linalg.norm(displacements, axis=1)
#                 all_displacements.append(radial)
#
#         if all_displacements:
#             all_displacements = np.concatenate(all_displacements)
#
#             # 计算直方图
#             hist, bin_edges = np.histogram(all_displacements, bins=bins, density=True)
#             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#             # 过滤零值以在对数图上显示
#             mask = hist > 0
#             ax1.plot(bin_centers[mask], hist[mask], '-',
#                      color=colors[i], alpha=0.7,
#                      label=f'Δt={lag}')
#     if log_x:
#         ax1.set_xscale('log')
#         ax1.set_xlabel('Displace Δx (log scale)')
#     else:
#         ax1.set_xlabel('Displace Δx')
#     if log_y:
#         ax1.set_yscale('log')
#         ax1.set_ylabel('p.d.f. (log scale)')
#     else:
#         ax1.set_ylabel('p.d.f.')
#     # ax1.set_title('Displacement distribution at different time intervals')
#     ax1.legend()
#     ax1.grid(True, which="both", ls="-", alpha=0.3)
#     fig1.suptitle(f"Displacement distribution analysis of {info} | Original step size kurtosis: {kurtosis_value:.1f}", fontsize=7)
#
#     path2 = reporter.add_figure_by_data(fig1,f"{info}_displace_distribution", title=f"Diffusion analyze of {info}")
#
#
#     # 2. MSD分析
#     fig2, ax2 = plt.subplots(1, 1, **plot_temp.temp_fig())
#     fig3, ax3 = plt.subplots(1, 1, **plot_temp.temp_fig())
#
#     # 计算MSD
#     lags, msd, std = _calculate_msd(trajectories, n_lags=n_lag, max_lag=max_lag)
#     # lags = np.arange(1, len(msd) + 1)
#
#     # 线性坐标图
#     # ax2.errorbar(lags, msd, yerr=std, fmt='-', alpha=0.7)
#     ax2.errorbar(lags, msd, fmt='-', alpha=0.7)
#     ax2.set_xlabel('Time interval (Δt)')
#     ax2.set_ylabel('MSD')
#     ax2.set_title('MSD-Δt')
#     ax2.grid(True)
#
#     # 对数坐标图
#     mask = (msd > 0) & (lags > 0)
#     log_msd = np.log(msd[mask])
#     log_lags = np.log(lags[mask])
#
#     ax3.plot(log_lags, log_msd, 'o-', alpha=0.5)
#     ax3.set_xlabel('ln(Δt)')
#     ax3.set_ylabel('ln(MSD)')
#     ax3.set_title('ln MSD-Δt')
#     ax3.grid(True)
#
#     # 拟合幂律指数
#     if len(log_lags) > 1:
#         slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_msd)
#         ax3.plot(log_lags, slope * log_lags + intercept, 'r--',
#                  label=f'fit <y=x$^α$>: α={slope:.2f}, R²={r_value ** 2:.2f}')
#         ax3.legend()
#         alpha = slope
#         if r_value < 0.95 or True:
#             slope, intercept, r_value, p_value, std_err = linregress(lags[mask], log_msd)
#             ax3.plot(log_lags, slope * np.exp(log_lags) + intercept, '--', color='#FF7F0E',
#                      label=f'fit<y=α$^x$>: α={slope:.2f}, R²={r_value ** 2:.2f}')
#             ax3.legend()
#     else:
#         alpha = None
#
#
#     # # 4. 扩散类型判断
#     # if slope < 0.9:
#     #     diffusion_type = "亚扩散 (subdiffusion)"
#     # elif 0.9 <= slope <= 1.1:
#     #     diffusion_type = "正常扩散 (Brownian motion)"
#     # elif slope > 1.1:
#     #     if excess_kurtosis > 5:  # 显著的重尾特征
#     #         diffusion_type = "莱维飞行 (Lévy flight)"
#     #     else:
#     #         diffusion_type = "超扩散 (superdiffusion)"
#     # else:
#     #     diffusion_type = "混合扩散类型"
#
#     # 添加分析结果到图表
#     # fig2.suptitle(f"MSD analyze of {info}  | slope: {alpha:.1f}", fontsize=7)
#
#     plt.tight_layout()
#     path3 = reporter.add_figure_by_data(fig2,f"diffusion_MSD_{info}")
#     path4 = reporter.add_figure_by_data(fig3,f"diffusion_MSD_ln_{info}")
#
#     reporter(f"[+] -'{info}' MSD alpha={alpha:.1f}; kurtosis={kurtosis_value:.1f}").md().log()
#
#     # plt.show()
#     plt.close(fig1)
#     plt.close(fig2)
#     plt.close(fig3)
#     return  kurtosis_value, alpha, [path1, path2, path3,path4]


def combine_analyze_diffusion(trajectories, reporter: Reporter, plot_temp: PlotTemplate, info: str,
                              max_lag=1000, n_lag=None, bins=50, log_x=False, log_y=False, frame=None):
    """
    分析扩散轨迹的扩散类型，并将四个子图整合到一张图中（2x2布局）
    """
    if not trajectories:
        raise ValueError("轨迹列表不能为空")

    # 1. 峰度分析
    kurtosis_value, excess_kurtosis = _analyze_kurtosis(trajectories)

    # 2. 创建2x2的子图布局
    fig = plt.figure(**plot_temp.temp_fig(1.2, 1.2))  # 稍大的画布
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # 位移分布
    ax2 = fig.add_subplot(gs[0, 1])  # MSD-t
    ax3 = fig.add_subplot(gs[1, 0])  # log(MSD)-log(t)
    ax4 = fig.add_subplot(gs[1, 1])  # 新增：log(MSD)-t

    # ========== 子图1: 位移分布 ==========
    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lags)))

    for i, lag in enumerate(lags):
        all_displacements = []
        for traj in trajectories:
            if len(traj) > lag:
                displacements = traj[lag:] - traj[:-lag]
                radial = np.linalg.norm(displacements, axis=1)
                all_displacements.append(radial)

        if all_displacements:
            all_displacements = np.concatenate(all_displacements)
            hist, bin_edges = np.histogram(all_displacements, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = hist > 0
            ax1.plot(bin_centers[mask], hist[mask], '-', color=colors[i], alpha=0.7, label=f'Δt={lag}')

    ax1.set_xlabel('Displace Δx (log scale)' if log_x else 'Displace Δx')
    ax1.set_ylabel('p.d.f. (log scale)' if log_y else 'p.d.f.')
    if log_x: ax1.set_xscale('log')
    if log_y: ax1.set_yscale('log')
    ax1.legend(fontsize=6)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.set_title('Displacement Distribution', fontsize=7)

    # ========== 子图2: MSD-t ==========
    lags, msd, std = _calculate_msd(trajectories, n_lags=n_lag, max_lag=max_lag)
    ax2.errorbar(lags, msd, fmt='-', alpha=0.7)
    ax2.set_xlabel('Time interval (Δt)')
    ax2.set_ylabel('MSD')
    # ax2.set_title('MSD vs Δt', fontsize=8)
    ax2.grid(True)

    # ========== 子图3: log(MSD)-log(t) ==========
    mask = (msd > 0) & (lags > 0)
    log_msd = np.log(msd[mask])
    log_lags = np.log(lags[mask])

    ax3.plot(log_lags, log_msd, 'o-', alpha=0.5)
    ax3.set_xlabel('ln(Δt)')
    ax3.set_ylabel('ln(MSD)')
    # ax3.set_title('Power Law Fit (ln MSD vs ln Δt)', fontsize=8)
    ax3.grid(True)

    # 幂律拟合
    if len(log_lags) > 1:
        slope, intercept, r_value, _, _ = linregress(log_lags, log_msd)
        ax3.plot(log_lags, slope * log_lags + intercept, 'r--',
                 label=f'α={slope:.2f}, R²={r_value ** 2:.2f}')
        ax3.legend(fontsize=6)

    # ========== 子图4: log(MSD)-t (新增) ==========
    ax4.plot(lags[mask], log_msd, 's-', color='green', alpha=0.6)
    ax4.set_xlabel('Δt')
    ax4.set_ylabel('ln(MSD)')
    # ax4.set_title('Exponential Fit (ln MSD vs Δt)', fontsize=8)
    ax4.grid(True)

    # 指数拟合（用于区分扩散类型）
    if len(lags[mask]) > 1:
        slope, intercept, r_value, _, _ = linregress(lags[mask], log_msd)
        ax4.plot(lags[mask], slope * lags[mask] + intercept, '--', color='orange',
                 label=f'k={slope:.2e}, R²={r_value ** 2:.2f}')
        ax4.legend(fontsize=6)

    # ========== 整体调整 ==========
    fig.suptitle(
        f"Diffusion Analysis: {info}\n"
        f"Kurtosis={kurtosis_value:.1f} | "
        f"Power Law α={slope:.2f} | "
        f"Exp Coeff k={slope:.2e}",
        fontsize=9, y=1.02
    )
    plt.tight_layout()
    if frame is not None:
        plot_temp.add_frame_counter(fig=fig, text=f"step={frame}")

    # 保存合并后的图像
    path = reporter.add_figure_by_data(
        fig,
        f"combined_diffusion_analysis_{info}",
        title=f"Combined Diffusion Analysis: {info}",
        describe=f"Kurtosis={kurtosis_value:.1f}, Power law exponent={slope:.2f}"
    )
    plt.close(fig)
    return path


# # 辅助函数 (需提前定义)
# def _calculate_msd(trajectories, max_lag=None):
#     """计算均方位移"""
#     if not trajectories:
#         return np.array([]), np.array([])
#
#     min_length = min(len(traj) for traj in trajectories)
#     max_lag = min(max_lag or min_length // 2, min_length - 1)
#
#     msd_ensemble = np.zeros(max_lag)
#     msd_squared = np.zeros(max_lag)
#     count = np.zeros(max_lag, dtype=int)
#
#     for traj in trajectories:
#         n = len(traj)
#         for lag in range(1, max_lag + 1):
#             start_indices = np.arange(0, n - lag)
#             displacements = traj[start_indices + lag] - traj[start_indices]
#             sq_displacements = np.sum(displacements ** 2, axis=1)
#
#             if len(sq_displacements) > 0:
#                 msd_ensemble[lag - 1] += np.sum(sq_displacements)
#                 msd_squared[lag - 1] += np.sum(sq_displacements ** 2)
#                 count[lag - 1] += len(sq_displacements)
#
#     msd = msd_ensemble / count
#     msd_variance = (msd_squared / count) - (msd_ensemble / count) ** 2
#     std_msd = np.sqrt(np.maximum(msd_variance, 0))
#
#     return msd, std_msd


def _calculate_msd(trajectories, max_lag=None, n_lags=None, log_spaced=True, min_lag=1):
    """
    计算均方位移 (Mean Squared Displacement)

    参数:
        trajectories: 轨迹列表，每个轨迹形状为(n_steps, 2)
        max_lag: 最大时间延迟
        n_lags: 要计算的延迟点数量 (None表示计算所有可能的延迟)
        log_spaced: 是否对数均匀分布延迟点 (True) 或线性均匀分布 (False)
        min_lag: 最小时间延迟 (默认为1)

    返回:
        lags: 实际计算的延迟点
        msd: 平均MSD值
        std_msd: MSD的标准差
    """
    if not trajectories:
        return np.array([]), np.array([]), np.array([])

    # 确定最大可能延迟
    min_length = min(len(traj) for traj in trajectories)
    max_possible_lag = min_length - 1

    # 设置默认max_lag
    if max_lag is None:
        max_lag = max_possible_lag
    else:
        max_lag = min(max_lag, max_possible_lag)

    # 确定要计算的延迟点
    if n_lags is None:
        # 计算所有可能的延迟
        lags = np.arange(min_lag, max_lag + 1)
    else:
        if log_spaced:
            # 对数均匀分布的延迟点
            lags = np.unique(np.logspace(
                np.log10(min_lag),
                np.log10(max_lag),
                num=n_lags,
                base=10
            ).astype(int))
        else:
            # 线性均匀分布的延迟点
            lags = np.unique(np.linspace(
                min_lag,
                max_lag,
                num=n_lags
            ).astype(int))

    # 确保延迟点在有效范围内
    lags = lags[(lags >= min_lag) & (lags <= max_lag)]
    n_lags_actual = len(lags)

    # 初始化存储数组
    msd_ensemble = np.zeros(n_lags_actual)
    msd_squared = np.zeros(n_lags_actual)
    count = np.zeros(n_lags_actual, dtype=int)

    # 遍历所有轨迹
    for traj in trajectories:
        n = len(traj)

        # 遍历所有选定的延迟
        for i, lag in enumerate(lags):
            # 确保有足够的数据点
            if n > lag:
                start_indices = np.arange(0, n - lag)
                displacements = traj[start_indices + lag] - traj[start_indices]
                sq_displacements = np.sum(displacements ** 2, axis=1)

                if len(sq_displacements) > 0:
                    msd_ensemble[i] += np.sum(sq_displacements)
                    msd_squared[i] += np.sum(sq_displacements ** 2)
                    count[i] += len(sq_displacements)

    # 计算平均MSD和标准差
    msd = np.zeros(n_lags_actual)
    std_msd = np.zeros(n_lags_actual)

    for i in range(n_lags_actual):
        if count[i] > 0:
            msd[i] = msd_ensemble[i] / count[i]
            if count[i] > 1:
                variance = (msd_squared[i] / count[i]) - (msd_ensemble[i] / count[i]) ** 2
                std_msd[i] = np.sqrt(max(variance, 0))

    return lags, msd, std_msd



def _analyze_kurtosis(trajectories):
    """分析位移分布的峰度"""
    displacements = []

    for traj in trajectories:
        steps = traj[1:] - traj[:-1]
        dists = np.linalg.norm(steps, axis=1)
        displacements.append(dists)

    displacements = np.concatenate(displacements)
    kurt = kurtosis(displacements, fisher=False)
    excess_kurtosis = kurt - 3

    return kurt, excess_kurtosis

def _plot_trajectory(trajectory_list, reporter:Reporter, plot_temp: PlotTemplate, info="Track_show"):
    fig = plt.figure(**plot_temp.temp_fig())
    for trajectory in trajectory_list:
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, linewidth=0.5)
        # plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', label='Start')
        # plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', label='End')
        # plt.title(f"{title}\n(α={alpha}, Steps={len(trajectory)})")
        # plt.xlabel("X"); plt.ylabel("Y")
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axis('off')
    # plt.show()
    rtn = reporter.add_figure_by_data(fig, f"diffusion_analyze_track_plot_{info}")
    plt.close(fig)
    return rtn


